"""
Taxonomy manager and supervised category/priority classifier trained on `tickets_train.jsonl.gz`.
Supports both explicit labels and labels extracted from administrator responses.
Never uses validation data for training.
"""
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag import CROSS_LINGUAL_EXPANSIONS, RulesRAGEngine, _tokenize
from app.schemas import ExtractedFacts


class CategorySpec:
    def __init__(
        self,
        cid: str,
        name: str,
        description: str = "",
        default_priority: str = "medium",
        needs_human_default: bool = True,
        related_rule_prefixes: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> None:
        self.id = cid
        self.name = name
        self.description = description
        self.default_priority = default_priority
        self.needs_human_default = needs_human_default
        self.related_rule_prefixes = related_rule_prefixes or []
        self.keywords = [k.lower() for k in (keywords or [])]


class TaxonomyManager:
    """Loads and validates categories and priorities from `taxonomy.json`."""

    def __init__(self, taxonomy_path: Path) -> None:
        self.taxonomy_path = taxonomy_path
        self.categories: Dict[str, CategorySpec] = {}
        self.priorities: List[str] = []
        self.priority_criteria: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        raw = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))

        raw_cats = raw.get("categories", []) if isinstance(raw, dict) else raw
        if isinstance(raw_cats, dict):
            for cid, val in raw_cats.items():
                if isinstance(val, dict):
                    self.categories[cid] = CategorySpec(
                        cid=cid,
                        name=str(val.get("name") or cid),
                        description=str(val.get("description") or ""),
                        default_priority=str(val.get("default_priority") or "medium"),
                        needs_human_default=bool(val.get("needs_human_default", True)),
                        related_rule_prefixes=list(val.get("related_rule_prefixes", [])),
                        keywords=list(val.get("keywords", [])),
                    )
                else:
                    self.categories[cid] = CategorySpec(cid=cid, name=str(val))
        elif isinstance(raw_cats, list):
            for item in raw_cats:
                if isinstance(item, str):
                    self.categories[item] = CategorySpec(cid=item, name=item)
                elif isinstance(item, dict):
                    cid = str(item.get("id") or item.get("name") or "").strip()
                    if not cid:
                        continue
                    self.categories[cid] = CategorySpec(
                        cid=cid,
                        name=str(item.get("name") or cid),
                        description=str(item.get("description") or ""),
                        default_priority=str(item.get("default_priority") or "medium"),
                        needs_human_default=bool(item.get("needs_human_default", True)),
                        related_rule_prefixes=list(item.get("related_rule_prefixes", [])),
                        keywords=list(item.get("keywords", [])),
                    )

        raw_prios = raw.get("priorities", []) if isinstance(raw, dict) else []
        if isinstance(raw_prios, list):
            for p in raw_prios:
                if isinstance(p, str):
                    self.priorities.append(p)
                elif isinstance(p, dict):
                    pid = str(p.get("id") or p.get("name") or "").strip()
                    if pid:
                        self.priorities.append(pid)
                        self.priority_criteria[pid] = str(p.get("criteria") or "")
        elif isinstance(raw_prios, dict):
            for pid, pval in raw_prios.items():
                self.priorities.append(str(pid))
                if isinstance(pval, dict):
                    self.priority_criteria[str(pid)] = str(pval.get("criteria") or "")
                elif isinstance(pval, str):
                    self.priority_criteria[str(pid)] = pval

        if not self.priorities:
            self.priorities = ["critical", "high", "medium", "low"]

    @property
    def valid_categories(self) -> Set[str]:
        return set(self.categories.keys())

    @property
    def valid_priorities(self) -> Set[str]:
        return set(self.priorities)


def _extract_features(text: str, kind: str = "") -> List[str]:
    tokens = _tokenize(text)
    feats = list(tokens)
    # Bigrams
    for i in range(len(tokens) - 1):
        feats.append(f"{tokens[i]}_{tokens[i+1]}")
    # Kind feature
    if kind:
        feats.append(f"__kind_{kind.lower().strip()}")
    # Character 4-grams of original lower text for morphological robustness
    clean = re.sub(r"\s+", " ", text.lower()).strip()
    for i in range(0, max(0, len(clean) - 3), 2):
        feats.append(f"__c4_{clean[i:i+4]}")
    return feats


class TicketClassifier:
    """
    Supervised Multinomial NB + Taxonomy Hybrid Classifier for ticket category and priority.
    Trains on `tickets_train.jsonl.gz` (explicit labels + labels extracted from admin replies).
    """

    def __init__(
        self,
        taxonomy: TaxonomyManager,
        rag: RulesRAGEngine,
        train_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.rag = rag
        self.train_path = train_path
        self.weights_path = weights_path

        self.cat_doc_counts: Dict[str, int] = {}
        self.cat_token_counts: Dict[str, Dict[str, float]] = {}
        self.cat_total_tokens: Dict[str, float] = {}
        self.vocab: Set[str] = set()

        self.prio_doc_counts: Dict[str, int] = {}
        self.prio_token_counts: Dict[str, Dict[str, float]] = {}
        self.prio_total_tokens: Dict[str, float] = {}

        self.is_trained: bool = False
        self.trained_samples_count: int = 0

        if weights_path and weights_path.exists():
            try:
                self.load_weights(weights_path)
            except Exception:
                pass

        if not self.is_trained and train_path and train_path.exists():
            self.train_from_jsonl(train_path, save_to=weights_path)

    def extract_labels_from_record(self, rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract (category, priority) from a training record.
        Uses explicit fields if available; otherwise extracts them from administrator replies
        and rule citations in `messages`.
        """
        cat = rec.get("category") or rec.get("label")
        prio = rec.get("priority")

        if isinstance(cat, str) and cat in self.taxonomy.valid_categories:
            valid_cat: Optional[str] = cat
        else:
            valid_cat = None

        if isinstance(prio, str) and prio in self.taxonomy.valid_priorities:
            valid_prio: Optional[str] = prio
        else:
            valid_prio = None

        if valid_cat and valid_prio:
            return valid_cat, valid_prio

        # Inspect admin replies in messages
        messages = rec.get("messages", [])
        admin_texts: List[str] = []
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role") or m.get("author") or "").lower()
                    if role in ("admin", "support", "moderator", "operator", "assistant"):
                        admin_texts.append(str(m.get("text") or ""))
        if rec.get("admin_reply"):
            admin_texts.append(str(rec["admin_reply"]))

        combined_admin = " ".join(admin_texts)
        lower_admin = combined_admin.lower()

        # 1. Check if admin reply explicitly mentions a taxonomy category ID
        if not valid_cat:
            for cid in self.taxonomy.valid_categories:
                if cid.lower() in lower_admin:
                    valid_cat = cid
                    break

        # 2. Check rule IDs cited by administrator and map via rules.json to category
        cited_rids = re.findall(
            r"(?:п\.|пункт\w*|правил\w*|rule|band\w*)\s*№?\s*(\d{1,3}\.\d{1,3})|(\d{1,3}\.\d{1,3})\s*-?\s*band",
            lower_admin,
        )
        flat_rids = [a or b for a, b in cited_rids if (a or b)]
        if not valid_cat and flat_rids:
            for rid in flat_rids:
                if rid in self.rag.rules_by_id:
                    rule_cat = self.rag.rules_by_id[rid].category
                    if rule_cat in self.taxonomy.valid_categories:
                        valid_cat = rule_cat
                        break
                # Or match via related_rule_prefixes in taxonomy
                for cid, cspec in self.taxonomy.categories.items():
                    if any(rid.startswith(pref) for pref in cspec.related_rule_prefixes):
                        valid_cat = cid
                        break
                if valid_cat:
                    break

        # 3. Extract priority from admin reply or infer from rule/category
        if not valid_prio:
            for pid in self.taxonomy.valid_priorities:
                if re.search(rf"\b{re.escape(pid.lower())}\b", lower_admin):
                    valid_prio = pid
                    break

        if not valid_prio and valid_cat:
            if "2.2" in flat_rids or "2.3" in flat_rids:
                valid_prio = "high" if "high" in self.taxonomy.valid_priorities else None
            if not valid_prio:
                valid_prio = self.taxonomy.categories[valid_cat].default_priority

        return valid_cat, valid_prio

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
            open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
        else:
            open_fn = lambda p: open(p, "r", encoding="utf-8")

        with open_fn(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def train_from_jsonl(self, train_path: Path, save_to: Optional[Path] = None) -> int:
        """Train classifier on `tickets_train.jsonl.gz` records."""
        records = self._read_jsonl(train_path)
        self.cat_doc_counts = {c: 0 for c in self.taxonomy.valid_categories}
        self.cat_token_counts = {c: {} for c in self.taxonomy.valid_categories}
        self.cat_total_tokens = {c: 0.0 for c in self.taxonomy.valid_categories}

        self.prio_doc_counts = {p: 0 for p in self.taxonomy.valid_priorities}
        self.prio_token_counts = {p: {} for p in self.taxonomy.valid_priorities}
        self.prio_total_tokens = {p: 0.0 for p in self.taxonomy.valid_priorities}
        self.vocab = set()

        trained = 0
        for rec in records:
            cat, prio = self.extract_labels_from_record(rec)
            if not cat:
                continue
            messages = rec.get("messages", [])
            player_texts = [
                str(m.get("text") or "")
                for m in messages
                if isinstance(m, dict)
                and str(m.get("role") or "player").lower() in ("player", "user", "client", "reporter")
            ]
            if not player_texts and rec.get("text"):
                player_texts = [str(rec["text"])]
            text = " ".join(player_texts).strip()
            if not text:
                continue

            kind = str(rec.get("kind") or "complaint")
            feats = _extract_features(text, kind=kind)
            trained += 1

            self.cat_doc_counts[cat] = self.cat_doc_counts.get(cat, 0) + 1
            c_map = self.cat_token_counts.setdefault(cat, {})
            for ft in feats:
                self.vocab.add(ft)
                c_map[ft] = c_map.get(ft, 0.0) + 1.0
                self.cat_total_tokens[cat] = self.cat_total_tokens.get(cat, 0.0) + 1.0

            if prio and prio in self.taxonomy.valid_priorities:
                self.prio_doc_counts[prio] = self.prio_doc_counts.get(prio, 0) + 1
                p_map = self.prio_token_counts.setdefault(prio, {})
                for ft in feats:
                    p_map[ft] = p_map.get(ft, 0.0) + 1.0
                    self.prio_total_tokens[prio] = self.prio_total_tokens.get(prio, 0.0) + 1.0

        # Also seed with taxonomy description & keywords so unseen categories still have priors
        for cid, cspec in self.taxonomy.categories.items():
            seed_text = f"{cspec.name} {cspec.description} {' '.join(cspec.keywords * 3)}"
            seed_feats = _extract_features(seed_text)
            c_map = self.cat_token_counts.setdefault(cid, {})
            self.cat_doc_counts[cid] = self.cat_doc_counts.get(cid, 0) + 2
            for ft in seed_feats:
                self.vocab.add(ft)
                c_map[ft] = c_map.get(ft, 0.0) + 1.5
                self.cat_total_tokens[cid] = self.cat_total_tokens.get(cid, 0.0) + 1.5

        self.trained_samples_count = trained
        self.is_trained = True

        if save_to:
            self.save_weights(save_to)
        return trained

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trained_samples_count": self.trained_samples_count,
            "cat_doc_counts": self.cat_doc_counts,
            "cat_token_counts": self.cat_token_counts,
            "cat_total_tokens": self.cat_total_tokens,
            "prio_doc_counts": self.prio_doc_counts,
            "prio_token_counts": self.prio_token_counts,
            "prio_total_tokens": self.prio_total_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_weights(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Verify categories match current taxonomy
        if set(payload.get("cat_doc_counts", {}).keys()) != self.taxonomy.valid_categories:
            return
        self.trained_samples_count = int(payload.get("trained_samples_count", 0))
        self.cat_doc_counts = payload["cat_doc_counts"]
        self.cat_token_counts = payload["cat_token_counts"]
        self.cat_total_tokens = payload["cat_total_tokens"]
        self.prio_doc_counts = payload["prio_doc_counts"]
        self.prio_token_counts = payload["prio_token_counts"]
        self.prio_total_tokens = payload["prio_total_tokens"]
        self.vocab = set()
        for cmap in self.cat_token_counts.values():
            self.vocab.update(cmap.keys())
        self.is_trained = True

    def classify(
        self,
        text: str,
        kind: str,
        facts: ExtractedFacts,
        is_spam: bool,
        is_duplicate: bool,
    ) -> Tuple[str, str, float, List[str]]:
        """
        Predict (category, priority, confidence, reasons) strictly within `taxonomy.json`.
        """
        reasons: List[str] = []
        lower_text = text.lower()
        expanded_tokens = _tokenize(text)
        expanded_str = f"{lower_text} {' '.join(expanded_tokens)}"

        # Fast-path for pure spam / meaningless flood
        if is_spam and "spam_offtopic" in self.taxonomy.valid_categories:
            # Make sure it's not an urgent complaint with repeated exclamation marks
            has_real_complaint = any(
                kw in expanded_str
                for kw in ("оскорб", "обман", "чит", "взлом", "донат", "забанил", "убил", "краш")
            )
            if not has_real_complaint:
                cat = "spam_offtopic"
                prio = self.taxonomy.categories[cat].default_priority
                if prio not in self.taxonomy.valid_priorities:
                    prio = self.taxonomy.priorities[-1]
                reasons.append(f"Категория '{cat}' назначена по признакам спама/оффтопа")
                return cat, prio, 0.92, reasons

        feats = _extract_features(text, kind=kind)
        vocab_size = max(1, len(self.vocab))
        total_docs = max(1, sum(self.cat_doc_counts.values()))

        # 1. Compute Multinomial NB log-likelihoods per category
        log_scores: Dict[str, float] = {}
        for cid, cspec in self.taxonomy.categories.items():
            doc_cnt = self.cat_doc_counts.get(cid, 1)
            prior = math.log((doc_cnt + 1.0) / (total_docs + len(self.taxonomy.categories)))
            denom = self.cat_total_tokens.get(cid, 0.0) + vocab_size * 0.5
            t_map = self.cat_token_counts.get(cid, {})

            ll = prior
            for ft in feats:
                cnt = t_map.get(ft, 0.0)
                ll += math.log((cnt + 0.5) / denom)

            # Normalize by feature count so longer texts don't dominate scale
            norm_ll = ll / max(1, len(feats))

            # Keyword & domain criteria bonus from taxonomy.json
            kw_hits = 0
            for kw in cspec.keywords:
                if kw in expanded_str:
                    kw_hits += 1
            bonus = min(0.45, kw_hits * 0.11)

            # Mentioned valid rules prefix alignment
            for r_id in facts.valid_mentioned_rules:
                if any(r_id.startswith(pref) for pref in cspec.related_rule_prefixes):
                    bonus += 0.25

            # Kind alignment bonus
            if kind == "appeal" and cid == "ban_appeal":
                bonus += 0.15
            elif kind == "bug" and cid == "technical_bug":
                bonus += 0.15
            elif kind == "question" and cid == "gameplay_question" and kw_hits > 0:
                bonus += 0.15

            # Prevent non-spam tickets from accidentally falling into spam_offtopic
            if cid == "spam_offtopic" and not is_spam and not is_duplicate and kw_hits == 0:
                bonus -= 0.35

            log_scores[cid] = norm_ll + bonus

        # Convert log_scores via softmax with temperature into probabilities
        max_s = max(log_scores.values())
        temp = 0.12
        exp_scores = {
            cid: math.exp((s - max_s) / temp) for cid, s in log_scores.items()
        }
        sum_exp = sum(exp_scores.values()) or 1.0
        probs = {cid: val / sum_exp for cid, val in exp_scores.items()}

        ranked_cats = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_prob = ranked_cats[0]
        second_prob = ranked_cats[1][1] if len(ranked_cats) > 1 else 0.0

        # Calibrate confidence in [0.45, 0.98]
        margin = best_prob - second_prob
        confidence = min(0.98, max(0.45, round(0.55 + 0.43 * margin, 2)))
        if facts.unknown_mentioned_rules:
            confidence = min(confidence, 0.58)
        elif not facts.has_evidence and best_cat in ("insult_toxicity", "rp_violation", "scam_fraud"):
            confidence = min(confidence, 0.78)

        reasons.append(
            f"Классификатор определил категорию '{best_cat}' ({self.taxonomy.categories[best_cat].name}) с уверенностью {confidence:.2f}"
        )

        # 2. Determine Priority strictly according to taxonomy.json criteria
        best_prio = self._determine_priority(best_cat, expanded_str, facts, is_spam, is_duplicate, reasons)

        return best_cat, best_prio, confidence, reasons

    def _determine_priority(
        self,
        category: str,
        expanded_str: str,
        facts: ExtractedFacts,
        is_spam: bool,
        is_duplicate: bool,
        reasons: List[str],
    ) -> str:
        cspec = self.taxonomy.categories[category]
        prio = cspec.default_priority
        if prio not in self.taxonomy.valid_priorities:
            prio = self.taxonomy.priorities[len(self.taxonomy.priorities) // 2]

        # Apply explicit priority escalation/de-escalation criteria from taxonomy.json
        if is_spam or category == "spam_offtopic":
            if "low" in self.taxonomy.valid_priorities:
                prio = "low"
                reasons.append("Приоритет 'low': спам или оффтоп-обращение")
                return prio

        # Critical criteria: account_recovery, cheating_exploits
        if category in ("cheating_exploits", "account_recovery") and "critical" in self.taxonomy.valid_priorities:
            prio = "critical"
            reasons.append(f"Приоритет 'critical' по критериям таксономии для категории '{category}'")
            return prio

        # High criteria for insult_toxicity: parents/relatives insult or hate speech (rules 2.2, 2.3)
        if category == "insult_toxicity":
            severe_insult = any(
                w in expanded_str
                for w in ("родн", "мать", "маму", "родител", "нациз", "розжиг", "шеше", "анасын", "mother", "parents", "2.2", "2.3")
            )
            if severe_insult and "high" in self.taxonomy.valid_priorities:
                prio = "high"
                reasons.append("Приоритет повышен до 'high': упоминание родных или розжиг (п. 2.2 / 2.3)")
                return prio

        reasons.append(f"Приоритет '{prio}' назначен согласно критериям taxonomy.json для '{category}'")
        return prio
