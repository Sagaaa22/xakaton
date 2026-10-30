"""
Taxonomy manager and supervised category/priority/subcategory classifier trained on `tickets_train.jsonl.gz`.
Supports explicit labels, labels extracted from administrator responses, detailed violation subtypes
(cheats, DM/RDM, PG, MG, RK, DB, SK/TK, NonRP, trade scam, fraud, insults, bug abuse, tech issues, appeals),
and live Feedback Loop weight updates.
"""
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag import CROSS_LINGUAL_EXPANSIONS, RulesRAGEngine, _tokenize
from app.schemas import ExtractedFacts


# Detailed violation subcategories mapped to parent taxonomy categories
SUBCATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "cheats": {
        "parent": "cheating_exploits",
        "name": "Читы и стороннее ПО (Aim / SpeedHack / Teleport / GM)",
        "patterns": [r"\b(?:чит\w*|софт\w*|аим\w*|спидхак\w*|телепорт\w*|валхак\w*|гм\b|бессмерт\w*|cheat\w*|aimbot|speedhack|wallhack|hack\w*)\b"],
    },
    "bug_abuse": {
        "parent": "cheating_exploits",
        "name": "Багоюз и использование уязвимостей (Дюп)",
        "patterns": [r"\b(?:багоюз\w*|багаюз\w*|дюп\w*|текстур\w*\s+прошел|сквозь\s+текстур\w*|exploit\w*|dupe|bug\s*abuse)\b"],
    },
    "dm_rdm": {
        "parent": "rp_violation",
        "name": "DM / RDM (Убийство без RP-причины)",
        "patterns": [r"\b(?:дм\b|рдм\b|dm\b|rdm\b|deathmatch|убил\w*\s+без\s+причин\w*|расстрелял\w*\s+просто\s+так|өлтірді|o'ldirdi|killed\s+me\s+without\s+reason)\b"],
    },
    "db": {
        "parent": "rp_violation",
        "name": "DB (DriveBy — таран или убийство транспортом)",
        "patterns": [r"\b(?:дб\b|db\b|driveby|задавил\w*\s+машин\w*|сбил\w*\s+на\s+машин\w*|таран\w*)\b"],
    },
    "sk_tk": {
        "parent": "rp_violation",
        "name": "SK / TK (SpawnKill / TeamKill)",
        "patterns": [r"\b(?:ск\b|тк\b|sk\b|tk\b|spawn\s*kill|team\s*kill|на\s+спавн\w*\s+убил|своих\s+убива\w*)\b"],
    },
    "pg": {
        "parent": "rp_violation",
        "name": "PG (PowerGaming — воображение супергероя)",
        "patterns": [r"\b(?:пг\b|pg\b|power\s*gaming|павергейм\w*|кулаками\s+на\s+оружие|один\s+против\s+пятерых)\b"],
    },
    "mg": {
        "parent": "rp_violation",
        "name": "MG (MetaGaming — OOC информация в IC)",
        "patterns": [r"\b(?:мг\b|mg\b|meta\s*gaming|метагейм\w*|оос\s+инф\w*\s+в\s+ис)\b"],
    },
    "rk": {
        "parent": "rp_violation",
        "name": "RK (RevengeKill — возвращение на место смерти)",
        "patterns": [r"\b(?:рк\b|rk\b|revenge\s*kill|вернулся\s+отомстить|после\s+смерти\s+вернулся)\b"],
    },
    "nonrp_cop": {
        "parent": "rp_violation",
        "name": "NonRP поведение / NonRP Гос. сотрудник",
        "patterns": [r"\b(?:нонрп|нрп|nonrp|nrp|коп\w*|арест\w*\s+без\s+рп|розыск\s+без\s+причин\w*|уход\s+от\s+рп|орп\b)\b"],
    },
    "trade_scam": {
        "parent": "scam_fraud",
        "name": "Обман при сделках и обмене имуществом",
        "patterns": [r"\b(?:обмен\w*|сделк\w*|договорились|продал\s+за|кинул\w*\s+на\s+машин\w*|кинул\w*\s+при\s+обмен\w*|trade\s*scam|алдап\s+кетті|aldab\s+ketdi)\b"],
    },
    "fraud_ppiv": {
        "parent": "scam_fraud",
        "name": "Мошенничество / Покупка-продажа за реал (ППИВ)",
        "patterns": [r"\b(?:ппив\b|вирт\w*\s+за\s+реал|мошенник\w*|скам\w*|развел\w*|развод\w*|scam\w*|fraud\w*|firibgar|алаяқ)\b"],
    },
    "family_insult": {
        "parent": "insult_toxicity",
        "name": "Упоминание и оскорбление родных (п. 2.2)",
        "patterns": [r"\b(?:родн\w*|мать|маму|мамк\w*|отца|папу|родител\w*|шеше\w*|анасын|онанг\w*|mother|parents|family\s+insult)\b"],
    },
    "hate_speech": {
        "parent": "insult_toxicity",
        "name": "Межнациональная рознь и дискриминация (п. 2.3)",
        "patterns": [r"\b(?:розжиг\w*|нациз\w*|расиз\w*|нацист\w*)\b"],
    },
    "insult_chat": {
        "parent": "insult_toxicity",
        "name": "Оскорбления и неадекватное поведение в чате/войсе",
        "patterns": [r"\b(?:оскорб\w*|аскарб\w*|оскарб\w*|мат\b|материл\w*|обозвал\w*|токсик\w*|балағат\w*|боқта\w*|haqorat|so'k\w*|insult\w*|toxic)\b"],
    },
    "property_loss": {
        "parent": "technical_bug",
        "name": "Пропажа игрового имущества при тех. сбое",
        "patterns": [r"\b(?:пропал\w*|исчез\w*|слетел\w*\s+тюнинг|слетел\w*\s+багажник)\b"],
    },
    "tech_bug": {
        "parent": "technical_bug",
        "name": "Технические проблемы, вылеты и баги",
        "patterns": [r"\b(?:вылет\w*|краш\w*|лагает|фризит|не\s+заходит|черный\s+экран|ошибк\w*|crash\w*|bug\w*|қате|xato)\b"],
    },
    "ban_appeal": {
        "parent": "ban_appeal",
        "name": "Апелляция и обжалование наказания (Разбан)",
        "patterns": [r"\b(?:разбан\w*|забанил\w*|бан\b|амнисти\w*|обжалов\w*|деморган\w*|варн\w*|unban|appeal|баннан|bandan)\b"],
    },
    "billing_refund": {
        "parent": "billing_refund",
        "name": "Донат, платежи и возврат средств",
        "patterns": [r"\b(?:донат\w*|доннат\w*|возврат\w*|верните\s+деньг\w*|оплат\w*|чек\b|списал\w*|refund|donate|ақша|pul)\b"],
    },
    "account_recovery": {
        "parent": "account_recovery",
        "name": "Взлом аккаунта и восстановление доступа",
        "patterns": [r"\b(?:взлом\w*|украли\s+акк\w*|пароль|2fa|ппв\b|восстанов\w*\s+акк\w*|hacked|ұрланды|o'g'irlandi)\b"],
    },
    "gameplay_question": {
        "parent": "gameplay_question",
        "name": "Игровой вопрос по механикам и командам",
        "patterns": [r"\b(?:как\s+открыть|какой\s+команд\w*|где\s+найти|подскажите|қалай|қайда|qanday|how\s+to)\b"],
    },
    "spam_offtopic": {
        "parent": "spam_offtopic",
        "name": "Спам, флуд и нецелевые обращения",
        "patterns": [r"\b(?:спам|тест|алло|аууу)\b"],
    },
}

DEFAULT_SUBCATEGORY_FOR_CATEGORY: Dict[str, Tuple[str, str]] = {
    "cheating_exploits": ("cheats", "Читы и стороннее ПО"),
    "rp_violation": ("dm_rdm", "DM / RDM и нарушение RolePlay"),
    "scam_fraud": ("trade_scam", "Обман при сделках и мошенничество"),
    "insult_toxicity": ("insult_chat", "Оскорбления и токсичное поведение"),
    "technical_bug": ("tech_bug", "Технические проблемы и баги"),
    "ban_appeal": ("ban_appeal", "Апелляция на наказание"),
    "billing_refund": ("billing_refund", "Донат и возврат средств"),
    "account_recovery": ("account_recovery", "Восстановление доступа к аккаунту"),
    "gameplay_question": ("gameplay_question", "Игровой вопрос"),
    "spam_offtopic": ("spam_offtopic", "Спам / Дубликат / Оффтоп"),
}


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


TYPO_REPLACEMENTS = {
    "аскарб": "оскорб",
    "оскарб": "оскорб",
    "чиитер": "читер",
    "читерр": "читер",
    "багаюз": "багоюз",
    "разбаньти": "разбаньте",
    "доннат": "донат",
    "машын": "машин",
    "обмануу": "обману",
    "рдм": "дм убийство",
}


def _normalize_typos_for_features(text: str) -> str:
    lower = text.lower()
    for wrong, right in TYPO_REPLACEMENTS.items():
        if wrong in lower:
            lower = lower.replace(wrong, right)
    return lower


def _extract_features(text: str, kind: str = "") -> List[str]:
    norm_text = _normalize_typos_for_features(text)
    tokens = _tokenize(norm_text)
    feats = list(tokens)
    for i in range(len(tokens) - 1):
        feats.append(f"{tokens[i]}_{tokens[i+1]}")
    if kind:
        feats.append(f"__kind_{kind.lower().strip()}")
    clean = re.sub(r"\s+", " ", norm_text).strip()
    for i in range(0, max(0, len(clean) - 3), 2):
        feats.append(f"__c4_{clean[i:i+4]}")
    return feats


class TicketClassifier:
    """
    Supervised Multinomial NB + Taxonomy Hybrid Classifier for ticket category, subcategory, and priority.
    Trains on `tickets_train.jsonl.gz` and supports live Feedback Loop updates.
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
        self.feedback_updates_count: int = 0

        if weights_path and weights_path.exists():
            try:
                self.load_weights(weights_path)
            except Exception:
                pass

        if not self.is_trained and train_path and train_path.exists():
            self.train_from_jsonl(train_path, save_to=weights_path)

    def detect_subcategory(self, text: str, category: str) -> Tuple[str, str]:
        """
        Determine detailed violation subcategory (e.g., cheats, bug_abuse, dm_rdm, pg, mg,
        trade_scam, family_insult, etc.) aligned with the primary taxonomy category.
        """
        norm = _normalize_typos_for_features(text)
        # First check subcategories belonging to the predicted parent category
        for sub_id, spec in SUBCATEGORY_SPECS.items():
            if spec["parent"] == category:
                for pat in spec["patterns"]:
                    if re.search(pat, norm, re.IGNORECASE):
                        return sub_id, spec["name"]

        # Second check: if a specific subcategory pattern strongly matches anywhere
        for sub_id, spec in SUBCATEGORY_SPECS.items():
            for pat in spec["patterns"]:
                if re.search(pat, norm, re.IGNORECASE):
                    return sub_id, spec["name"]

        # Fallback to default subcategory for the parent category
        return DEFAULT_SUBCATEGORY_FOR_CATEGORY.get(
            category, (category, self.taxonomy.categories[category].name if category in self.taxonomy.categories else category)
        )

    def apply_feedback(
        self,
        text: str,
        kind: str,
        correct_category: Optional[str],
        correct_priority: Optional[str] = None,
        weight_boost: float = 3.0,
    ) -> bool:
        """
        Feedback Loop: reinforce classifier weights with administrator's verified correction.
        """
        if not text or not text.strip():
            return False
        updated = False
        feats = _extract_features(text, kind=kind or "complaint")

        if correct_category and correct_category in self.taxonomy.valid_categories:
            self.cat_doc_counts[correct_category] = self.cat_doc_counts.get(correct_category, 0) + int(weight_boost)
            c_map = self.cat_token_counts.setdefault(correct_category, {})
            for ft in feats:
                self.vocab.add(ft)
                c_map[ft] = c_map.get(ft, 0.0) + weight_boost
                self.cat_total_tokens[correct_category] = self.cat_total_tokens.get(correct_category, 0.0) + weight_boost
            updated = True

        if correct_priority and correct_priority in self.taxonomy.valid_priorities:
            self.prio_doc_counts[correct_priority] = self.prio_doc_counts.get(correct_priority, 0) + int(weight_boost)
            p_map = self.prio_token_counts.setdefault(correct_priority, {})
            for ft in feats:
                p_map[ft] = p_map.get(ft, 0.0) + weight_boost
                self.prio_total_tokens[correct_priority] = self.prio_total_tokens.get(correct_priority, 0.0) + weight_boost
            updated = True

        if updated:
            self.feedback_updates_count += 1
            if self.weights_path:
                try:
                    self.save_weights(self.weights_path)
                except Exception:
                    pass
        return updated

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

        if not valid_cat:
            for cid in self.taxonomy.valid_categories:
                if cid.lower() in lower_admin:
                    valid_cat = cid
                    break

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
                for cid, cspec in self.taxonomy.categories.items():
                    if any(rid.startswith(pref) for pref in cspec.related_rule_prefixes):
                        valid_cat = cid
                        break
                if valid_cat:
                    break

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
            "feedback_updates_count": self.feedback_updates_count,
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
        if set(payload.get("cat_doc_counts", {}).keys()) != self.taxonomy.valid_categories:
            return
        self.trained_samples_count = int(payload.get("trained_samples_count", 0))
        self.feedback_updates_count = int(payload.get("feedback_updates_count", 0))
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
        norm_text = _normalize_typos_for_features(text)
        expanded_tokens = _tokenize(norm_text)
        expanded_str = f"{norm_text} {' '.join(expanded_tokens)}"

        if is_spam and "spam_offtopic" in self.taxonomy.valid_categories:
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

        feats = _extract_features(norm_text, kind=kind)
        vocab_size = max(1, len(self.vocab))
        total_docs = max(1, sum(self.cat_doc_counts.values()))

        log_scores: Dict[str, float] = {}
        matched_keywords_by_cat: Dict[str, List[str]] = {}
        for cid, cspec in self.taxonomy.categories.items():
            doc_cnt = self.cat_doc_counts.get(cid, 1)
            prior = math.log((doc_cnt + 1.0) / (total_docs + len(self.taxonomy.categories)))
            denom = self.cat_total_tokens.get(cid, 0.0) + vocab_size * 0.5
            t_map = self.cat_token_counts.get(cid, {})

            ll = prior
            for ft in feats:
                cnt = t_map.get(ft, 0.0)
                ll += math.log((cnt + 0.5) / denom)

            norm_ll = ll / max(1, len(feats))

            kw_hits = []
            for kw in cspec.keywords:
                if kw in expanded_str:
                    kw_hits.append(kw)
            matched_keywords_by_cat[cid] = kw_hits
            bonus = min(0.45, len(kw_hits) * 0.11)

            for sub_id, sub_spec in SUBCATEGORY_SPECS.items():
                if sub_spec["parent"] == cid:
                    if any(re.search(pat, norm_text, re.IGNORECASE) for pat in sub_spec["patterns"]):
                        bonus += 0.30
                        break

            for r_id in facts.valid_mentioned_rules:
                if any(r_id.startswith(pref) for pref in cspec.related_rule_prefixes):
                    bonus += 0.25

            if kind == "appeal" and cid == "ban_appeal":
                bonus += 0.15
            elif kind == "bug" and cid == "technical_bug":
                bonus += 0.15
            elif kind == "question" and cid == "gameplay_question" and len(kw_hits) > 0:
                bonus += 0.15

            if cid == "spam_offtopic" and not is_spam and not is_duplicate and len(kw_hits) == 0:
                bonus -= 0.35

            log_scores[cid] = norm_ll + bonus

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

        margin = best_prob - second_prob
        confidence = min(0.98, max(0.45, round(0.55 + 0.43 * margin, 2)))
        if facts.unknown_mentioned_rules:
            confidence = min(confidence, 0.58)
        elif not facts.has_evidence and best_cat in ("insult_toxicity", "rp_violation", "scam_fraud"):
            confidence = min(confidence, 0.78)

        sub_id, sub_name = self.detect_subcategory(text, best_cat)
        kw_Triggers = matched_keywords_by_cat.get(best_cat, [])
        trigger_info = f" (триггеры: {', '.join(kw_Triggers[:4])})" if kw_Triggers else ""
        reasons.append(
            f"Классификатор определил категорию '{best_cat}' ({self.taxonomy.categories[best_cat].name}) "
            f"и подкатегорию '{sub_id}' ({sub_name}) с уверенностью {int(confidence*100)}%{trigger_info}"
        )
        if facts.slang_or_typos_normalized:
            reasons.append(f"Распознан игровой сленг/опечатки: {'; '.join(facts.slang_or_typos_normalized[:3])}")

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

        if is_spam or category == "spam_offtopic":
            if "low" in self.taxonomy.valid_priorities:
                prio = "low"
                reasons.append("Приоритет 'low': спам или оффтоп-обращение")
                return prio

        if category in ("cheating_exploits", "account_recovery") and "critical" in self.taxonomy.valid_priorities:
            prio = "critical"
            reasons.append(f"Приоритет 'critical' по критериям таксономии для категории '{category}'")
            return prio

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
