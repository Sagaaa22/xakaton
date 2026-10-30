"""
RAG (Retrieval-Augmented Generation) engine over `rules.json`.
- Splits rules into fragments preserving exact rule IDs.
- Retrieves relevant fragments using hybrid BM25 + char n-gram TF-IDF + cross-lingual expansion.
- Strictly verifies that every returned rule ID exists in `rules.json`.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas import CitedRule


# Cross-lingual gaming & support term expansion (kk / uz / en / slang -> ru rule concepts)
CROSS_LINGUAL_EXPANSIONS: Dict[str, str] = {
    # Kazakh (kk)
    "балағат": "оскорбления игроков чат неадекватное мат",
    "боқта": "оскорбления игроков голосовой чат мат",
    "шеше": "оскорбление упоминание родных родителей",
    "анасын": "оскорбление упоминание родных родителей",
    "алдап": "нонрп обман мошенничество сделка обмен имущество скам",
    "алаяқ": "нонрп обман мошенничество скам",
    "көлік": "автомобиль транспорт машина обмен",
    "өлтірді": "убийство без причины дм нанесение урона",
    "баннан": "обжалование наказаний апелляция разбан блокировка",
    "шығару": "разбан снятие блокировки апелляция",
    "ақша": "донат возврат денежных средств платеж чек",
    "қайтару": "возврат денежных средств донат refund",
    "ұрланды": "безопасность аккаунта взлом восстановление пароля",
    "бұзылды": "взлом аккаунта восстановление доступа",
    "қате": "технические сбои вылеты краши ошибки",
    "қалай": "справка игровые команды меню навигатор",
    "қайда": "навигатор gps работы команды",
    # Uzbek (uz)
    "haqorat": "оскорбления игроков чат неадекватное мат",
    "so'k": "оскорбления игроков чат мат",
    "sokdi": "оскорбления игроков чат мат",
    "ona": "оскорбление упоминание родных",
    "aldab": "нонрп обман мошенничество сделка обмен скам",
    "firibgar": "нонрп обман мошенничество скам",
    "o'ldirdi": "убийство без причины дм нанесение урона",
    "oldirdi": "убийство без причины дм нанесение урона",
    "bandan": "обжалование наказаний апелляция разбан блокировка",
    "chiqaring": "разбан снятие блокировки апелляция",
    "nohaq": "не согласен с наказанием апелляция разбан",
    "pul": "донат возврат денежных средств платеж",
    "qaytarish": "возврат денежных средств донат refund",
    "o'g'irlandi": "безопасность аккаунта взлом восстановление",
    "xato": "технические сбои ошибки обжалование",
    "qanday": "справка игровые команды меню навигатор",
    "qayerda": "навигатор gps работы команды",
    # English (en) & Gaming slang
    "insult": "оскорбления игроков чат унижение",
    "insulted": "оскорбления игроков чат унижение",
    "toxic": "оскорбления неадекватное поведение чат",
    "mother": "упоминание оскорбление родных",
    "parents": "упоминание оскорбление родных",
    "killed": "убийство без причины дм deathmatch",
    "dm": "дм deathmatch убийство без причины",
    "db": "дб driveby таран убийство транспортом машиной",
    "sk": "ск spawnkill убийство на спавне",
    "tk": "тк teamkill убийство членов своей фракции",
    "pg": "пг powergaming супергерой",
    "mg": "мг metagaming ooc информация",
    "rk": "рк revengekills возвращение на место гибели",
    "nonrp": "нонрп поведение сотрудник гос структур коп обман",
    "nrp": "нонрп поведение сотрудник гос структур коп",
    "cop": "сотрудник гос структур мвд полиция арест розыск",
    "arrested": "арест без рп отыгровок сотрудник гос структур",
    "scam": "нонрп обман при сделках и обмене имущество",
    "scammed": "нонрп обман при сделках и обмене имущество",
    "fraud": "мошенничество обман при сделках",
    "cheat": "использование читов вредоносного по стороннее",
    "cheater": "использование читов спидхак аим телепорт",
    "speedhack": "читы спидхак speedhack стороннее по",
    "aimbot": "читы аим aim стороннее по",
    "exploit": "багоюз использование недоработок мода дюп",
    "dupe": "багоюз клонирование предметов дюп",
    "unban": "порядок обжалования наказаний апелляция разбан блокировка",
    "banned": "обжалование наказаний блокировка бан апелляция",
    "appeal": "обжалование наказаний апелляция разбан",
    "refund": "правила возврата денежных средств донат",
    "donate": "зачисление донат валюты задержки платежей чек",
    "payment": "платежи донат квитанция чек оплаты",
    "hacked": "безопасность аккаунта взлом восстановление пароля",
    "2fa": "восстановление доступа двухфакторная аутентификация",
    "crash": "технические сбои вылеты краши пропажа имущества",
    "bug": "технические сбои баги пропажа имущества",
    "gps": "справка базовым игровым системам командам навигатор",
    "command": "справка базовым игровым системам командам",
    "commands": "справка базовым игровым системам командам",
    "licenses": "лицензии команда /lic паспорт /pass",
    "фрапс": "видеозапись доказательства /time",
    "откат": "видеозапись доказательства /time",
    "дюп": "багоюз клонирование предметов валюты дюп",
    "спидхак": "использование читов speedhack стороннее по",
    "аим": "использование читов aim стороннее по",
    "гм": "использование читов godmode бессмертие",
    "коп": "сотрудник гос структур мвд арест розыск nonrp cop",
    "мусор": "сотрудник гос структур полиция оскорбление",
    "вирты": "покупка продажа игровой валюты имущество",
    "разбан": "обжалование наказаний апелляция блокировка",
    "разбаньте": "обжалование наказаний апелляция блокировка",
    "верните": "возврат денежных средств донат пропажа имущества",
    "обзыва": "оскорбления игроков чат неадекватное поведение",
    "обозва": "оскорбления игроков чат неадекватное поведение",
    "матери": "оскорбления игроков чат мат",
}


@dataclass
class RuleRecord:
    id: str
    section: str
    category: str
    title: str
    text: str
    punishment: str


@dataclass
class RuleFragment:
    chunk_id: str
    rule_id: str
    section: str
    category: str
    title: str
    quote: str
    full_text: str
    punishment: str
    tokens: List[str]
    char_ngrams: Dict[str, float]


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s/'ʻ’-]", " ", text.lower())
    raw_tokens = [t.strip("'-") for t in cleaned.split() if len(t.strip("'-")) >= 2]
    expanded = list(raw_tokens)
    for tok in raw_tokens:
        for prefix, expansion in CROSS_LINGUAL_EXPANSIONS.items():
            if tok == prefix or (len(prefix) >= 4 and tok.startswith(prefix)):
                expanded.extend(expansion.split())
        # Russian basic stem normalization (strip common endings for better recall)
        if len(tok) >= 6:
            expanded.append(tok[:5])
    return expanded


def _char_ngrams(text: str, n: int = 3) -> Dict[str, float]:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return {cleaned: 1.0} if cleaned else {}
    counts: Dict[str, float] = {}
    for i in range(len(cleaned) - n + 1):
        gram = cleaned[i : i + n]
        counts[gram] = counts.get(gram, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(k, 0.0) for k, val in a.items())


class RulesRAGEngine:
    """
    Chunking, retrieval, and strict ID verification over `rules.json`.
    Guarantees that no non-existent rule ID can ever be returned.
    """

    def __init__(self, rules_path: Path) -> None:
        self.rules_path = rules_path
        self.rules_by_id: Dict[str, RuleRecord] = {}
        self.fragments: List[RuleFragment] = []
        self.idf: Dict[str, float] = {}
        self.avg_dl: float = 1.0
        self._load_and_index()

    @property
    def valid_rule_ids(self) -> Set[str]:
        return set(self.rules_by_id.keys())

    def is_valid_rule_id(self, rule_id: str) -> bool:
        return str(rule_id).strip() in self.rules_by_id

    def verify_rule_ids(self, rule_ids: List[str]) -> List[str]:
        """Filter list of rule IDs, keeping ONLY verified IDs that exist in rules.json."""
        verified: List[str] = []
        for rid in rule_ids:
            clean_id = str(rid).strip()
            if clean_id in self.rules_by_id and clean_id not in verified:
                verified.append(clean_id)
        return verified

    def _parse_rules_json(self, raw_data: Any) -> List[RuleRecord]:
        records: List[RuleRecord] = []
        if isinstance(raw_data, dict):
            if "rules" in raw_data and isinstance(raw_data["rules"], list):
                items = raw_data["rules"]
            elif "sections" in raw_data and isinstance(raw_data["sections"], list):
                items = []
                for sec in raw_data["sections"]:
                    sec_name = sec.get("title") or sec.get("name") or ""
                    for r in sec.get("rules", []):
                        if isinstance(r, dict):
                            r = dict(r)
                            r.setdefault("section", sec_name)
                            items.append(r)
            else:
                # Dict mapping rule_id -> text or dict
                items = []
                for k, v in raw_data.items():
                    if str(k).startswith("_"):
                        continue
                    if isinstance(v, str):
                        items.append({"id": str(k), "title": f"Пункт {k}", "text": v})
                    elif isinstance(v, dict):
                        d = dict(v)
                        d.setdefault("id", str(k))
                        items.append(d)
        elif isinstance(raw_data, list):
            items = raw_data
        else:
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(
                item.get("id")
                or item.get("rule_id")
                or item.get("code")
                or item.get("number")
                or ""
            ).strip()
            if not rid:
                continue
            section = str(item.get("section") or item.get("chapter") or "").strip()
            category = str(item.get("category") or "").strip()
            title = str(item.get("title") or item.get("name") or f"Пункт {rid}").strip()
            text = str(
                item.get("text") or item.get("description") or item.get("content") or title
            ).strip()
            punishment = str(
                item.get("punishment") or item.get("penalty") or item.get("sanction") or ""
            ).strip()
            records.append(
                RuleRecord(
                    id=rid,
                    section=section,
                    category=category,
                    title=title,
                    text=text,
                    punishment=punishment,
                )
            )
        return records

    def _split_into_fragments(self, rule: RuleRecord) -> List[str]:
        """Split rule text into semantic sentence fragments while keeping each quote faithful."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rule.text) if s.strip()]
        if not sentences:
            return [rule.text]
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= 260:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks or [rule.text]

    def _load_and_index(self) -> None:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        raw_data = json.loads(self.rules_path.read_text(encoding="utf-8"))
        records = self._parse_rules_json(raw_data)

        self.rules_by_id = {r.id: r for r in records}
        self.fragments = []

        df: Dict[str, int] = {}
        total_len = 0

        for rule in records:
            text_chunks = self._split_into_fragments(rule)
            for idx, chunk_quote in enumerate(text_chunks):
                index_text = f"{rule.id} {rule.section} {rule.title} {chunk_quote} {rule.punishment}"
                tokens = _tokenize(index_text)
                total_len += len(tokens)
                for tok in set(tokens):
                    df[tok] = df.get(tok, 0) + 1
                frag = RuleFragment(
                    chunk_id=f"{rule.id}#{idx}",
                    rule_id=rule.id,
                    section=rule.section,
                    category=rule.category,
                    title=rule.title,
                    quote=chunk_quote,
                    full_text=rule.text,
                    punishment=rule.punishment,
                    tokens=tokens,
                    char_ngrams=_char_ngrams(f"{rule.title} {chunk_quote}"),
                )
                self.fragments.append(frag)

        n_docs = max(1, len(self.fragments))
        self.avg_dl = total_len / n_docs if n_docs else 1.0
        self.idf = {
            tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def retrieve(
        self,
        query_text: str,
        category: Optional[str] = None,
        mentioned_rule_ids: Optional[List[str]] = None,
        top_k: int = 3,
        min_score: float = 0.18,
    ) -> Tuple[List[CitedRule], List[str], List[str], bool]:
        """
        Retrieve relevant rule fragments for a ticket.
        Returns:
            (cited_rules, valid_mentioned_ids, unknown_mentioned_ids, has_reliable_confirmation)
        """
        mentioned_rule_ids = mentioned_rule_ids or []
        valid_mentioned: List[str] = []
        unknown_mentioned: List[str] = []

        for rid in mentioned_rule_ids:
            clean_id = str(rid).strip()
            if self.is_valid_rule_id(clean_id):
                if clean_id not in valid_mentioned:
                    valid_mentioned.append(clean_id)
            else:
                if clean_id and clean_id not in unknown_mentioned:
                    unknown_mentioned.append(clean_id)

        q_tokens = _tokenize(query_text)
        # Expand query with cross-lingual mappings sochar n-grams also match Russian rules
        expanded_q_text = f"{query_text} {' '.join(q_tokens)}"
        q_ngrams = _char_ngrams(expanded_q_text)

        k1 = 1.5
        b = 0.75

        best_by_rule: Dict[str, Tuple[float, RuleFragment]] = {}

        for frag in self.fragments:
            # BM25 lexical score
            doc_len = max(1, len(frag.tokens))
            tf_map: Dict[str, int] = {}
            for t in frag.tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            bm25 = 0.0
            for qt in q_tokens:
                if qt in tf_map:
                    tf = tf_map[qt]
                    idf = self.idf.get(qt, 0.5)
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_dl))
                    bm25 += idf * ((tf * (k1 + 1.0)) / denom)

            # Normalize BM25 roughly to [0, 1] range
            bm25_norm = min(1.0, bm25 / 12.0)
            char_sim = _cosine_dict(q_ngrams, frag.char_ngrams)

            score = 0.65 * bm25_norm + 0.35 * char_sim

            # Category alignment bonus (only when query already has some lexical/semantic overlap
            # or when category matches strongly)
            if category and frag.category == category and score > 0.05:
                score += 0.18
                # Prefer foundational X.1 rule when specific sub-rule triggers (like relatives/politics) are absent
                if frag.rule_id.endswith(".1"):
                    score += 0.08
                elif frag.rule_id == "2.2" and not any(
                    w in expanded_q_text for w in ("родн", "мать", "маму", "родител", "шеше", "анасын", "mother", "parents")
                ):
                    score -= 0.08

            # Explicit valid rule ID mention boost
            if frag.rule_id in valid_mentioned:
                score += 0.45

            score = min(1.0, round(score, 4))

            prev = best_by_rule.get(frag.rule_id)
            if prev is None or score > prev[0]:
                best_by_rule[frag.rule_id] = (score, frag)

        ranked = sorted(best_by_rule.values(), key=lambda x: x[0], reverse=True)

        cited_rules: List[CitedRule] = []
        for score, frag in ranked[:top_k]:
            if score < min_score:
                continue
            # Double-check rule ID existence against source of truth
            if not self.is_valid_rule_id(frag.rule_id):
                continue
            cited_rules.append(
                CitedRule(
                    id=frag.rule_id,
                    section=frag.section,
                    title=frag.title,
                    quote=frag.quote,
                    punishment=frag.punishment,
                    relevance_score=round(score, 3),
                )
            )

        has_reliable = (
            len(cited_rules) > 0
            and cited_rules[0].relevance_score >= 0.25
            and len(unknown_mentioned) == 0
        )
        return cited_rules, valid_mentioned, unknown_mentioned, has_reliable
