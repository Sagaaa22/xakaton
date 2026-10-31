"""
RAG (Retrieval-Augmented Generation) engine over `rules.json`.
- Splits rules into fragments preserving exact rule IDs from `rules.json`.
- Supports official Grand Mobile handout `rules.json` structure (category docs + `points[].ref`)
  as well as flat rule lists.
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
    "балағат": "оскорбления игроков чат неадекватное мат 2.1 2.4",
    "боқта": "оскорбления игроков голосовой чат мат 2.1",
    "шеше": "оскорбление упоминание родных родителей 2.6",
    "анасын": "оскорбление упоминание родных родителей 2.6",
    "алдап": "обман мошенничество сделка обмен имущество скам 4.1 4.2",
    "алаяқ": "обман мошенничество скам имущество 4.1 4.2",
    "көлік": "автомобиль транспорт машина обмен",
    "өлтірді": "убийство без причины дм нанесение урона 5.3 5.4",
    "баннан": "обжалование наказаний апелляция разбан блокировка администратор 3.4",
    "шығару": "разбан снятие блокировки апелляция 3.4",
    "ақша": "донат возврат денежных средств платеж чек",
    "қайтару": "возврат денежных средств донат refund платеж",
    "ұрланды": "безопасность аккаунта взлом передача 1.4",
    "бұзылды": "взлом аккаунта восстановление доступа 1.4",
    "қате": "технические сбои вылеты краши ошибки баг застрял 3.1",
    "қалай": "вопрос механика игры команды меню навигатор",
    "қайда": "навигатор gps работы команды вопрос",
    # Uzbek (uz)
    "haqorat": "оскорбления игроков чат неадекватное мат 2.1 2.4",
    "so'k": "оскорбления игроков чат мат 2.1",
    "sokdi": "оскорбления игроков чат мат 2.1",
    "ona": "оскорбление упоминание родных 2.6",
    "aldab": "обман мошенничество сделка обмен скам 4.1 4.2",
    "firibgar": "обман мошенничество скам 4.1 4.2",
    "o'ldirdi": "убийство без причины дм нанесение урона 5.3 5.4",
    "oldirdi": "убийство без причины дм нанесение урона 5.3 5.4",
    "bandan": "обжалование наказаний апелляция разбан блокировка 3.4",
    "chiqaring": "разбан снятие блокировки апелляция 3.4",
    "nohaq": "не согласен с наказанием апелляция администратор 3.4",
    "pul": "донат возврат денежных средств платеж",
    "qaytarish": "возврат денежных средств донат refund",
    "o'g'irlandi": "безопасность аккаунта взлом восстановление 1.4",
    "xato": "технические сбои ошибки обжалование",
    "qanday": "вопрос игровые команды меню навигатор",
    "qayerda": "навигатор gps работы команды вопрос",
    # English (en) & Gaming slang
    "insult": "оскорбления игроков чат унижение 2.1 2.4",
    "insulted": "оскорбления игроков чат унижение 2.1 2.4",
    "toxic": "оскорбления неадекватное поведение чат 2.1",
    "mother": "упоминание оскорбление родных 2.6",
    "parents": "упоминание оскорбление родных 2.6",
    "killed": "убийство без причины дм deathmatch 5.3 5.4",
    "dm": "дм deathmatch убийство нанесение урона без причины 5.3 5.4",
    "db": "дб driveby таран убийство транспортом машиной 5.3",
    "sk": "ск spawnkill убийство на спавне 5.3",
    "tk": "тк teamkill убийство членов своей фракции 5.3",
    "pg": "пг powergaming преувеличение возможностей персонажа 5.3",
    "mg": "мг metagaming использование ooc информации в ic 5.4",
    "rk": "рк revengekills возвращение на место убийства 5.3",
    "nonrp": "нонрп поведение нарушение рп логики 5.3 5.4",
    "nrp": "нонрп поведение нарушение рп 5.3 5.4",
    "gz": "зелёная зона green zone нанесение урона убийство GZ 1.1",
    "зз": "зелёная зона green zone нанесение урона убийство GZ 1.1",
    "cop": "сотрудник гос структур мвд полиция арест розыск ПГО 5.1",
    "arrested": "арест без рп отыгровок сотрудник гос структур ПГО 5.1",
    "scam": "обман мошенничество при сделках и обмене имущество 4.1 4.2",
    "scammed": "обман мошенничество при сделках и обмене имущество 4.1 4.2",
    "fraud": "мошенничество обман при сделках 4.1 4.2",
    "cheat": "использование читов вредоносного стороннего по 5.2",
    "cheater": "использование читов спидхак аим телепорт 5.2",
    "speedhack": "читы спидхак speedhack стороннее по 5.2",
    "aimbot": "читы аим aim стороннее по 5.2",
    "exploit": "багоюз использование недоработок мода дюп 5.1",
    "dupe": "багоюз клонирование предметов дюп 5.1",
    "unban": "не согласен с наказанием жалоба на администратора апелляция разбан 3.4",
    "banned": "не согласен с наказанием жалоба на администратора бан апелляция 3.4",
    "appeal": "не согласен с наказанием жалоба на администратора апелляция разбан 3.4",
    "refund": "платеж донат пополнение баланс чек",
    "donate": "зачисление донат валюты задержки платежей чек",
    "payment": "платежи донат квитанция чек оплаты",
    "hacked": "безопасность аккаунта взлом передача 1.4",
    "2fa": "восстановление доступа аккаунт защита 1.4",
    "crash": "помощь застрял баг телепорт спавн 3.1",
    "bug": "помощь застрял баг недоработки мода 3.1 5.1",
    "stuck": "застрял текстуры помощь репорт 3.1",
    "gps": "вопрос по механике игры команды навигатор",
    "command": "вопрос по механике игры команды",
    "commands": "вопрос по механике игры команды",
    "фрапс": "видеозапись доказательства /time ЖБ 1.2.2",
    "откат": "видеозапись доказательства /time ЖБ 1.2.2",
    "дюп": "багоюз клонирование предметов валюты дюп 5.1",
    "спидхак": "использование читов speedhack стороннее по 5.2",
    "аим": "использование читов aim стороннее по 5.2",
    "гм": "использование читов godmode бессмертие 5.2",
    "рванка": "использование читов вредоносное по рванка 5.2",
    "софт": "использование читов стороннее по софт 5.2",
    "коп": "сотрудник гос структур мвд арест розыск ПГО 5.1",
    "вирты": "покупка продажа игровой валюты за реальные деньги 4.4",
    "рмт": "продажа покупка за реальные деньги реклама 2.3 4.4",
    "rmt": "продажа покупка за реальные деньги реклама 2.3 4.4",
    "разбан": "не согласен с наказанием жалоба на администратора 3.4",
    "разбаньте": "не согласен с наказанием жалоба на администратора 3.4",
    "верните": "возврат средств донат пропажа имущества",
    "обзыва": "оскорбления игроков чат неадекватное поведение 2.1",
    "обозва": "оскорбления игроков чат неадекватное поведение 2.1",
    "матери": "оскорбления игроков чат мат родных 2.1 2.6",
    "оск": "оскорбления игроков чат родных 2.1 2.4 2.6",
    "упом": "упоминание родных в оскорбительном характере 2.6",
    "скам": "обман мошенничество с имуществом 4.1 4.2",
    "кинул": "обман мошенничество сделка имущество 4.1 4.2",
    "застрял": "просьба о помощи застрял телепорт спавн репорт 3.1",
    "флип": "просьба о помощи перевернулся репорт 3.1",
    "оффтоп": "оффтоп и пустые сообщения в репорт 3.1",
}

# Prefix mapping for official handout `rules.json` category documents
DOC_ID_TO_PREFIX: Dict[int, str] = {
    30: "",      # Общие правила игры -> "1.1".."5.9"
    31: "ЖБ ",   # Правила жалоб -> "ЖБ 1.1.1".."ЖБ 1.3.1"
    32: "GZ ",   # Правила нахождения в Зелёной Зоне -> "GZ" and "GZ 1.1".."GZ 2.3"
    33: "ЛД ",   # Правила лидеров
    34: "ПГО ",  # Правила государственных организаций
    35: "ГВ ",   # Правила Войны семей
    36: "СМ ",   # Правила захвата цехов
    40: "ОПГ ",  # Правила криминальных организаций
    41: "БВ ",   # Правила войны за бизнес
    42: "ВЗФ ",  # Правила захвата AirDrop (ВЗФ)
}

# Mapping from canonical rule IDs to official taxonomy category codes
RULE_TO_TAXONOMY_CATEGORY: Dict[str, str] = {
    "2.1": "complaint.insult",
    "2.2": "complaint.insult",
    "2.4": "complaint.insult",
    "2.5": "complaint.insult",
    "2.6": "complaint.insult",
    "2.7": "complaint.insult",
    "3.2": "complaint.insult",
    "3.3": "complaint.insult",
    "5.1": "complaint.cheat",
    "5.2": "complaint.cheat",
    "5.3": "complaint.dm",
    "5.4": "complaint.dm",
    "5.5": "complaint.dm",
    "5.7": "complaint.dm",
    "GZ": "complaint.dm",
    "GZ 1.1": "complaint.dm",
    "GZ 1.2": "complaint.dm",
    "GZ 1.3": "complaint.dm",
    "GZ 2.1": "complaint.dm",
    "GZ 2.2": "complaint.dm",
    "GZ 2.3": "complaint.dm",
    "DM": "complaint.dm",
    "DB": "complaint.dm",
    "SK": "complaint.dm",
    "TK": "complaint.dm",
    "PG": "complaint.dm",
    "MG": "complaint.dm",
    "RK": "complaint.dm",
    "NonRP": "complaint.dm",
    "Mass DM": "complaint.dm",
    "Mass DM GZ": "complaint.dm",
    "Mass SK": "complaint.dm",
    "4.1": "complaint.scam",
    "4.2": "complaint.scam",
    "4.8": "complaint.scam",
    "4.9": "complaint.scam",
    "2.3": "complaint.rmt_ads",
    "4.3": "complaint.rmt_ads",
    "4.4": "complaint.rmt_ads",
    "4.5": "complaint.rmt_ads",
    "4.6": "complaint.rmt_ads",
    "4.7": "complaint.rmt_ads",
    "3.4": "complaint.admin",
    "ЖБ 1.3.1": "complaint.admin",
    "ЖБ 1.1.1": "complaint.invalid",
    "ЖБ 1.1.2": "complaint.invalid",
    "ЖБ 1.2.1": "complaint.invalid",
    "ЖБ 1.2.2": "complaint.invalid",
    "ЖБ 1.2.3": "complaint.invalid",
    "ЖБ 1.2.4": "complaint.invalid",
    "ЖБ 1.2.5": "complaint.invalid",
    "ЖБ 1.2.6": "complaint.invalid",
    "ЖБ 1.2.7": "complaint.invalid",
    "1.1": "report.account",
    "1.2": "report.account",
    "1.3": "report.account",
    "1.4": "report.account",
    "3.1": "report.offtopic",
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
    cleaned = re.sub(r"[^\w\s/'ʻ’.-]", " ", text.lower())
    raw_tokens = [t.strip("'.-") for t in cleaned.split() if len(t.strip("'.-")) >= 2]
    expanded = list(raw_tokens)
    for tok in raw_tokens:
        for prefix, expansion in CROSS_LINGUAL_EXPANSIONS.items():
            if tok == prefix or (len(prefix) >= 4 and tok.startswith(prefix)):
                expanded.extend(expansion.lower().split())
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


def _split_rule_body_and_punishment(raw_text: str) -> Tuple[str, str]:
    """
    In official `rules.json`, point texts often include sanctions after `|`, e.g.:
    `Запрещены оскорбления... | Музыка в войс — блокировка чата 20-40 минут.\n| Оскорбления — ...`
    Splits into (clean_rule_body, joined_punishment).
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    body_parts: List[str] = []
    punish_parts: List[str] = []
    for ln in lines:
        if "|" in ln:
            left, right = ln.split("|", 1)
            if left.strip():
                body_parts.append(left.strip())
            if right.strip():
                punish_parts.append(right.strip())
        else:
            body_parts.append(ln)
    body = " ".join(body_parts).strip() or raw_text.strip()
    punishment = " | ".join(punish_parts).strip()
    return body, punishment


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

    def normalize_rule_ref(self, raw_ref: str) -> Optional[str]:
        """
        Normalize a player- or metadata-mentioned rule reference (e.g. '2.1 ОП', '1.1 GZ', '1.2.2 ППЖ')
        to a canonical ID that exists in `self.rules_by_id`.
        Returns None if the referenced rule does not exist in `rules.json`.
        """
        if not raw_ref:
            return None
        s = re.sub(r"\s+", " ", str(raw_ref).strip())
        if s in self.rules_by_id:
            return s
        s_upper = s.upper()
        if s_upper in self.rules_by_id:
            return s_upper

        # Strip common suffixes like "ОП", "OP", "правил", "п."
        cleaned = re.sub(r"^(?:п\.|пункт\w*|правил\w*|rule|band\w*)\s*№?\s*", "", s, flags=re.IGNORECASE).strip()
        if cleaned in self.rules_by_id:
            return cleaned

        # Check "<num> ОП" / "<num> OP" -> "<num>"
        m_op = re.match(r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ОП|OP|О\.П\.)$", cleaned, flags=re.IGNORECASE)
        if m_op and m_op.group(1) in self.rules_by_id:
            return m_op.group(1)

        # Check "<num> GZ" / "<num> ЗЗ" / "GZ <num>" / "ЗЗ <num>"
        m_gz1 = re.match(r"^(?:GZ|ЗЗ)\s*(\d{1,2}\.\d{1,2})$", cleaned, flags=re.IGNORECASE)
        if m_gz1 and f"GZ {m_gz1.group(1)}" in self.rules_by_id:
            return f"GZ {m_gz1.group(1)}"
        m_gz2 = re.match(r"^(\d{1,2}\.\d{1,2})\s*(?:GZ|ЗЗ|ПЗЗ)$", cleaned, flags=re.IGNORECASE)
        if m_gz2 and f"GZ {m_gz2.group(1)}" in self.rules_by_id:
            return f"GZ {m_gz2.group(1)}"
        if cleaned.upper() in ("GZ", "ЗЗ", "GREEN ZONE") and "GZ" in self.rules_by_id:
            return "GZ"

        # Check "<num> ЖБ" / "<num> ППЖ" / "ЖБ <num>"
        m_jb1 = re.match(r"^(?:ЖБ|ППЖ)\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)$", cleaned, flags=re.IGNORECASE)
        if m_jb1 and f"ЖБ {m_jb1.group(1)}" in self.rules_by_id:
            return f"ЖБ {m_jb1.group(1)}"
        m_jb2 = re.match(r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*(?:ЖБ|ППЖ)$", cleaned, flags=re.IGNORECASE)
        if m_jb2 and f"ЖБ {m_jb2.group(1)}" in self.rules_by_id:
            return f"ЖБ {m_jb2.group(1)}"

        # Check section prefixes: ПГО, ЛД, ОПГ, БВ, ГВ, СМ, ВЗФ
        for pref in ("ПГО", "ЛД", "ОПГ", "БВ", "ГВ", "СМ", "ВЗФ"):
            m1 = re.match(rf"^{pref}\s*(\d{{1,2}}\.\d{{1,2}}(?:\.\d{{1,2}})?)$", cleaned, flags=re.IGNORECASE)
            if m1 and f"{pref} {m1.group(1)}" in self.rules_by_id:
                return f"{pref} {m1.group(1)}"
            m2 = re.match(rf"^(\d{{1,2}}\.\d{{1,2}}(?:\.\d{{1,2}})?)\s*{pref}$", cleaned, flags=re.IGNORECASE)
            if m2 and f"{pref} {m2.group(1)}" in self.rules_by_id:
                return f"{pref} {m2.group(1)}"

        # Check if 3-part number like "1.2.2" or "1.1.2" belongs to ЖБ
        if re.match(r"^\d+\.\d+\.\d+$", cleaned) and f"ЖБ {cleaned}" in self.rules_by_id:
            return f"ЖБ {cleaned}"

        # Check RP terms case-insensitively
        for k in self.rules_by_id:
            if k.lower() == cleaned.lower():
                return k

        return None

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

            # Official Grand Mobile handout format: category document with `points: [{ref, text}, ...]`
            if "points" in item and isinstance(item["points"], list):
                doc_id = item.get("id")
                doc_cat_name = str(item.get("category") or "").strip()
                prefix = DOC_ID_TO_PREFIX.get(doc_id, "") if isinstance(doc_id, int) else ""
                points = item["points"]

                if points:
                    for pt in points:
                        if not isinstance(pt, dict):
                            continue
                        ref = str(pt.get("ref") or "").strip()
                        raw_pt_text = str(pt.get("text") or "").strip()
                        if not ref or not raw_pt_text:
                            continue
                        rid = f"{prefix}{ref}".strip()
                        body, punishment = _split_rule_body_and_punishment(raw_pt_text)
                        tax_cat = RULE_TO_TAXONOMY_CATEGORY.get(rid, "")
                        if not tax_cat:
                            if prefix == "ЖБ ":
                                tax_cat = "complaint.invalid"
                            elif prefix == "GZ ":
                                tax_cat = "complaint.dm"
                            elif prefix == "ПГО " and ref.startswith("5."):
                                tax_cat = "complaint.dm"

                        title = f"{doc_cat_name} — п. {rid}"
                        rec = RuleRecord(
                            id=rid,
                            section=doc_cat_name,
                            category=tax_cat,
                            title=title,
                            text=body,
                            punishment=punishment,
                        )
                        records.append(rec)

                        # Also register shorthand "GZ" from point 1.1 of Doc 32 (referenced in taxonomy.json)
                        if doc_id == 32 and ref == "1.1":
                            records.append(
                                RuleRecord(
                                    id="GZ",
                                    section=doc_cat_name,
                                    category="complaint.dm",
                                    title=f"{doc_cat_name} (GZ)",
                                    text=body,
                                    punishment=punishment,
                                )
                            )
                else:
                    # Doc 43 ("RP термины и их описание") has empty `points` and bullet list in `text`
                    doc_text = str(item.get("text") or "")
                    for line in doc_text.splitlines():
                        m = re.match(r"^\*\s*\*\*([^*]+)\*\*\s*[—\-–]\s*(.+)$", line.strip())
                        if m:
                            term = m.group(1).strip()
                            defn = m.group(2).strip()
                            tax_cat = RULE_TO_TAXONOMY_CATEGORY.get(term, "complaint.dm")
                            records.append(
                                RuleRecord(
                                    id=term,
                                    section=doc_cat_name or "RP термины",
                                    category=tax_cat,
                                    title=f"RP термин: {term}",
                                    text=defn,
                                    punishment="",
                                )
                            )
                continue

            # Fallback: flat rule item format
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
            category = str(item.get("category") or RULE_TO_TAXONOMY_CATEGORY.get(rid, "")).strip()
            title = str(item.get("title") or item.get("name") or f"Пункт {rid}").strip()
            raw_text = str(
                item.get("text") or item.get("description") or item.get("content") or title
            ).strip()
            punishment = str(
                item.get("punishment") or item.get("penalty") or item.get("sanction") or ""
            ).strip()
            if not punishment and "|" in raw_text:
                raw_text, punishment = _split_rule_body_and_punishment(raw_text)
            records.append(
                RuleRecord(
                    id=rid,
                    section=section,
                    category=category,
                    title=title,
                    text=raw_text,
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
            elif len(current) + 1 + len(sent) <= 280:
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
        category_rule_ids: Optional[List[str]] = None,
        top_k: int = 3,
        min_score: float = 0.16,
    ) -> Tuple[List[CitedRule], List[str], List[str], bool]:
        """
        Retrieve relevant rule fragments for a ticket.
        Returns:
            (cited_rules, valid_mentioned_ids, unknown_mentioned_ids, has_reliable_confirmation)
        """
        mentioned_rule_ids = mentioned_rule_ids or []
        category_rule_ids = category_rule_ids or []
        valid_mentioned: List[str] = []
        unknown_mentioned: List[str] = []

        for rid in mentioned_rule_ids:
            norm_id = self.normalize_rule_ref(rid)
            if norm_id and self.is_valid_rule_id(norm_id):
                if norm_id not in valid_mentioned:
                    valid_mentioned.append(norm_id)
            else:
                clean_id = str(rid).strip()
                if clean_id and clean_id not in unknown_mentioned:
                    unknown_mentioned.append(clean_id)

        q_tokens = _tokenize(query_text)
        expanded_q_text = f"{query_text.lower()} {' '.join(q_tokens)}"
        q_ngrams = _char_ngrams(expanded_q_text)

        k1 = 1.5
        b = 0.75

        best_by_rule: Dict[str, Tuple[float, RuleFragment]] = {}

        for frag in self.fragments:
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

            bm25_norm = min(1.0, bm25 / 12.0)
            char_sim = _cosine_dict(q_ngrams, frag.char_ngrams)

            score = 0.65 * bm25_norm + 0.35 * char_sim

            # Category alignment bonus
            if category and (frag.category == category or frag.rule_id in category_rule_ids):
                if score > 0.03 or frag.rule_id in category_rule_ids:
                    score += 0.22
                # Prefer primary general rules over specialized event rules (ВЗФ, БВ, СМ, ГВ, ОПГ, ЛД)
                if not any(frag.rule_id.startswith(p) for p in ("ВЗФ ", "БВ ", "СМ ", "ГВ ", "ОПГ ", "ЛД ")):
                    score += 0.06
                if frag.rule_id in ("2.1", "5.2", "5.3", "4.1", "2.3", "3.4", "3.1", "1.4", "ЖБ 1.1.2"):
                    score += 0.05
                if frag.rule_id == "2.6" and any(
                    w in expanded_q_text for w in ("родн", "мать", "маму", "родител", "шеше", "анасын", "mother", "parents", "оск род", "упом род")
                ):
                    score += 0.18
            elif any(frag.rule_id.startswith(p) for p in ("ВЗФ ", "БВ ", "СМ ", "ГВ ", "ОПГ ", "ЛД ")):
                # Slight penalty for faction/event rules unless explicitly mentioned
                if frag.rule_id not in valid_mentioned:
                    score -= 0.08

            # Explicit valid rule ID mention boost
            if frag.rule_id in valid_mentioned:
                score += 0.48

            score = min(1.0, max(0.0, round(score, 4)))

            prev = best_by_rule.get(frag.rule_id)
            if prev is None or score > prev[0]:
                best_by_rule[frag.rule_id] = (score, frag)

        ranked = sorted(best_by_rule.values(), key=lambda x: x[0], reverse=True)

        cited_rules: List[CitedRule] = []
        for score, frag in ranked[:top_k]:
            if score < min_score:
                continue
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
            and cited_rules[0].relevance_score >= 0.22
            and len(unknown_mentioned) == 0
        )
        return cited_rules, valid_mentioned, unknown_mentioned, has_reliable
