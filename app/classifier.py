"""
Taxonomy manager and supervised category/priority classifier trained on `tickets_train.jsonl.gz`.
Supports official Grand Mobile handout `taxonomy.json` (`code`, `kind`, `title`, `rules`, `examples`)
and `tickets_train.jsonl.gz` (extracting training labels from administrator responses, verdicts, and rule citations).
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


# Enriched domain keywords per official taxonomy category code
CATEGORY_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "complaint.insult": [
        "оск", "оскорб", "мат", "матер", "родн", "мать", "маму", "папу", "отца", "семью",
        "упом", "неадекват", "токсик", "униж", "розжиг", "наци", "капс", "caps", "флуд",
        "музык", "войс", "голосов", "чат", "обозвал", "обзывал", "послал", "шлюх", "даун",
        " клоун", "балағат", "боқта", "шеше", "анасын", "haqorat", "so'k", "sokdi", "ona",
        "insult", "insulted", "toxic", "mother", "parents", "2.1", "2.2", "2.4", "2.6", "3.2", "3.3",
    ],
    "complaint.cheat": [
        "чит", "софт", "аим", "aim", "спидхак", "speedhack", "телепорт", "рванк", "гм",
        "godmode", "бессмерт", "багоюз", "багаюз", "дюп", "сторонн", "вредоносн", "флай",
        "fly", "клео", "cleo", "автокликер", "макрос", "cheat", "cheater", "soft", "5.1", "5.2",
    ],
    "complaint.dm": [
        "дм", "dm", "дб", "db", "ск", "sk", "тк", "tk", "пг", "pg", "мг", "mg", "рк", "rk",
        "нонрп", "нрп", "nonrp", "nrp", "зз", "gz", "зелен", "зелён", "убил", "убийств",
        "урон", "стреля", "расстрел", "задмил", "задавил", "таран", "связал", "наручник",
        "арест", "розыск", "коп", "полиц", "мвд", "фсб", "уход от рп", "лив от рп", "орп",
        "без причины", "просто так убил", "өлтірді", "o'ldirdi", "oldirdi", "killed",
        "5.3", "5.4", "5.5", "5.7", "ПГО",
    ],
    "complaint.scam": [
        "обман", "скам", "scam", "кинул", "развод", "развел", "развёл", "мошеннич",
        "обмен", "сделк", "договор", "долг", "не вернул", "не отдал", "машин", "авто",
        "дом", "бизнес", "скин", "аксессуар", "доплат", "алдап", "алаяқ", "aldab",
        "firibgar", "scammed", "fraud", "4.1", "4.2", "4.8", "4.9",
    ],
    "complaint.rmt_ads": [
        "рмт", "rmt", "вирт", "реальн", "продаж", "покупк", "продам", "куплю", "реклам",
        "промокод", "тг", "телеграм", "дискорд", "канал", "стороннего проекта", "донат за",
        "рубл", "тенге", "сум", "2.3", "4.3", "4.4", "4.5", "4.6", "4.7",
    ],
    "complaint.admin": [
        "админ", "администратор", "не согласен", "бан", "забанил", "разбан", "обжалован",
        "апелляц", "посадил", "деморган", "мут", "варн", "warn", "опру", "опра", "откат от адм",
        "халатн", "блат", "превыш", "unban", "appeal", "banned", "баннан", "шығару",
        "bandan", "chiqaring", "nohaq", "3.4",
    ],
    "complaint.other": [
        "другое", "прочее", "слив", "форум", "никнейм", "фейк", "транслит", "помеха",
    ],
    "complaint.invalid": [
        "не по форме", "нет скрин", "без доказательств", "дубль", "дубликат", "ошибся разделом",
        "ЖБ 1.1.2", "ЖБ 1.2.7", "ППЖ",
    ],
    "report.question": [
        "как ", "где ", "какая команда", "какой командой", "сколько ", "когда ", "почему ",
        "можно ли", "что делать если", "подскажите", "вопрос", "устроиться", "лиценз",
        "паспорт", "gps", "навигатор", "промокод ввести", "қалай", "қайда", "qanday",
        "qayerda", "how ", "where ", "command",
    ],
    "report.help": [
        "помогите", "помощь", "хелп", "help", "yordam", "көмек", "застрял", "спавн",
        "респавн", "флип", "flip", "переверн", "телепорт", "тп ", "тпхните", "вытащ",
        "баг", "текстур", "вода", "воду", "рек", "упал", "провалил", "пропал транспорт",
        "пропала машина", "краш", "вылет", "забагался",
    ],
    "report.complaint": [
        "тут дм", "дмят", "читер", "софт", "рванка", "следите", "спек", "spec", "id ",
        "айди ", "стреляет", "оскорбляет", "таранит", "дбшит", "нонрп коп", "нарушает",
    ],
    "report.account": [
        "взлом", "взломали", "аккаунт", "пароль", "доступ", "забыл пароль", "украли аккаунт",
        "привязк", "почт", "аутентификатор", "2fa", "пропали деньги", "пропало имущество",
        "передан", "продан", "ұрланды", "бұзылды", "o'g'irlandi", "hacked", "1.4",
    ],
    "report.payment": [
        "донат", "оплат", "платеж", "платёж", "списал", "деньги не пришли", "не пришел донат",
        "не пришёл донат", "чек", "квитанц", "пополн", "коин", "gc", "возврат средств",
        "refund", "payment", "donate", "ақша", "төлем", "қайтару", "pul", "to'lov",
    ],
    "report.offtopic": [
        "привет", "ку", "йй", "алло", "вы тут", "дайте денег", "дайте миллион", "дайте админку",
        "проверка связи", "тест", "люблю вас", "лучшие", "спасибо", "оффтоп", "+",
    ],
}

DEFAULT_CATEGORY_PRIORITY: Dict[str, str] = {
    "report.account": "p0",
    "report.payment": "p0",
    "complaint.cheat": "p1",
    "complaint.scam": "p1",
    "complaint.rmt_ads": "p1",
    "complaint.admin": "p1",
    "complaint.insult": "p2",
    "complaint.dm": "p2",
    "complaint.other": "p2",
    "report.complaint": "p2",
    "report.help": "p2",
    "report.question": "p3",
    "report.offtopic": "p3",
    "complaint.invalid": "p3",
}


class CategorySpec:
    def __init__(
        self,
        cid: str,
        name: str,
        kind: str = "complaint",
        description: str = "",
        default_priority: str = "p2",
        needs_human_default: bool = True,
        related_rule_prefixes: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> None:
        self.id = cid
        self.name = name
        self.kind = kind
        self.description = description
        self.default_priority = default_priority
        self.needs_human_default = needs_human_default
        self.related_rule_prefixes = related_rule_prefixes or []
        combined_kw = list(keywords or []) + CATEGORY_DOMAIN_KEYWORDS.get(cid, [])
        self.keywords = list(dict.fromkeys(k.lower() for k in combined_kw if k))


class TaxonomyManager:
    """Loads and validates categories and priorities from `taxonomy.json`."""

    def __init__(self, taxonomy_path: Path) -> None:
        self.taxonomy_path = taxonomy_path
        self.categories: Dict[str, CategorySpec] = {}
        self.priorities: List[str] = []
        self.priority_titles: Dict[str, str] = {}
        self.priority_criteria: Dict[str, str] = {}
        self.needs_human_always: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        raw = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))

        if isinstance(raw, dict) and isinstance(raw.get("needs_human_always"), list):
            self.needs_human_always = {str(x).strip() for x in raw["needs_human_always"]}

        raw_cats = raw.get("categories", []) if isinstance(raw, dict) else raw
        if isinstance(raw_cats, dict):
            for cid, val in raw_cats.items():
                if isinstance(val, dict):
                    self.categories[cid] = CategorySpec(
                        cid=cid,
                        name=str(val.get("title") or val.get("name") or cid),
                        kind=str(val.get("kind") or ("report" if cid.startswith("report.") else "complaint")),
                        description=str(val.get("description") or ""),
                        default_priority=str(val.get("default_priority") or DEFAULT_CATEGORY_PRIORITY.get(cid, "p2")),
                        needs_human_default=bool(
                            val.get("needs_human_default", cid in self.needs_human_always or cid.startswith("complaint."))
                        ),
                        related_rule_prefixes=list(val.get("rules") or val.get("related_rule_prefixes") or []),
                        keywords=list(val.get("examples") or val.get("keywords") or []),
                    )
                else:
                    self.categories[cid] = CategorySpec(cid=cid, name=str(val))
        elif isinstance(raw_cats, list):
            for item in raw_cats:
                if isinstance(item, str):
                    self.categories[item] = CategorySpec(cid=item, name=item)
                elif isinstance(item, dict):
                    cid = str(item.get("code") or item.get("id") or item.get("name") or "").strip()
                    if not cid:
                        continue
                    kind = str(item.get("kind") or ("report" if cid.startswith("report.") else "complaint")).strip()
                    default_prio = str(
                        item.get("default_priority") or DEFAULT_CATEGORY_PRIORITY.get(cid, "p2")
                    )
                    needs_human_def = bool(
                        item.get(
                            "needs_human_default",
                            (cid in self.needs_human_always)
                            or (kind == "complaint" and cid != "complaint.invalid")
                            or cid in ("report.account", "report.payment", "report.complaint"),
                        )
                    )
                    self.categories[cid] = CategorySpec(
                        cid=cid,
                        name=str(item.get("title") or item.get("name") or cid),
                        kind=kind,
                        description=str(item.get("description") or ""),
                        default_priority=default_prio,
                        needs_human_default=needs_human_def,
                        related_rule_prefixes=list(item.get("rules") or item.get("related_rule_prefixes") or []),
                        keywords=list(item.get("examples") or item.get("keywords") or []),
                    )

        raw_prios = raw.get("priorities", []) if isinstance(raw, dict) else []
        if isinstance(raw_prios, list):
            for p in raw_prios:
                if isinstance(p, str):
                    self.priorities.append(p)
                elif isinstance(p, dict):
                    pid = str(p.get("code") or p.get("id") or p.get("name") or "").strip()
                    if pid:
                        self.priorities.append(pid)
                        self.priority_titles[pid] = str(p.get("title") or p.get("name") or pid)
                        self.priority_criteria[pid] = str(p.get("rule") or p.get("criteria") or "")
        elif isinstance(raw_prios, dict):
            for pid, pval in raw_prios.items():
                self.priorities.append(str(pid))
                if isinstance(pval, dict):
                    self.priority_titles[str(pid)] = str(pval.get("title") or pid)
                    self.priority_criteria[str(pid)] = str(pval.get("rule") or pval.get("criteria") or "")
                elif isinstance(pval, str):
                    self.priority_criteria[str(pid)] = pval

        if not self.priorities:
            self.priorities = ["p0", "p1", "p2", "p3"]

    @property
    def valid_categories(self) -> Set[str]:
        return set(self.categories.keys())

    @property
    def valid_priorities(self) -> Set[str]:
        return set(self.priorities)


def _extract_features(text: str, kind: str = "", rule_refs_raw: str = "") -> List[str]:
    combined = f"{text} {rule_refs_raw}".strip()
    tokens = _tokenize(combined)
    feats = list(tokens)
    for i in range(len(tokens) - 1):
        feats.append(f"{tokens[i]}_{tokens[i+1]}")
    if kind:
        feats.append(f"__kind_{kind.lower().strip()}")
    if rule_refs_raw:
        feats.append(f"__rref_{rule_refs_raw.lower().strip()[:16]}")
    clean = re.sub(r"\s+", " ", combined.lower()).strip()
    for i in range(0, max(0, min(280, len(clean)) - 3), 2):
        feats.append(f"__c4_{clean[i:i+4]}")
    return feats


class TicketClassifier:
    """
    Supervised Multinomial NB + Taxonomy Hybrid Classifier for ticket category and priority.
    Trains on `tickets_train.jsonl.gz` (explicit labels + labels extracted from admin replies/verdicts).
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
        Extract (category, priority) from a training or validation record.
        Uses explicit fields if available; otherwise derives them from `kind`, `verdict`,
        `violation`, `rule_refs_raw`, `images_count`, player messages, and administrator replies.
        """
        cat = rec.get("category") or rec.get("label")
        prio = rec.get("priority")

        valid_cat: Optional[str] = cat if (isinstance(cat, str) and cat in self.taxonomy.valid_categories) else None
        valid_prio: Optional[str] = prio if (isinstance(prio, str) and prio in self.taxonomy.valid_priorities) else None

        if valid_cat and valid_prio:
            return valid_cat, valid_prio

        kind = str(rec.get("kind") or "complaint").strip().lower()
        verdict = str(rec.get("verdict") or "").strip().lower()
        violation = str(rec.get("violation") or "").strip()
        rule_refs_raw = str(rec.get("rule_refs_raw") or "").strip()
        images_count = int(rec.get("images_count") or 0)

        messages = rec.get("messages", [])
        player_texts: List[str] = []
        admin_texts: List[str] = []
        msg_attachments = 0
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role") or m.get("author") or "player").lower()
                    txt = str(m.get("text") or "")
                    att = m.get("attachments")
                    if isinstance(att, int):
                        msg_attachments += max(0, att)
                    elif isinstance(att, list):
                        msg_attachments += len(att)
                    if role in ("admin", "support", "moderator", "operator", "assistant"):
                        admin_texts.append(txt)
                    elif role in ("player", "user", "client", "reporter"):
                        player_texts.append(txt)
        if rec.get("admin_reply"):
            admin_texts.append(str(rec["admin_reply"]))
        if not player_texts and rec.get("text"):
            player_texts.append(str(rec["text"]))

        player_str = " ".join(player_texts).strip()
        lower_player = player_str.lower()
        lower_admin = " ".join(admin_texts).lower()
        lower_rref = rule_refs_raw.lower()
        lower_viol = violation.lower()
        combined_clues = f"{lower_rref} {lower_player} {lower_viol}"

        has_evidence = (
            images_count > 0
            or msg_attachments > 0
            or bool(re.search(r"(?:<url>|https?://|imgur|yapx|youtu|vk\.com/video)", lower_player))
        )

        # Normalize any rule references in rule_refs_raw or violation
        norm_rref = self.rag.normalize_rule_ref(rule_refs_raw)
        norm_viol = self.rag.normalize_rule_ref(violation)

        if not valid_cat:
            if kind == "report" and "report.question" in self.taxonomy.valid_categories:
                valid_cat = self._label_report_category(lower_player, lower_admin)
            elif "complaint.insult" in self.taxonomy.valid_categories:
                valid_cat = self._label_complaint_category(
                    lower_player=lower_player,
                    lower_admin=lower_admin,
                    lower_rref=lower_rref,
                    lower_viol=lower_viol,
                    norm_rref=norm_rref,
                    norm_viol=norm_viol,
                    verdict=verdict,
                    has_evidence=has_evidence,
                    combined_clues=combined_clues,
                )
            else:
                # Fallback for custom/legacy taxonomies
                for cid, cspec in self.taxonomy.categories.items():
                    if any(kw in combined_clues for kw in cspec.keywords[:10]):
                        valid_cat = cid
                        break
                if not valid_cat and self.taxonomy.categories:
                    valid_cat = next(iter(self.taxonomy.categories.keys()))

        if not valid_prio and valid_cat:
            valid_prio = self._derive_priority_for_label(
                category=valid_cat,
                kind=kind,
                lower_player=lower_player,
                lower_rref=lower_rref,
                has_evidence=has_evidence,
            )

        return valid_cat, valid_prio

    def _label_report_category(self, lower_player: str, lower_admin: str) -> str:
        # 1. Payment / donate
        if any(
            w in lower_player
            for w in (
                "донат", "оплат", "платеж", "платёж", "списал", "не пришли деньги", "не пришел донат",
                "не пришёл донат", "чек", "квитанц", "пополн", "gc", "коин", "refund", "donate",
                "payment", "ақша", "төлем", "қайтару", "to'lov",
            )
        ):
            return "report.payment"

        # 2. Account / access / hack / lost property after login
        if any(
            w in lower_player
            for w in (
                "взлом", "пароль", "забыл пароль", "украли акк", "доступ к акк", "привязк",
                "аутентификатор", "2fa", "пропали деньги", "пропало имущество", "пропал аккаунт",
                "аккаунт заблок", "ұрланды", "бұзылды", "o'g'irlandi", "akkauntimni", "hacked",
            )
        ):
            return "report.account"

        # 3. Complaint against another player via report
        if any(
            w in lower_player
            for w in (
                "дмят", "тут дм", "меня дм", "читер", "софт", "рванка", "следите за", "спек ",
                "spec ", "таранит", "дбшит", "оскорбляет", "стреляет в", "нонрп коп", "нарушает",
            )
        ) or re.search(r"\b(?:id|айди|ид)\s*(?:\d+|<digits>)\b", lower_player):
            return "report.complaint"

        # 4. Question about game mechanics / commands / rules
        if (
            "?" in lower_player
            or re.search(
                r"\b(?:как|где|какая|какой|какие|сколько|когда|почему|зачем|можно\s+ли|что\s+делать|подскажите|қалай|қайда|неге|qanday|qayerda|how|where|which)\b",
                lower_player,
            )
        ):
            # Unless it's a direct plea for physical teleport/unstuck
            if not any(w in lower_player for w in ("застрял", "спавн", "флип", "вытащ", "телепортируй")):
                return "report.question"

        # 5. Request for help: stuck, bug, flip, spawn, teleport
        if any(
            w in lower_player
            for w in (
                "помог", "помощь", "хелп", "help", "yordam", "көмек", "застрял", "спавн",
                "респавн", "флип", "flip", "переверн", "телепорт", "тп ", "тпхн", "вытащ",
                "текстур", "баг", "в речку", "в воде", "в воду", "упал", "провалил", "машин",
                "авто", "краш", "вылет", "забагал",
            )
        ) or any(
            w in lower_admin
            for w in ("спешу на помощь", "иду на помощь", "yordamga", "помог ", "вытащил", "телепортировал")
        ):
            return "report.help"

        # 6. Offtopic / greetings / begging / "+" / short empty messages
        return "report.offtopic"

    def _label_complaint_category(
        self,
        lower_player: str,
        lower_admin: str,
        lower_rref: str,
        lower_viol: str,
        norm_rref: Optional[str],
        norm_viol: Optional[str],
        verdict: str,
        has_evidence: bool,
        combined_clues: str,
    ) -> str:
        # 1. Invalid complaint: no evidence attached
        if not has_evidence:
            return "complaint.invalid"

        # 2. Check complaint against administrator / appeal FIRST
        # (Players appealing a ban/mute/warn often write the rule they were punished for, e.g. 5.2 or 2.1, in rule_refs_raw)
        if (
            any(
                w in combined_clues
                for w in (
                    "3.4", "жалоба на адм", "администратор выдал", "админ выдал", "админ забанил",
                    "админ посадил", "админские", "не согласен", "запрос опры", "опру", "апел",
                    "пересмотр", "разбан", "обжалован", "дали бан", "дали варн", "дали ванн",
                    "дали мут", "дали деморган", "дала варн", "дала бан", "бан просто так",
                    "варн просто так", "мут просто так", "за что бан", "за что варн", "за что ванн",
                    "за что мут", "ban berdi", "mute berdi", "mute beradi", "warn berdi",
                    "warn beryabdi", "unban", "bandan chiqaring", "баннан шығару",
                )
            )
            or verdict in ("appeal_rejected", "appeal_approved")
        ):
            return "complaint.admin"

        # 3. Check normalized rule from rule_refs_raw or violation (only complaint.* categories)
        for norm_r in (norm_rref, norm_viol):
            if norm_r and norm_r in self.rag.rules_by_id:
                rcat = self.rag.rules_by_id[norm_r].category
                if rcat in self.taxonomy.valid_categories and rcat.startswith("complaint.") and rcat != "complaint.invalid":
                    return rcat

        # 4. Cheats & exploits (5.1, 5.2)
        if any(
            w in combined_clues
            for w in (
                "5.2", "5.1", "чит", "софт", "soft", "аим", "aim", "спидхак", "speedhack",
                "телепорт", "рванк", "гм", "godmode", "багоюз", "багаюз", "дюп", "сторонн",
                "клео", "cleo", "cheat", "cheater",
            )
        ):
            return "complaint.cheat"

        # 5. Scam & property fraud (4.1, 4.2, 4.8, 4.9)
        if any(
            w in combined_clues
            for w in (
                "4.1", "4.2", "4.8", "4.9", "обман", "скам", "scam", "кинул", "развод",
                "развел", "развёл", "мошеннич", "условия сделки", "не вернул долг", "обмен",
                "алдап", "алаяқ", "aldab", "firibgar",
            )
        ):
            return "complaint.scam"

        # 6. RMT & ads (2.3, 4.3-4.7)
        if any(
            w in combined_clues
            for w in (
                "2.3", "4.3", "4.4", "4.5", "4.6", "4.7", "рмт", "rmt", "покупка вирт",
                "продажа вирт", "продам вирт", "куплю вирт", "за реальные деньги", "за реал",
                "продажа акк", "покупка акк", "реклама тг", "реклама канал", "промокод",
            )
        ):
            return "complaint.rmt_ads"

        # 7. Insults & toxicity (2.1, 2.2, 2.4, 2.5, 2.6, 2.7, 3.2, 3.3)
        if any(
            w in combined_clues
            for w in (
                "2.1", "2.2", "2.4", "2.5", "2.6", "2.7", "3.2", "3.3", "оск", "оскорб",
                "мат", "матер", "родн", "мать", "маму", "папу", "отца", "семью", "упом",
                "неадекват", "токсик", "униж", "розжиг", "наци", "капс", "caps", "флуд",
                "музык", "войс", "обозвал", "обзывал", "шлюх", "даун", "клоун", "балағат",
                "боқта", "шеше", "анасын", "haqorat", "so'k", "sokdi", "insult", "toxic",
            )
        ):
            return "complaint.insult"

        # 8. DM, DB, SK, TK, PG, MG, RK, GZ, NonRP (5.3, 5.4, 5.5, 5.7, GZ, ПГО)
        if any(
            w in combined_clues
            for w in (
                "5.3", "5.4", "5.5", "5.7", "дм", "dm", "дб", "db", "ск", "sk", "тк", "tk",
                "пг", "pg", "мг", "mg", "рк", "rk", "нонрп", "нрп", "nonrp", "nrp", "зз",
                "gz", "зелен", "зелён", "убил", "убийств", "урон", "стреля", "расстрел",
                "задмил", "задавил", "таран", "связал", "наручник", "арест", "розыск",
                "коп", "полиц", "мвд", "фсб", "уход от рп", "лив от рп", "орп", "пго",
                "өлтірді", "o'ldirdi", "oldirdi", "killed",
            )
        ):
            return "complaint.dm"

        # 9. Invalid format / duplicate / question in complaints
        if verdict in ("invalid_format", "duplicate") or any(
            w in lower_player for w in ("не по форме", "дубль", "дубликат")
        ):
            return "complaint.invalid"

        return "complaint.other"

    def _derive_priority_for_label(
        self,
        category: str,
        kind: str,
        lower_player: str,
        lower_rref: str,
        has_evidence: bool,
    ) -> str:
        # Support legacy priority names if custom taxonomy is loaded
        if "p0" not in self.taxonomy.valid_priorities:
            return self.taxonomy.categories[category].default_priority

        combined = f"{lower_player} {lower_rref}"

        # p0: "Взлом или потеря доступа к аккаунту, платежи, массовый баг, угрозы вне игры"
        if category in ("report.account", "report.payment"):
            return "p0"
        if any(
            w in combined
            for w in (
                "деанон", "сват", "докс", "угрозы в реал", "угроза жизни", "шантаж и угроз",
                "массовый баг", "дюп валют", "взломали аккаунт", "украли аккаунт",
            )
        ):
            return "p0"

        # p3: "Вопросы по механике, предложения, оффтоп, жалобы без доказательств"
        if category in ("report.offtopic", "complaint.invalid"):
            return "p3"
        if category == "report.question":
            if any(w in combined for w in ("правил", "разрешен", "запрещен", "наказан", "бан", "мут", "варн")):
                return "p2"
            return "p3"
        if kind == "complaint" and not has_evidence and category not in ("complaint.cheat", "complaint.scam", "complaint.rmt_ads", "complaint.admin"):
            return "p3"

        # p1: "Читы, мошенничество с крупным имуществом, реклама RMT, жалобы с доказательствами"
        if category in ("complaint.cheat", "complaint.scam", "complaint.rmt_ads"):
            return "p1"
        if category == "complaint.admin" and has_evidence:
            return "p1"
        if category == "report.complaint" and any(w in combined for w in ("чит", "софт", "рванк", "спидхак", "аим", "рмт")):
            return "p1"
        # Severe complaints with evidence (relatives insult 2.6, hate speech 2.2, mass DM)
        if has_evidence and any(
            w in combined
            for w in ("2.6", "2.2", "родн", "мать", "маму", "отца", "шеше", "анасын", "онасин", "mother", "масс дм", "mass dm")
        ):
            return "p1"

        # p2: "Оскорбления, DM, нарушения РП, вопросы по правилам"
        return "p2"

    def _read_jsonl_sampled(self, path: Path, max_per_kind: int = 8000) -> List[Dict[str, Any]]:
        """
        Read training records from JSONL / JSONL.GZ.
        In `tickets_train.jsonl.gz` (118,337 lines), the first 67,755 lines are `complaint`
        and the remaining 50,582 lines are `report`. Sampling up to `max_per_kind` from each
        ensures balanced coverage of both `complaint.*` and `report.*` in ~1.5 seconds.
        """
        records: List[Dict[str, Any]] = []
        if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
            open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
        else:
            open_fn = lambda p: open(p, "r", encoding="utf-8")

        counts_by_kind: Dict[str, int] = {}
        with open_fn(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Fast check if we already have enough complaints before parsing JSON
                if counts_by_kind.get("complaint", 0) >= max_per_kind and '"kind": "complaint"' in line[:80]:
                    continue
                rec = json.loads(line)
                k = str(rec.get("kind") or "complaint")
                if counts_by_kind.get(k, 0) >= max_per_kind:
                    if counts_by_kind.get("complaint", 0) >= max_per_kind and counts_by_kind.get("report", 0) >= max_per_kind:
                        break
                    continue
                counts_by_kind[k] = counts_by_kind.get(k, 0) + 1
                records.append(rec)
        return records

    def train_from_jsonl(self, train_path: Path, save_to: Optional[Path] = None) -> int:
        """Train classifier on `tickets_train.jsonl.gz` records."""
        records = self._read_jsonl_sampled(train_path)
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
            if not cat or cat not in self.taxonomy.valid_categories:
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
            rref = str(rec.get("rule_refs_raw") or "")
            feats = _extract_features(text, kind=kind, rule_refs_raw=rref)
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

        # Seed with taxonomy title, description & keywords so every category has strong domain priors
        for cid, cspec in self.taxonomy.categories.items():
            seed_text = f"{cspec.name} {cspec.description} {' '.join(cspec.keywords * 4)}"
            seed_feats = _extract_features(seed_text, kind=cspec.kind)
            c_map = self.cat_token_counts.setdefault(cid, {})
            self.cat_doc_counts[cid] = self.cat_doc_counts.get(cid, 0) + 5
            for ft in seed_feats:
                self.vocab.add(ft)
                c_map[ft] = c_map.get(ft, 0.0) + 2.0
                self.cat_total_tokens[cid] = self.cat_total_tokens.get(cid, 0.0) + 2.0

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
        rref_str = (facts.rule_refs_raw or "").lower()
        expanded_tokens = _tokenize(f"{text} {rref_str}")
        expanded_str = f"{lower_text} {rref_str} {' '.join(expanded_tokens)}"
        norm_kind = (kind or "complaint").strip().lower()

        # Check if official complaint.* / report.* taxonomy is active
        has_kind_split = any(c.startswith("complaint.") for c in self.taxonomy.valid_categories) and any(
            c.startswith("report.") for c in self.taxonomy.valid_categories
        )

        # Effective kind: if caller passed non-standard kind like "question" or "bug", map appropriately
        effective_kind = norm_kind
        if has_kind_split and norm_kind not in ("complaint", "report"):
            if norm_kind in ("question", "help"):
                effective_kind = "report"
            else:
                effective_kind = "complaint"

        # Fast-path for pure spam / meaningless flood
        spam_cat = "report.offtopic" if "report.offtopic" in self.taxonomy.valid_categories else (
            "spam_offtopic" if "spam_offtopic" in self.taxonomy.valid_categories else None
        )
        if is_spam and spam_cat:
            has_real_complaint = any(
                kw in expanded_str
                for kw in ("оскорб", "обман", "чит", "взлом", "донат", "забанил", "убил", "краш", "скам")
            )
            if not has_real_complaint:
                cat = "complaint.invalid" if (effective_kind == "complaint" and "complaint.invalid" in self.taxonomy.valid_categories) else spam_cat
                prio = self._determine_priority(cat, effective_kind, expanded_str, facts, is_spam, is_duplicate, reasons)
                reasons.append(f"Категория '{cat}' назначена по признакам спама/оффтопа")
                return cat, prio, 0.92, reasons

        # Fast-path for dataset complaints without evidence when rule_refs_raw is present (official dataset schema)
        if (
            has_kind_split
            and effective_kind == "complaint"
            and not facts.has_evidence
            and facts.rule_refs_raw is not None
            and "complaint.invalid" in self.taxonomy.valid_categories
        ):
            cat = "complaint.invalid"
            prio = self._determine_priority(cat, effective_kind, expanded_str, facts, is_spam, is_duplicate, reasons)
            reasons.append("Категория 'complaint.invalid': жалоба из датасета без приложенных доказательств (images_count=0 и нет ссылок)")
            return cat, prio, 0.88, reasons

        feats = _extract_features(text, kind=effective_kind, rule_refs_raw=rref_str)
        vocab_size = max(1, len(self.vocab))
        total_docs = max(1, sum(self.cat_doc_counts.values()))

        # Compute deterministic domain-rule signals to blend with Multinomial NB
        rule_hint_cat: Optional[str] = None
        if has_kind_split:
            if effective_kind == "report":
                rule_hint_cat = self._label_report_category(lower_text, "")
            else:
                # For interactive complaints without rule_refs_raw, allow substantive classification even if no screenshot attached
                norm_rref = self.rag.normalize_rule_ref(facts.rule_refs_raw or "")
                rule_hint_cat = self._label_complaint_category(
                    lower_player=lower_text,
                    lower_admin="",
                    lower_rref=rref_str,
                    lower_viol="",
                    norm_rref=norm_rref,
                    norm_viol=None,
                    verdict="",
                    has_evidence=True if facts.rule_refs_raw is None else facts.has_evidence,
                    combined_clues=f"{rref_str} {lower_text}",
                )
            # Also allow cross-kind override if interactive caller sent kind='complaint' (default) for a clear payment/account/question/offtopic
            if facts.rule_refs_raw is None and effective_kind == "complaint":
                if any(w in lower_text for w in ("не пришел донат", "не пришёл донат", "ошибка оплаты", "верните деньги за донат", "чек оплаты", "донатил", "refund", "қайтару", "to'lov")):
                    if "report.payment" in self.taxonomy.valid_categories:
                        rule_hint_cat = "report.payment"
                elif any(w in lower_text for w in ("взломали аккаунт", "забыл пароль", "украли аккаунт", "потерял доступ к аккаунту", "ұрланды", "o'g'irlandi", "hacked my account")):
                    if "report.account" in self.taxonomy.valid_categories:
                        rule_hint_cat = "report.account"
                elif (
                    "?" in lower_text
                    and any(w in lower_text for w in ("какой командой", "какая команда", "как открыть", "где купить", "как устроиться", "қалай", "қайда", "qanday", "qayerda", "how do i", "which command"))
                    and not any(w in lower_text for w in ("оскорб", "убил", "обман", "чит", "забанил"))
                ):
                    if "report.question" in self.taxonomy.valid_categories:
                        rule_hint_cat = "report.question"
                elif any(w in lower_text for w in ("застрял в текстурах", "переверните машину", "вытащите из воды", "вылетает игра", "крашит при")):
                    if "report.help" in self.taxonomy.valid_categories:
                        rule_hint_cat = "report.help"

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

            norm_ll = ll / max(1, len(feats))

            kw_hits = sum(1 for kw in cspec.keywords if kw in expanded_str)
            bonus = min(0.50, kw_hits * 0.09)

            # Mentioned valid rules alignment
            for r_id in facts.valid_mentioned_rules:
                if any(r_id == pref or r_id.startswith(pref) for pref in cspec.related_rule_prefixes):
                    bonus += 0.35
                elif r_id in self.rag.rules_by_id and self.rag.rules_by_id[r_id].category == cid:
                    bonus += 0.30

            # Kind alignment
            if has_kind_split:
                if cspec.kind == effective_kind:
                    bonus += 0.35
                elif cid != rule_hint_cat:
                    bonus -= 0.65

            if rule_hint_cat and cid == rule_hint_cat:
                bonus += 0.55

            if cid in ("spam_offtopic", "report.offtopic") and not is_spam and not is_duplicate and kw_hits == 0:
                bonus -= 0.35

            log_scores[cid] = norm_ll + bonus

        max_s = max(log_scores.values())
        temp = 0.12
        exp_scores = {cid: math.exp((s - max_s) / temp) for cid, s in log_scores.items()}
        sum_exp = sum(exp_scores.values()) or 1.0
        probs = {cid: val / sum_exp for cid, val in exp_scores.items()}

        ranked_cats = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_prob = ranked_cats[0]
        second_prob = ranked_cats[1][1] if len(ranked_cats) > 1 else 0.0

        margin = best_prob - second_prob
        confidence = min(0.98, max(0.45, round(0.55 + 0.43 * margin, 2)))
        if facts.unknown_mentioned_rules:
            confidence = min(confidence, 0.58)
        elif not facts.has_evidence and best_cat in (
            "complaint.insult", "complaint.dm", "complaint.scam",
            "insult_toxicity", "rp_violation", "scam_fraud",
        ):
            confidence = min(confidence, 0.78)

        reasons.append(
            f"Классификатор определил категорию '{best_cat}' ({self.taxonomy.categories[best_cat].name}) с уверенностью {confidence:.2f}"
        )

        best_prio = self._determine_priority(
            best_cat, effective_kind, lower_text, facts, is_spam, is_duplicate, reasons
        )

        return best_cat, best_prio, confidence, reasons

    def _determine_priority(
        self,
        category: str,
        kind: str,
        expanded_str: str,
        facts: ExtractedFacts,
        is_spam: bool,
        is_duplicate: bool,
        reasons: List[str],
    ) -> str:
        # Official handout priorities: p0, p1, p2, p3
        if "p0" in self.taxonomy.valid_priorities:
            prio = self._derive_priority_for_label(
                category=category,
                kind=kind,
                lower_player=expanded_str,
                lower_rref=facts.rule_refs_raw or "",
                has_evidence=facts.has_evidence,
            )
            title = self.taxonomy.priority_titles.get(prio, prio)
            reasons.append(
                f"Приоритет '{prio}' ({title}) назначен по критериям taxonomy.json для категории '{category}'"
            )
            return prio

        # Fallback for custom/legacy priority codes (critical, high, medium, low)
        cspec = self.taxonomy.categories[category]
        prio = cspec.default_priority
        if prio not in self.taxonomy.valid_priorities:
            prio = self.taxonomy.priorities[len(self.taxonomy.priorities) // 2]

        if is_spam or category == "spam_offtopic":
            if "low" in self.taxonomy.valid_priorities:
                reasons.append("Приоритет 'low': спам или оффтоп-обращение")
                return "low"

        if category in ("cheating_exploits", "account_recovery") and "critical" in self.taxonomy.valid_priorities:
            reasons.append(f"Приоритет 'critical' по критериям таксономии для категории '{category}'")
            return "critical"

        if category == "insult_toxicity":
            severe_insult = any(
                w in expanded_str
                for w in ("родн", "мать", "маму", "родител", "нациз", "розжиг", "шеше", "анасын", "mother", "parents", "2.2", "2.6")
            )
            if severe_insult and "high" in self.taxonomy.valid_priorities:
                reasons.append("Приоритет повышен до 'high': упоминание родных или розжиг")
                return "high"

        reasons.append(f"Приоритет '{prio}' назначен согласно критериям taxonomy.json для '{category}'")
        return prio
