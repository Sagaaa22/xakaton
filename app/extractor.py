"""
Language detection, structured fact extraction, and spam/duplicate detection.
"""
from collections import deque
import hashlib
import re
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from app.schemas import ExtractedFacts, MessageItem, TriageRequest


KAZAKH_SPECIFIC_CHARS = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
UZBEK_CYRILLIC_CHARS = set("ўҳЎҲ")

KAZAKH_WORDS = {
    "сәлем", "сәлеметсіз", "ойыншы", "мені", "көмек", "серверде", "балағаттады",
    "боқтап", "боқтады", "алдап", "кетті", "бар", "жоқ", "қалай", "қайда",
    "өтініш", "тіркедім", "көлік", "ақша", "баннан", "шығару", "дәлел",
    "шағым", "ереже", "неге", "ұрланды", "жатыр", "деген", "кезінде", "сату",
}

UZBEK_WORDS = {
    "salom", "assalomu", "alaykum", "o'yinchi", "oyinchi", "meni", "haqorat",
    "qildi", "qilyapti", "so'kdi", "sokdi", "skrinshot", "bor", "yo'q", "yoq",
    "iltimos", "bandan", "chiqaring", "nohaq", "akkauntimni", "serverda",
    "aldab", "ketdi", "yordam", "bering", "qanday", "qayerda", "pul",
    "qaytarish", "qilishdi", "xato", "ilova", "qildim", "ovozli", "chatda",
}

ENGLISH_WORDS = {
    "player", "insulted", "chat", "server", "killed", "without", "reason",
    "video", "attached", "screenshot", "hello", "please", "help", "unban",
    "banned", "refund", "money", "how", "where", "which", "command", "commands",
    "arrested", "rule", "violation", "cheater", "my", "me", "the", "and",
    "for", "with", "near", "spawn", "hospital", "open", "navigation", "jobs",
    "licenses", "dialogue", "car", "scammed", "hacked", "account", "crash",
}

RUSSIAN_WORDS = {
    "игрок", "меня", "оскорбил", "оскорблял", "чат", "чате", "сервере",
    "сервер", "скриншот", "скриншоты", "скрины", "есть", "нет", "убил",
    "без", "причины", "обманул", "разбан", "разбаньте", "забанили", "деньги",
    "донат", "верните", "помогите", "подскажите", "как", "где", "жалоба",
    "нарушил", "правило", "пункт", "приложил", "прикрепил", "прошу", "моего",
    "мой", "аккаунт", "взломали", "пропал", "пропали", "вылетает", "крашит",
    "командой", "открыть", "паспорт", "машину", "обмене", "сделки", "фрапс",
}

STOPWORDS_NICK = {
    "Grand", "Mobile", "Server", "Admin", "Player", "Report", "Google",
    "Authenticator", "Android", "Xiaomi", "Poco", "Iphone", "Samsung",
}


def detect_language(text: str) -> Tuple[str, bool, List[str]]:
    """
    Detect primary language ('ru', 'kk', 'uz', 'en', 'und') and whether mixed language is used.
    Returns: (primary_lang, is_mixed, detected_languages_list)
    """
    if not text or not text.strip():
        return "und", False, []

    cleaned = text.strip()
    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if len(alpha_chars) < 2:
        return "und", False, []

    lower_text = cleaned.lower()
    # Normalize apostrophes for Uzbek matching
    norm_text = re.sub(r"[’ʻ`´]", "'", lower_text)
    words = re.findall(r"[a-zа-яёәіңғүұқөһў'-]+", norm_text)

    scores: Dict[str, float] = {"ru": 0.0, "kk": 0.0, "uz": 0.0, "en": 0.0}

    # 1. Character-level signals
    kk_chars = sum(1 for ch in cleaned if ch in KAZAKH_SPECIFIC_CHARS)
    uz_cyr_chars = sum(1 for ch in cleaned if ch in UZBEK_CYRILLIC_CHARS)
    uz_lat_apostrophes = len(re.findall(r"\b(?:o'|g'|sh|ch)", norm_text))
    cyrillic_count = sum(1 for ch in alpha_chars if "\u0400" <= ch <= "\u04ff")
    latin_count = sum(1 for ch in alpha_chars if "a" <= ch.lower() <= "z")

    scores["kk"] += kk_chars * 2.5
    scores["uz"] += uz_cyr_chars * 3.0 + uz_lat_apostrophes * 1.8

    # 2. Word-level dictionary signals
    for w in words:
        w_clean = w.strip("'-")
        if not w_clean:
            continue
        if w_clean in KAZAKH_WORDS:
            scores["kk"] += 3.0
        if w_clean in UZBEK_WORDS or w in UZBEK_WORDS:
            scores["uz"] += 3.0
        if w_clean in ENGLISH_WORDS:
            scores["en"] += 2.5
        if w_clean in RUSSIAN_WORDS:
            scores["ru"] += 2.5

    # Suffix heuristics for Uzbek (-serverda, -band, -imni, -dilar, -yapti)
    uz_suffix_hits = len(
        re.findall(r"\b[a-z']+(?:serverda|imni|qildi|yapti|chiqaring|bandan|skrinshot)\b", norm_text)
    )
    scores["uz"] += uz_suffix_hits * 2.5

    # Script fallback if no specific dictionary words matched
    if max(scores.values()) == 0.0:
        # Check if gibberish (e.g. consonant cluster or keyboard mash or repeated single char)
        unique_chars = set(ch.lower() for ch in alpha_chars)
        if len(unique_chars) <= 2:
            return "und", False, []
        if cyrillic_count > latin_count and cyrillic_count >= 3:
            scores["ru"] += 1.0
        elif latin_count > cyrillic_count and latin_count >= 3:
            scores["en"] += 1.0
        else:
            return "und", False, []
    else:
        # Add mild script baseline
        if cyrillic_count >= 4 and scores["ru"] > 0:
            scores["ru"] += 1.0
        if latin_count >= 4 and scores["en"] > 0:
            scores["en"] += 1.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_lang, primary_score = ranked[0]
    secondary_lang, secondary_score = ranked[1]

    if primary_score <= 0.0:
        return "und", False, []

    # Check for genuine mixed-language usage (e.g. ru + en, kk + ru, uz + ru)
    # Require secondary language to have at least 4.0 score (approx >= 2 distinct vocabulary hits)
    # and at least 25% of primary language score.
    is_mixed = False
    detected_langs = [primary_lang]
    if secondary_score >= 4.0 and (secondary_score / primary_score) >= 0.25:
        is_mixed = True
        detected_langs.append(secondary_lang)

    return primary_lang, is_mixed, detected_langs


class TicketFactExtractor:
    """Extracts structured facts, spam indicators, and duplicate signals from tickets."""

    def __init__(self, max_cache_size: int = 500) -> None:
        self._recent_hashes: Deque[str] = deque(maxlen=max_cache_size)
        self._hash_set: Set[str] = set()

    def reset_cache(self) -> None:
        self._recent_hashes.clear()
        self._hash_set.clear()

    def _normalize_for_dedup(self, text: str) -> str:
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]

    def detect_spam_and_duplicate(
        self,
        player_messages: List[MessageItem],
        record_in_cache: bool = True,
    ) -> Tuple[bool, bool, List[str]]:
        """
        Detect whether the ticket is spam/flood or a duplicate.
        Returns: (is_spam, is_duplicate, reasons)
        """
        reasons: List[str] = []
        is_spam = False
        is_duplicate = False

        texts = [m.text.strip() for m in player_messages if m.text and m.text.strip()]
        combined = " ".join(texts)
        lower = combined.lower()

        # 1. Check intra-ticket duplicate messages (user sent identical message multiple times)
        if len(texts) >= 2:
            norm_msgs = [re.sub(r"\s+", " ", t.lower()) for t in texts]
            if len(set(norm_msgs)) < len(norm_msgs):
                is_duplicate = True
                reasons.append("Обнаружены дублирующиеся сообщения внутри одного обращения")

        # 2. Check inter-ticket duplicate cache
        msg_hash = self._normalize_for_dedup(combined)
        if msg_hash in self._hash_set:
            is_duplicate = True
            reasons.append("Повторное обращение (дубликат ранее отправленного тикета)")
        elif record_in_cache and len(combined) >= 5:
            if len(self._recent_hashes) == self._recent_hashes.maxlen:
                oldest = self._recent_hashes.popleft()
                self._hash_set.discard(oldest)
            self._recent_hashes.append(msg_hash)
            self._hash_set.add(msg_hash)

        # 3. Check character repetition flood (e.g. "аааааааа", "11111111", "???????")
        if re.search(r"(.)\1{6,}", lower):
            is_spam = True
            reasons.append("Обнаружен флуд повторяющимися символами")

        # 4. Check keyboard mash / off-topic begging / test spam
        spam_patterns = [
            r"\b(?:фыва\w*|йцук\w*|asdf\w*|qwer\w*)\b",
            r"\b(?:дайте\s+\d*\s*(?:миллион\w*|кк|вирт\w*)\s+просто\s+так)\b",
            r"\b(?:продам\s+гараж\s+в\s+реале)\b",
            r"^(?:тест\s*)+$",
            r"\b(?:ау{3,}|алло{3,})\b",
        ]
        for pat in spam_patterns:
            if re.search(pat, lower):
                is_spam = True
                reasons.append("Признаки спама, оффтопа или тестового флуда в тексте")
                break

        # 5. Very short meaningless text (only digits or punctuation)
        alpha_count = sum(1 for c in combined if c.isalpha())
        if alpha_count < 3 and len(combined) > 0:
            is_spam = True
            reasons.append("Сообщение не содержит осмысленного текста")

        return is_spam, is_duplicate, reasons

    def extract_facts(
        self,
        request: TriageRequest,
        valid_rule_ids: Set[str],
    ) -> Tuple[str, ExtractedFacts]:
        """
        Extract language and structured facts from a TriageRequest.
        """
        player_msgs = [
            m for m in request.messages if m.role in ("player", "user", "client", "reporter")
        ]
        if not player_msgs:
            player_msgs = request.messages

        combined_text = "\n".join(m.text for m in player_msgs if m.text)
        lang, is_mixed, detected_langs = detect_language(combined_text)

        # 1. Attachments & Evidence
        attachments_count = sum(m.attachments_count for m in player_msgs)
        evidence_types: List[str] = []
        if attachments_count > 0:
            evidence_types.append("attachments")

        lower_text = combined_text.lower()
        has_url = bool(
            re.search(
                r"(?:https?://\S+|(?:imgur\.com|yapx\.ru|youtu\.be|youtube\.com|ibb\.co|vk\.com/video)\S*)",
                lower_text,
            )
        )
        if has_url:
            evidence_types.append("media_url")

        negated_evidence = bool(
            re.search(
                r"(?:скрин\w*\s+нет|нет\s+скрин\w*|без\s+скрин\w*|не\s+успел\s+сделать|без\s+фрапс\w*|доказательств\s+нет|дәлел\s+жоқ|skrinshot\s+yo'q|no\s+screenshot)",
                lower_text,
            )
        )
        if not negated_evidence:
            if re.search(r"(?:скрин\w*|skrinshot|screenshot)", lower_text):
                evidence_types.append("screenshot_mention")
            if re.search(r"(?:видео\w*|фрапс\w*|откат\w*|video|fraps)", lower_text):
                evidence_types.append("video_mention")
            if re.search(r"(?:чек\b|квитанц\w*|receipt)", lower_text):
                evidence_types.append("receipt_mention")

        # Crucial safety rule: actual evidence exists ONLY if attachments > 0 or media URL is provided
        has_evidence = (attachments_count > 0) or has_url

        # 2. Nicknames extraction
        nicknames: List[str] = []
        # Standard RP nicknames with underscore: Name_Surname
        for match in re.findall(r"\b([A-Za-z][A-Za-z0-9]{1,15}_[A-Za-z0-9]{1,15})\b", combined_text):
            if match not in nicknames and match not in STOPWORDS_NICK:
                nicknames.append(match)

        # Contextual nicknames without underscore (e.g., "ник Anton", "игрок Baha")
        ctx_nick_patterns = [
            r"(?:игрок\w*|ник\w*|аккаунт\w*|админ\w*|player|ойыншы|o'yinchi|akkaunt\w*)\s+([A-Za-z][A-Za-z0-9_]{2,20})",
            r"\b([A-Za-z][A-Za-z0-9_]{2,20})\s+(?:деген\s+ойыншы|akkauntimni)",
        ]
        for pat in ctx_nick_patterns:
            for match in re.findall(pat, combined_text, flags=re.IGNORECASE):
                if (
                    match not in nicknames
                    and match not in STOPWORDS_NICK
                    and match.lower() not in {"без", "на", "за", "не", "он", "меня", "мой", "from", "with", "killed", "insulted"}
                ):
                    nicknames.append(match)

        if request.nickname and request.nickname not in nicknames:
            nicknames.insert(0, request.nickname)
        if request.offender_nickname and request.offender_nickname not in nicknames:
            nicknames.append(request.offender_nickname)

        # Determine offender vs reporter nickname
        offender_nick: Optional[str] = request.offender_nickname
        reporter_nick: Optional[str] = request.nickname

        reporter_match = re.search(
            r"(?:мой\s+ник|ник\s+аккаунта|аккаунт\w*|ник|akkauntimni)\s+([A-Za-z][A-Za-z0-9_]{2,20})",
            combined_text,
            flags=re.IGNORECASE,
        )
        if reporter_match and not reporter_nick:
            cand = reporter_match.group(1)
            if cand in nicknames:
                reporter_nick = cand

        offender_match = re.search(
            r"(?:игрок\w*|жалоба\s+на|обманул\s+игрок|читер|коп|админ\w*|player|o'yinchi)\s+([A-Za-z][A-Za-z0-9_]{2,20})",
            combined_text,
            flags=re.IGNORECASE,
        )
        if offender_match and not offender_nick:
            cand = offender_match.group(1)
            if cand in nicknames:
                offender_nick = cand

        if not offender_nick and not reporter_nick and len(nicknames) == 1:
            if request.kind in ("appeal", "bug") or re.search(
                r"(?:разбан\w*|забанил\w*|донат\w*|взломал\w*|пропал\w*)", lower_text
            ):
                reporter_nick = nicknames[0]
            else:
                offender_nick = nicknames[0]

        # 3. Server extraction
        server_val: Optional[Any] = request.server
        if server_val is None:
            srv_patterns = [
                r"(?:на\s+|#|\b)(\d{1,2})\s*(?:-м|-й|-serverda|\s+сервер\w*|\s+server\b)",
                r"(?:сервер\w*|server)\s*#?\s*(\d{1,2})\b",
            ]
            for pat in srv_patterns:
                m = re.search(pat, lower_text)
                if m:
                    server_val = int(m.group(1))
                    break

        # 4. Mentioned Rule IDs extraction
        mentioned_rules: List[str] = []
        rule_patterns = [
            r"(?:п\.|пункт\w*|правил\w*|rule|band\w*)\s*№?\s*(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)",
            r"\b(\d{1,3}\.\d{1,3})\s*-?\s*band\b",
        ]
        for pat in rule_patterns:
            for m in re.findall(pat, lower_text):
                if m not in mentioned_rules:
                    mentioned_rules.append(m)

        # Also check conjunction like "п. 2.2 и 2.3"
        conj_matches = re.findall(
            r"(?:п\.|пункт\w*|правил\w*)\s*\d{1,3}\.\d{1,3}\s*(?:и|,|/)\s*(\d{1,3}\.\d{1,3})",
            lower_text,
        )
        for m in conj_matches:
            if m not in mentioned_rules:
                mentioned_rules.append(m)

        valid_mentioned = [r for r in mentioned_rules if r in valid_rule_ids]
        unknown_mentioned = [r for r in mentioned_rules if r not in valid_rule_ids]

        # 5. Monetary amounts extraction
        amounts = re.findall(
            r"\b(\d+(?:[\.,]\d+)?\s*(?:руб\w*|р\b|тенге|тг|сум\w*|кк|млн|миллион\w*|gc|коин\w*|usd|\$))",
            lower_text,
        )

        # 6. Mentioned game commands
        commands = []
        for cmd in re.findall(r"(?:^|\s)(/[a-zA-Z]{1,12})\b", combined_text):
            if cmd.lower() not in commands:
                commands.append(cmd.lower())

        # 7. Preserve extra input fields if present
        extra_fields: Dict[str, Any] = {}
        if request.extra:
            extra_fields.update(request.extra)
        # Capture any arbitrary extra fields passed to TriageRequest
        known_keys = {
            "id", "ticket_id", "kind", "messages", "server", "nickname",
            "offender_nickname", "regenerate", "extra",
        }
        for k, v in request.__dict__.items():
            if k not in known_keys and not k.startswith("_") and v is not None:
                extra_fields[k] = v

        facts = ExtractedFacts(
            nicknames=nicknames,
            offender_nickname=offender_nick,
            reporter_nickname=reporter_nick,
            mentioned_rules=mentioned_rules,
            valid_mentioned_rules=valid_mentioned,
            unknown_mentioned_rules=unknown_mentioned,
            has_evidence=has_evidence,
            attachments_count=attachments_count,
            evidence_types=evidence_types,
            server=server_val,
            amounts=amounts,
            commands_mentioned=commands,
            is_mixed_language=is_mixed,
            detected_languages=detected_langs,
            extra_fields=extra_fields,
        )
        return lang, facts

