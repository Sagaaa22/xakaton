"""
Intent Recognition module for Grand Mobile Support Copilot.
Detects player intent from natural language in RU/KK/UZ/EN.
Includes admin request detection (direct call for a human admin).
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.intent")

# Intent categories with confidence boosters
INTENT_PATTERNS: Dict[str, List[str]] = {
    "admin_request": [
        # Russian
        r"позов[ите]{0,2}\s+администратор",
        r"нужен\s+(живой|настоящий)?\s*адм[ин]{0,2}",
        r"хочу\s+поговорить\s+с\s+человеком",
        r"пригласи[те]{0,2}\s+адм[ин]{0,2}",
        r"вызовите?\s+адм[ин]{0,2}",
        r"позовите?\s+адм[ин]{0,2}",
        r"нужен\s+адм[ин]{0,2}",
        r"позвони[те]{0,2}\s+адм[ин]{0,2}",
        r"можно\s+(к\s+)?адм[ин]{0,2}",
        r"чел[ове]{0,3}к[еу]?\s+(а не\s+)?бот",
        r"говорить с человеком",
        r"не с ботом",
        r"живой человек",
        r"настоящий администратор",
        # English
        r"call\s+(an?\s+)?admin",
        r"speak\s+(to|with)\s+(an?\s+)?admin",
        r"want\s+(a|an)?\s+human",
        r"real\s+person",
        r"talk\s+to\s+(a\s+)?human",
        r"i\s+need\s+(an?\s+)?admin",
        r"get\s+me\s+(an?\s+)?admin",
        # Kazakh
        r"админ\s+шақырыңыз",
        r"адм[ин]{0,2}\s+шақыр",
        r"тірі\s+адам",
        r"басқарушы\s+шақыр",
        # Uzbek
        r"admin\s+chaqir",
        r"jonli\s+odam",
        r"inson\s+bilan\s+gaplash",
    ],
    "appeal": [
        r"разбан[ьте]{0,2}",
        r"снимите?\s+бан",
        r"обжалов(ать|ую|ание)",
        r"несправедлив(о|ый|ое)\s+бан",
        r"почему\s+забанили",
        r"разблокируй[те]{0,2}",
        r"амнисти[яю]",
        r"unban\s+me",
        r"appeal\s+(my\s+)?ban",
        r"baннан\s+шығар",
        r"bandan\s+chiqar",
    ],
    "scam_report": [
        r"(меня|мне)\s+(обманули|кинули|развели|скамнули)",
        r"скам[нули]{0,4}",
        r"мошенник",
        r"обманул[аи]?\s+(при|на|в)\s+(обмен|сделк|продаж|покупк)",
        r"кинул\s+на\s+(машин|авто|дом|бизнес|деньги)",
        r"(забрал|украл|стащил)\s+(мою?\s+)?(машин|авто|дом|деньги|вещи)",
        r"trade\s+scam",
        r"i\s+got\s+scammed",
        r"алдап\s+кетт",  # kk: cheated and left
        r"мені?\s+(алданды|алдады)",
        r"aldab\s+ketdi",  # uz
        r"firibgar",  # uz: fraudster
    ],
    "bug_report": [
        r"(вылетает|краш[ает]{0,6}|зависает|фризит)",
        r"(не работает|не открывается|не запускается)\s+(игра|сервер|меню|магазин)",
        r"баг[уеи]?",
        r"глюч[ает]{0,6}",
        r"технич[еская]{0,6}\s+проблем",
        r"bug\s+report",
        r"game\s+crash",
        r"server\s+crash",
        r"qata\s+bor",  # uz: there's an error
    ],
    "account_problem": [
        r"взломали\s+(мой\s+)?аккаунт",
        r"потерял\s+доступ",
        r"не\s+могу\s+(войти|зайти)",
        r"восстановить\s+аккаунт",
        r"сменить\s+пароль",
        r"двухфакторн",
        r"2fa",
        r"account\s+(hacked|stolen|lost)",
        r"can'?t\s+(log\s+in|login|sign\s+in)",
        r"аккаунтымды\s+ұрлады",  # kk: my account was stolen
        r"akkauntimni\s+o'g'irlash",  # uz
    ],
    "property_problem": [
        r"пропал(а|и)?\s+(машин|авто|дом|бизнес|инвентарь|вещи|деньги|оружие)",
        r"исчез(ла|ли)?\s+(машин|авто|дом|бизнес)",
        r"слетел(а|и)?\s+(тюнинг|багажник|инвентарь)",
        r"потерял\s+(машин|авто|дом|бизнес|деньги|вещи)",
        r"украли?\s+(машин|авто|дом|бизнес|деньги)",
        r"(car|vehicle|house|property)\s+(missing|stolen|disappeared)",
        r"мүлкім\s+жоғалды",  # kk: my property is gone
        r"mulkim\s+yo'qoldi",  # uz
    ],
    "complaint": [
        r"жалоб[аую]",
        r"репорт\s+на",
        r"сообщ(аю|аем)\s+о\s+нарушении",
        r"нарушает?\s+правила",
        r"оскорбл(яет|ил|ял)",
        r"complaint\s+(about|against)",
        r"reporting\s+(a\s+)?player",
        r"I\s+want\s+to\s+report",
        r"шағым\s+береміз?",  # kk: we file a complaint
        r"shikoyat\s+qilam",  # uz: I complain
    ],
    "question": [
        r"(как|где|когда|почему|зачем)\s+[а-яa-z]",
        r"подскаж[иет]{0,4}",
        r"помог[ите]{0,4}",
        r"вопрос",
        r"не\s+понимаю",
        r"объясн[иет]{0,4}",
        r"(how|where|when|why)\s+",
        r"help\s+me",
        r"can\s+you\s+help",
        r"қалай",  # kk: how
        r"qanday",  # uz: how
    ],
    "technical_support": [
        r"техническ(ая|ую)\s+(поддержку?|помощь)",
        r"проблем[аыу]\s+с\s+(игрой|сервером|входом|аккаунтом)",
        r"не\s+работает",
        r"technical\s+support",
        r"having\s+issues?\s+with",
        r"техподдержка",
    ],
}

# Slang normalization map (slang -> standard form for classification)
SLANG_NORMALIZATION: Dict[str, str] = {
    # Scam-related
    "кинул": "обманул",
    "кинули": "обманули",
    "кинуть": "обмануть",
    "заскамили": "обманули при торговле",
    "скамнули": "обманули при торговле",
    "скам": "мошенничество",
    "развели": "обманули",
    "развод": "обман",
    # RP violations
    "дм": "dm",
    "нонрп": "nonrp",
    "нрп": "nrp нарушение",
    "пг": "powergaming",
    "мг": "metagaming",
    "рк": "revengekill",
    # Cheating
    "читер": "игрок с читами",
    "читерит": "использует читы",
    "хакер": "игрок со сторонним ПО",
    "спидхакер": "использует speedhack",
    "аимбот": "aimbot",
    # Account
    "взломали акк": "взломали аккаунт",
    "акк": "аккаунт",
    # Bans
    "раздемы": "разбан",
    "разбань": "разбаните",
    # Vehicles
    "тачку": "машину",
    "тачка": "машина",
    "тачки": "машины",
    # Generic
    "баг": "ошибка/баг",
    "глюк": "баг/ошибка",
    "фризит": "зависает",
    "крашится": "вылетает",
    # Kazakh slang/mixed
    "мені машинамды алып кетті": "меня обманули при торговле машиной",
    # Uzbek
    "aldab ketdi": "обманул и ушёл",
}

# Admin call triggers — fast check without regex
ADMIN_CALL_KEYWORDS = frozenset([
    "администратор", "адм", "admin", "живой человек", "настоящий человек",
    "не бот", "бот не", "поговорить с человеком", "живого",
    "позовите", "позови", "вызовите", "вызови", "пригласите",
    "شادыр", "шақыр",  # kk call
    "chaqir",  # uz call
])


def normalize_slang(text: str) -> str:
    """
    Normalize game slang and typos to help classification.
    IMPORTANT: Original text is preserved — this is used only for intent detection.
    """
    result = text
    for slang, normalized in SLANG_NORMALIZATION.items():
        result = re.sub(re.escape(slang), normalized, result, flags=re.IGNORECASE)
    return result


def detect_intent(
    text: str,
    category_hint: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    """
    Detect player intent from text.
    
    Returns: (intent_id, confidence, matched_signals)
    
    Supported intents:
    - complaint, question, technical_support, appeal, scam_report,
      bug_report, admin_request, account_problem, property_problem, other
    """
    if not text or not text.strip():
        return "other", 0.0, []

    lower_text = text.lower().strip()
    normalized = normalize_slang(lower_text)
    
    # Fast-path: admin request keywords
    words = set(re.findall(r"[а-яёa-z]+", lower_text))
    if words & ADMIN_CALL_KEYWORDS:
        # Check if it's a clear admin call (not just mentioning admin)
        admin_phrases = ["позов", "вызов", "нужен адм", "хочу с", "поговорить", "живой", "call", "speak", "шақыр", "chaqir"]
        if any(phrase in lower_text for phrase in admin_phrases):
            return "admin_request", 0.95, ["Прямое обращение с просьбой вызвать администратора"]

    # Score each intent
    scores: Dict[str, Tuple[float, List[str]]] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        matched = []
        for pattern in patterns:
            for source in (lower_text, normalized):
                if re.search(pattern, source, re.IGNORECASE):
                    matched.append(pattern)
                    break
        if matched:
            # Score = ratio of matched patterns, capped
            raw_score = min(1.0, len(matched) / max(1, len(patterns) * 0.3))
            scores[intent] = (raw_score, matched)

    if not scores:
        # Fallback: category-based intent guess
        category_intent_map = {
            "ban_appeal": "appeal",
            "scam_fraud": "scam_report",
            "technical_bug": "bug_report",
            "account_recovery": "account_problem",
            "billing_refund": "question",
            "gameplay_question": "question",
        }
        if category_hint and category_hint in category_intent_map:
            return category_intent_map[category_hint], 0.50, [f"intent derived from category: {category_hint}"]
        return "other", 0.30, []

    # Pick best scoring intent
    best_intent = max(scores, key=lambda k: scores[k][0])
    best_score, best_signals = scores[best_intent]
    
    # Disambiguate: if complaint and scam_report both scored, prefer scam_report
    if "scam_report" in scores and "complaint" in scores:
        best_intent = "scam_report"
        best_score = max(scores["scam_report"][0], scores["complaint"][0])
        best_signals = scores["scam_report"][1] + scores["complaint"][1]

    # Cap confidence between 0.35 and 0.98
    confidence = max(0.35, min(0.98, best_score))
    
    return best_intent, confidence, best_signals[:5]


INTENT_LABELS: Dict[str, str] = {
    "complaint": "Жалоба на игрока",
    "question": "Вопрос / Запрос информации",
    "technical_support": "Техническая поддержка",
    "appeal": "Апелляция / Разбан",
    "scam_report": "Сообщение о мошенничестве",
    "bug_report": "Сообщение об ошибке / баге",
    "admin_request": "Прямой вызов администратора",
    "account_problem": "Проблема с аккаунтом",
    "property_problem": "Проблема с игровым имуществом",
    "other": "Другое",
}

INTENT_REQUIRES_ADMIN: Dict[str, bool] = {
    "complaint": True,
    "question": False,
    "technical_support": False,
    "appeal": True,
    "scam_report": True,
    "bug_report": False,
    "admin_request": True,
    "account_problem": True,
    "property_problem": True,
    "other": False,
}
