"""
Intent Recognition module for Grand Mobile Support Copilot.
Detects player intent from natural language in RU/KK/UZ/EN.
Includes direct admin request detection (live human admin call) across all 4 languages,
plus gaming slang, typo, and inverted word order normalization.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.intent")

# Intent categories with regex patterns (supports RU, KK, UZ, EN, slang, typos, inverted word order)
INTENT_PATTERNS: Dict[str, List[str]] = {
    "admin_request": [
        # Russian (including typos, slang, inverted word order)
        r"позов[итеья]{0,3}\s+(мне\s+)?(живого\s+|настоящего\s+|старшего\s+)?адм[инастратора]{0,12}",
        r"адм[инастратора]{0,12}\s+(мне\s+)?(сюда\s+)?(позов[итеья]{0,3}|вызов[итеья]{0,3}|дай[те]{0,2}|нужен|надо)",
        r"нужен\s+(живой|настоящий|реальный|старший)?\s*адм[инстратор]{0,10}",
        r"(хочу|надо|можно)\s+(поговорить|пообщаться|связаться)\s+с\s+(живым\s+)?(человеком|адм[иномстратором]{0,12}|оператором|модератором)",
        r"пригласи[те]{0,2}\s+(живого\s+)?адм[инастратора]{0,12}",
        r"вызов[итеья]{0,3}\s+(мне\s+)?(живого\s+)?адм[инастратора]{0,12}",
        r"подключ[итеья]{0,3}\s+(живого\s+|настоящего\s+)?(адм[инастратора]{0,12}|оператора|человека|специалиста)",
        r"(живого|настоящего)\s+(адм[инастратора]{0,12}|человека|оператора)\s+(позов|подключ|дай|нужн)",
        r"позвони[те]{0,2}\s+адм[ину]{0,3}",
        r"можно\s+(к\s+|позвать\s+|подключить\s+)?адм[инау]{0,4}",
        r"чел[ове]{0,3}к[ауом]?\s+(а не\s+|вместо\s+)?бот",
        r"(говорить|общаться)\s+с\s+(живым\s+)?человеком",
        r"не\s+хочу\s+(с\s+)?ботом",
        r"хватит\s+бота",
        r"живой\s+(человек|админ|администратор|оператор)",
        r"настоящий\s+администратор",
        r"перевед[ите]{0,2}\s+на\s+(живого\s+)?(адм[инастратора]{0,12}|оператора|человека|специалиста)",
        r"свяж[итеья]{0,3}\s+с\s+(адм[иномстратором]{0,12}|живым\s+человеком)",
        # English (including inverted order & informal)
        r"call\s+(me\s+)?(a\s+|an\s+|live\s+|real\s+)?admin(istrator)?",
        r"(speak|talk|chat)\s+(to|with)\s+(a\s+|an\s+|live\s+|real\s+)?(admin(istrator)?|human|person|moderator|operator)",
        r"(want|need|get\s+me|connect\s+me\s+to|transfer\s+to)\s+(a\s+|an\s+)?(live\s+|real\s+)?(human|admin(istrator)?|person|operator)",
        r"real\s+(person|human|admin)",
        r"live\s+(admin|support|agent|human|moderator)",
        r"not\s+a\s+bot",
        r"human\s+admin\s+please",
        # Kazakh (including Cyrillic & informal)
        r"адм[инстраторды]{0,10}\s+шақыр[ыңыздаршы]{0,8}",
        r"шақыр[ыңыздаршы]{0,8}\s+адм[инстраторды]{0,10}",
        r"тірі\s+(адам[дымен]{0,5}|адм[инді]{0,6}|оператор[ды]{0,4})",
        r"басқарушы[ны]?\s+шақыр",
        r"адаммен\s+сөйлес[кім]{0,4}\s+келеді",
        r"бот\s+емес\s+(адам|админ)",
        r"админ\s+қос[ыңызшы]{0,6}",
        r"нағыз\s+админ",
        # Uzbek (Latin & Cyrillic variants)
        r"admin[ni]?\s+chaqir[inglar]{0,6}",
        r"chaqir[inglar]{0,6}\s+admin[ni]?",
        r"jonli\s+(odam|admin|operator)",
        r"(inson|odam|admin)\s+bilan\s+gaplash",
        r"admin[ni]?\s+ulab\s+ber[ing]{0,3}",
        r"haqiqiy\s+admin",
        r"bot\s+emas\s+admin",
    ],
    "appeal": [
        r"разбан[ьтеи]{0,3}",
        r"разбна[ьте]{0,2}",
        r"снимите?\s+(мне\s+)?бан",
        r"обжалов(ать|ую|ание)",
        r"апелляц[ияю]",
        r"несправедлив(о|ый|ое)\s+(бан|наказание|мут|варн)",
        r"почему\s+(меня\s+)?забанили",
        r"за\s+что\s+(дали\s+)?бан",
        r"разблокируй[те]{0,2}",
        r"амнисти[яю]",
        r"unban\s+me",
        r"appeal\s+(my\s+)?ban",
        r"wrong(ful)?\s+ban",
        r"баннан\s+шығар",
        r"бұғаттан\s+шығар",
        r"жазаны\s+алып\s+таста",
        r"bandan\s+chiqar",
        r"blokdan\s+chiqar",
    ],
    "scam_report": [
        r"(меня|мне)\s+(обманули|кинули|развели|скамнули|надули|обули)",
        r"скам[нулищик]{0,5}",
        r"мошенник",
        r"обманул[аи]?\s+(при|на|в)\s+(обмен|сделк|продаж|покупк|трейд)",
        r"кинул[аи]?\s+на\s+(машин|тачк|авто|дом|бизнес|деньги|вирты|акс|скин)",
        r"(забрал|украл|стащил)\s+(мою?\s+)?(машин|тачк|авто|дом|деньги|вещи)",
        r"нонрп\s+(обман|развод)",
        r"nrp\s+scam",
        r"trade\s+scam",
        r"i\s+(got|was)\s+scammed",
        r"алдап\s+кетт[іи]",
        r"мені?\s+(алданды|алдады|лақтырып\s+кетті)",
        r"алаяқ",
        r"aldab\s+ketdi",
        r"firibgar",
        r"tushirib\s+ketdi",
    ],
    "bug_report": [
        r"(вылетает|краш[итсяает]{0,6}|зависает|фризит|лагает)",
        r"(не\s+работает|не\s+открывается|не\s+запускается)\s+(игра|сервер|меню|магазин|инвентарь|квест)",
        r"баг[уеиов]?",
        r"глюч[аеит]{0,6}",
        r"технич[еская]{0,6}\s+проблем",
        r"провалился\s+под\s+(карту|текстур)",
        r"bug\s+report",
        r"game\s+crash",
        r"server\s+crash",
        r"qata\s+bor",
        r"xatolik\s+bor",
        r"ойын\s+шығып\s+кетеді",
        r"қате\s+бар",
    ],
    "account_problem": [
        r"взломали\s+(мой\s+)?акк(аунт)?",
        r"потерял\s+доступ",
        r"не\s+могу\s+(войти|зайти)",
        r"восстановить\s+акк(аунт)?",
        r"сменить\s+пароль",
        r"забыл\s+пароль",
        r"двухфакторн",
        r"2fa",
        r"гугл\s+аутентификатор",
        r"account\s+(hacked|stolen|lost|recovery)",
        r"can'?t\s+(log\s+in|login|sign\s+in)",
        r"аккаунтымды\s+(ұрлады|бұзды)",
        r"кіре\s+алмай\s+жатырмын",
        r"akkauntimni\s+(o'g'irlash|buzish)",
        r"parolni\s+unutdim",
    ],
    "property_problem": [
        r"пропал(а|и|о)?\s+(машин|тачк|авто|дом|бизнес|инвентарь|вещи|деньги|оружие|вирты)",
        r"исчез(ла|ли|ло)?\s+(машин|тачк|авто|дом|бизнес|имущество)",
        r"слетел(а|и|о)?\s+(тюнинг|багажник|инвентарь|дом|бизнес)",
        r"потерял\s+(машин|тачк|авто|дом|бизнес|деньги|вещи)",
        r"(car|vehicle|house|property|items?|money)\s+(missing|disappeared|lost|gone)",
        r"мүлкім\s+жоғалды",
        r"көлігім\s+жоғалып\s+кетті",
        r"mulkim\s+yo'qoldi",
        r"mashinam\s+yo'qolib\s+qoldi",
    ],
    "complaint": [
        r"жалоб[аую]",
        r"жб\s+на",
        r"репорт\s+на",
        r"сообщ(аю|аем)\s+о\s+нарушении",
        r"нарушает?\s+правила",
        r"оскорбл(яет|ил|ял)",
        r"оск\s+(родн|матер|в\s+чат)",
        r"убил\s+без\s+причины",
        r"задавил\s+машиной",
        r"читер|спидхак|аимбот|телепорт|гм\s+мод",
        r"complaint\s+(about|against)",
        r"reporting\s+(a\s+)?player",
        r"i\s+want\s+to\s+report",
        r"шағым\s+береміз?",
        r"ереже\s+бұзды",
        r"shikoyat\s+qilam",
        r"qoida\s+buzdi",
    ],
    "question": [
        r"(как|где|когда|почему|зачем|какой\s+командой)\s+[а-яa-z]",
        r"подскаж[иет]{0,4}",
        r"помог[ите]{0,4}",
        r"вопрос",
        r"не\s+понимаю",
        r"объясн[иет]{0,4}",
        r"(how|where|when|why|what\s+command)\s+",
        r"help\s+me",
        r"can\s+you\s+help",
        r"қалай",
        r"қайда",
        r"айтып\s+жіберіңізші",
        r"qanday",
        r"qayerda",
        r"yordam\s+bering",
    ],
    "technical_support": [
        r"техническ(ая|ую)\s+(поддержку?|помощь)",
        r"проблем[аыу]\s+с\s+(игрой|сервером|входом|аккаунтом|лаунчером)",
        r"не\s+работает",
        r"technical\s+support",
        r"having\s+issues?\s+with",
        r"техподдержка",
    ],
}

# Slang & typo normalization map (slang/typo -> standard form for classification & intent detection)
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
    "наебали": "обманули",
    "обули": "обманули",
    # RP violations & Slang
    "дм": "dm убийство без причины",
    "рдм": "rdm массовое убийство без причины",
    "дб": "db наезд машиной",
    "ск": "sk spawn kill",
    "тк": "tk team kill",
    "нонрп": "nonrp нарушение рп",
    "нрп": "nrp нарушение",
    "пг": "pg powergaming",
    "мг": "mg metagaming",
    "рк": "rk revengekill",
    "жб": "жалоба",
    "фрапс": "видеозапись доказательство",
    "откат": "видеозапись доказательство",
    "пруфы": "доказательства скриншоты",
    "пруф": "доказательство скриншот",
    "оск": "оскорбление",
    "оск род": "оскорбление родных",
    "мамку": "родных",
    # Cheating
    "читер": "игрок с читами",
    "читерит": "использует читы",
    "читак": "читер использует читы",
    "софтер": "использует читы софт",
    "сх": "speedhack спидхак",
    "вх": "wallhack чит",
    "гм": "godmode бессмертие чит",
    "хакер": "игрок со сторонним ПО",
    "спидхакер": "использует speedhack",
    "аимбот": "aimbot",
    # Account
    "взломали акк": "взломали аккаунт",
    "акк": "аккаунт",
    # Bans & typos
    "раздемы": "разбан",
    "разбань": "разбаньте",
    "разбнаьте": "разбаньте",
    "забанели": "забанили",
    "обманулы": "обманули",
    "оскарбили": "оскорбили",
    "адменистратор": "администратор",
    "адменистратора": "администратора",
    "пазавите": "позовите",
    "вызавите": "вызовите",
    "падключите": "подключите",
    "скреншот": "скриншот",
    "даказательства": "доказательства",
    # Vehicles & Economy
    "тачку": "машину",
    "тачка": "машина",
    "тачки": "машины",
    "гелик": "машина",
    "бэху": "машину",
    "кк": "миллионов виртов",
    "вирты": "игровая валюта",
    # Generic bugs
    "баг": "ошибка/баг",
    "глюк": "баг/ошибка",
    "фризит": "зависает",
    "крашится": "вылетает",
    # Kazakh slang/mixed
    "мені машинамды алып кетті": "меня обманули при торговле машиной",
    "алдап кетті": "обманул при сделке",
    "балағаттады": "оскорбил в чате",
    "админ шақырыңызшы": "позовите администратора",
    # Uzbek slang/mixed
    "aldab ketdi": "обманул и ушёл",
    "so'kdi": "оскорбил в чате",
    "admin chaqiring": "позовите администратора",
}

# Admin call triggers — fast check
ADMIN_CALL_KEYWORDS = frozenset([
    "администратор", "администратора", "админ", "админа", "адм", "admin", "administrator",
    "оператор", "оператора", "operator", "модератор", "модератора",
    "человек", "человека", "человеком", "human", "person",
    "шақыр", "шақырыңыз", "шақырыңызшы", "қосыңыз",
    "chaqir", "chaqiring", "chaqiringlar", "ulab",
])


def normalize_slang(text: str) -> str:
    """
    Normalize game slang and typos to help classification and intent detection.
    IMPORTANT: Original text is preserved — this is used only for analysis.
    """
    result = text
    # Sort longer keys first so multi-word phrases match before substrings
    for slang, normalized in sorted(SLANG_NORMALIZATION.items(), key=lambda x: len(x[0]), reverse=True):
        if " " in slang:
            result = re.sub(re.escape(slang), normalized, result, flags=re.IGNORECASE)
        else:
            result = re.sub(rf"\b{re.escape(slang)}\b", normalized, result, flags=re.IGNORECASE)
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

    # 1. Explicit check for direct admin call patterns first (highest priority when player asks for a live human)
    admin_matched = []
    for pattern in INTENT_PATTERNS["admin_request"]:
        for source in (lower_text, normalized):
            if re.search(pattern, source, re.IGNORECASE):
                admin_matched.append(pattern)
                break
    if admin_matched:
        return "admin_request", 0.96, ["Прямой запрос игрока на подключение живого администратора"]

    # Fast-path keyword combination for inverted word order (e.g., "админа мне быстро позови", "живой нужен оператор")
    words = set(re.findall(r"[а-яёәіңғүұқөһa-z']+", normalized))
    if words & ADMIN_CALL_KEYWORDS:
        admin_action_stems = (
            "позов", "вызов", "приглас", "подключ", "перевед", "свяж",
            "нужен", "нужна", "надо", "хочу", "дайте", "живой", "живого", "настоящ",
            "call", "speak", "talk", "connect", "live", "real", "human",
            "шақыр", "тірі", "нағыз", "қос", "сөйлес",
            "chaqir", "jonli", "haqiqiy", "gaplash", "ulab",
        )
        # Ensure it's not "админ меня забанил" (ban appeal) unless they also ask to call/connect an admin
        explicit_call_stems = (
            "позов", "вызов", "приглас", "подключ", "перевед", "свяж",
            "живой", "живого", "настоящ", "не бот", "ботом",
            "call", "speak", "talk", "connect", "human", "real person", "live admin",
            "шақыр", "тірі адам", "қосыңыз",
            "chaqir", "jonli odam", "ulab bering",
        )
        if any(stem in normalized for stem in explicit_call_stems) and any(stem in normalized for stem in admin_action_stems):
            return "admin_request", 0.94, ["Распознан прямой вызов живого администратора (с учётом порядка слов/сленга)"]

    # 2. Score remaining intents
    scores: Dict[str, Tuple[float, List[str]]] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        if intent == "admin_request":
            continue
        matched = []
        for pattern in patterns:
            for source in (lower_text, normalized):
                if re.search(pattern, source, re.IGNORECASE):
                    matched.append(pattern)
                    break
        if matched:
            raw_score = min(1.0, len(matched) / max(1.0, len(patterns) * 0.22))
            scores[intent] = (raw_score, matched)

    if not scores:
        category_intent_map = {
            "ban_appeal": "appeal",
            "scam_fraud": "scam_report",
            "technical_bug": "bug_report",
            "account_recovery": "account_problem",
            "billing_refund": "question",
            "gameplay_question": "question",
            "insult_toxicity": "complaint",
            "rp_violation": "complaint",
            "cheating_exploits": "complaint",
        }
        if category_hint and category_hint in category_intent_map:
            return category_intent_map[category_hint], 0.55, [f"intent derived from category: {category_hint}"]
        return "other", 0.30, []

    best_intent = max(scores, key=lambda k: scores[k][0])
    best_score, best_signals = scores[best_intent]

    if "scam_report" in scores and "complaint" in scores:
        best_intent = "scam_report"
        best_score = max(scores["scam_report"][0], scores["complaint"][0])
        best_signals = scores["scam_report"][1] + scores["complaint"][1]

    confidence = max(0.45, min(0.98, best_score))
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
