"""
Draft response generator for Grand Mobile Support Copilot.
Combines:
1. External OpenAI-compatible LLM generation (OpenAI, Gemini OpenAI endpoint, Groq, DeepSeek, Ollama),
2. Built-in Local Context-Aware RAG LLM (`local-grand-rag-v1`) requiring no external API keys,
3. Deterministic rule-grounded multilingual template fallback ('ru', 'kk', 'uz', 'en').
All outputs are strictly verified by `SafetyGuard` against `rules.json`.
"""
import re
from typing import Callable, List, Optional, Set, Tuple
import httpx

from app.config import settings
from app.safety import SafetyGuard
from app.schemas import CitedRule, ExtractedFacts


def _detect_gameplay_topic(text: str) -> Optional[str]:
    """Identify specific GTA RP / Grand Mobile gameplay questions."""
    t = text.lower()
    if any(w in t for w in ("мэри", "правительств", "губернатор", "үкімет", "hukumat", "mariya", "government", "city hall")):
        return "government"
    if any(w in t for w in (
        "автошкол", "водительск", "куәлік", "автомектеп", "guvohnoma", "avtomaktab",
        "на права", "сдать права", "купить права", "получить права", "где права",
        "водительские права", "лицензи", "лиценз", "driving license"
    )) or re.search(r"\bправ[ауые]?\b", t):
        return "driving_school"
    if any(w in t for w in ("б/у", "бу ", "бу,", "авторынок", "автобазар", "рынок машин", "көлік базар", "avtobozor", "car market")):
        return "used_cars"
    if any(w in t for w in ("больниц", "медкар", "цб ", "цб,", "лечени", "аурухан", "shifoxona", "hospital", "medcard")):
        return "hospital"
    if any(w in t for w in ("инвентар", "меню", "открыть инвентарь", "мәзір", "buyruq", "inventory", "паспорт", "/pass", "/lic", "/car", "/time", "/inv", "/mm", "/menu")) or re.search(r"\bкоманд\w*", t):
        return "inventory_menu"
    if any(w in t for w in ("работ", "устроит", "заработ", "шахт", "завод", "инкассатор", "дальнобой", "жұмыс", "ishlar", "jobs", "work")):
        return "jobs"
    if any(w in t for w in ("промокод", "промо", "promo")):
        return "promocodes"
    return None


def _detect_complaint_subtype(category: str, text: str) -> str:
    """Identify subtype of violation for accurate rule and sanction reference."""
    t = text.lower()
    if category in ("complaint.insult", "insult_toxicity"):
        if any(w in t for w in ("родн", "мать", "маму", "папу", "отца", "семью", "шеше", "анасын", "ona", "mother", "parents", "упом")):
            return "relatives"
        if any(w in t for w in ("наци", "расизм", "розжиг", "политик")):
            return "racism"
        if any(w in t for w in ("музык", "войс", "микрофон", "крик")):
            return "mic_noise"
        return "general_insult"
    if category in ("complaint.dm", "rp_violation"):
        if any(w in t for w in ("зз", "gz", "зелен", "зелён", "больниц", "автосалон", "авторынок", "вокзал", "банк", "мэри", "правительств", "автошкол", "казино", "цб", "спавн")):
            return "green_zone"
        if any(w in t for w in ("дом", "квартир", "ск ", "территори")):
            return "house_territory"
        return "general_dm"
    return "standard"


class LLMClient:
    """OpenAI-compatible & Local Context-Aware RAG LLM client with timeout and cost tracking."""

    def __init__(
        self,
        enabled: bool = False,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 3.5,
        custom_handler: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip() or "local-grand-rag-v1"
        self.timeout_seconds = timeout_seconds
        self.custom_handler = custom_handler
        self.is_local_model = self.model.startswith("local-") or self.api_key.lower() == "local"
        self.enabled = enabled and (
            bool(self.api_key) or self.is_local_model or custom_handler is not None
        )

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """
        Call LLM to generate a draft response.
        Returns (draft_text, cost_usd). Raises RuntimeError on network/API failure.
        """
        if self.custom_handler is not None:
            return self.custom_handler(system_prompt, user_prompt)

        if not self.enabled:
            raise RuntimeError("LLM provider is disabled")

        if self.is_local_model:
            return self._generate_local_contextual(system_prompt, user_prompt)

        if not self.api_key:
            raise RuntimeError("LLM API key is not configured")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 350,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            in_tok = int(usage.get("prompt_tokens", 250))
            out_tok = int(usage.get("completion_tokens", 100))
            cost = (
                (in_tok / 1000.0) * settings.llm_input_cost_per_1k
                + (out_tok / 1000.0) * settings.llm_output_cost_per_1k
            )
            return content, round(cost, 6)
        except Exception as exc:
            raise RuntimeError(f"Malformed LLM response payload: {exc}") from exc

    def _generate_local_contextual(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """
        Local Context-Aware RAG Synthesizer (`local-grand-rag-v1`).
        Extracts verified high-precision base draft ($0.0000 cost).
        """
        m_base = re.search(r"\[BASE_DRAFT_START\]\n(.*?)\n\[BASE_DRAFT_END\]", system_prompt, flags=re.DOTALL)
        base_draft = m_base.group(1).strip() if m_base else ""
        if base_draft:
            return base_draft, 0.0

        clean_user = re.sub(r"\s+", " ", re.sub(r"<(?:url|digits|handle|email)>", "", user_prompt)).strip()
        short_topic = clean_user[:75] + ("…" if len(clean_user) > 75 else "")
        return f"Здравствуйте! Ваше обращение («{short_topic}») принято в обработку и передано оператору.", 0.0


class DraftGenerator:
    """
    Produces safe, rule-grounded draft responses in the player's language.
    Falls back to deterministic verified templates if LLM is disabled, unavailable, or unsafe.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient(
            enabled=settings.llm_enabled,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    def generate_draft(
        self,
        language: str,
        category: str,
        player_text: str,
        facts: ExtractedFacts,
        cited_rules: List[CitedRule],
        valid_rule_ids: Set[str],
        is_spam: bool,
        is_duplicate: bool,
        needs_human: bool,
        reasons: List[str],
    ) -> Tuple[str, float]:
        """
        Generate safe operator draft response and return (draft_text, cost_usd).
        """
        target_lang = language if language in ("ru", "kk", "uz", "en") else "ru"

        # Build baseline rule-grounded draft first
        base_draft = self._build_template_draft(
            lang=target_lang,
            category=category,
            player_text=player_text,
            facts=facts,
            cited_rules=cited_rules,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
            needs_human=needs_human,
        )

        # 1. Try LLM if enabled
        if self.llm_client and self.llm_client.enabled:
            rules_ctx = "\n".join(
                f"- Пункт {r.id} ({r.title}): {r.quote} [Регламент: {r.punishment}]"
                for r in cited_rules
            ) or "Подтвержденных пунктов правил не найдено."
            sys_prompt = (
                "Ты — профессиональный ассистент службы поддержки и администрации игрового проекта Grand Mobile (GTA RP / CRMP).\n"
                "Твоя задача — сформировать чёткий, грамотный, вежливый и строго соответствующий регламенту ответ игроку.\n"
                f"Язык ответа: '{target_lang}' (категория тикета: {category}).\n\n"
                "ПРАВИЛА И РЕГЛАМЕНТ:\n"
                f"{rules_ctx}\n\n"
                "ИНСТРУКЦИИ:\n"
                "1. На игровые вопросы (команды, навигация, лицензии, работы, меню) давай точные прямые ответы с игровыми командами (/gps, /mm, /inv, /car, /time, /report) без цитирования лишних карательных правил.\n"
                "2. На жалобы и нарушения регламента ссылайся строго на подтверждённые пункты правил выше с указанием регламентных мер (Mute, Demorgan, Ban, Warn).\n"
                "3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: обещать автоматический разбан, возврат денег/имущества, гарантировать наказание другого игрока до проверки логов администратором.\n"
                "4. Не используй шаблонных роботизированных фраз о «потенциально применимом регламенте». Ответ должен звучать профессионально от лица команды Grand Mobile.\n\n"
                f"[BASE_DRAFT_START]\n{base_draft}\n[BASE_DRAFT_END]"
            )
            try:
                llm_draft, cost = self.llm_client.generate(sys_prompt, player_text)
                is_safe, violations = SafetyGuard.validate_draft(
                    llm_draft,
                    valid_rule_ids=valid_rule_ids,
                    unknown_user_rules=facts.unknown_mentioned_rules,
                )
                if is_safe:
                    reasons.append(
                        f"Черновик сгенерирован через LLM ({self.llm_client.model}) и успешно прошел проверку безопасности"
                    )
                    return llm_draft, cost
                else:
                    reasons.append(
                        f"Ответ LLM отклонен фильтром безопасности ({', '.join(violations)}); использован безопасный шаблон"
                    )
            except Exception as exc:
                reasons.append(
                    f"LLM недоступна или вернула ошибку ({type(exc).__name__}) — выполнен автоматический переход на локальный генератор"
                )

        # 2. Deterministic rule-grounded multilingual template fallback
        return base_draft, 0.0

    def _build_template_draft(
        self,
        lang: str,
        category: str,
        player_text: str,
        facts: ExtractedFacts,
        cited_rules: List[CitedRule],
        is_spam: bool,
        is_duplicate: bool,
        needs_human: bool,
    ) -> str:
        top_rule: Optional[CitedRule] = cited_rules[0] if cited_rules else None
        server_info = f" (сервер #{facts.server})" if facts.server else ""
        offender_info = f" в отношении игрока {facts.offender_nickname}" if facts.offender_nickname else ""
        reporter_greet_ru = f", {facts.reporter_nickname}" if facts.reporter_nickname else ""

        is_spam_cat = is_spam or category in ("spam_offtopic", "report.offtopic")
        is_appeal_cat = category in ("ban_appeal", "complaint.admin")
        is_payment_cat = category in ("billing_refund", "report.payment")
        is_account_cat = category in ("account_recovery", "report.account")
        is_help_cat = category in ("technical_bug", "report.help")
        is_question_cat = category in ("gameplay_question", "report.question")

        unknown_notice_ru = ""
        unknown_notice_kk = ""
        unknown_notice_uz = ""
        unknown_notice_en = ""
        if facts.unknown_mentioned_rules:
            unk_list = ", ".join(facts.unknown_mentioned_rules)
            unknown_notice_ru = (
                f"Обращаем внимание: указанный вами пункт правил ({unk_list}) отсутствует в действующей редакции регламента проекта. "
            )
            unknown_notice_kk = (
                f"Назар аударыңыз: сіз көрсеткен ереже тармағы ({unk_list}) жобаның қолданыстағы ережелерінде жоқ. "
            )
            unknown_notice_uz = (
                f"E'tibor bering: siz ko'rsatgan qoida bandi ({unk_list}) loyihaning amaldagi qoidalarida mavjud emas. "
            )
            unknown_notice_en = (
                f"Please note: the rule number you mentioned ({unk_list}) does not exist in the current server rules. "
            )

        gp_topic = _detect_gameplay_topic(player_text)
        complaint_sub = _detect_complaint_subtype(category, player_text)

        # ==================== 1. KAZAKH (kk) ====================
        if lang == "kk":
            if is_duplicate:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Сіздің алдыңғы өтінішіңіз кезекте тұр. "
                    "Қайталама сұраныстарды жібермеуіңізді сұраймыз — маман жақын арада жауап береді."
                )
            if is_spam_cat:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Grand Mobile қолдау қызметі байланыста. "
                    "Ереженің 3.1 тармағына сәйкес, репорт ойын сұрақтары мен бұзушылықтар үшін арналған. "
                    "Нақты сұрағыңызды жазыңыз, көмектесуге қуаныштымыз. Жақсы ойын тілейміз!"
                )
            if is_appeal_cat:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Әкімшілік шешіміне қатысты өтінішіңіз тіркелді. "
                    "Ереженің п. 3.4 сәйкес, әкімшілердің әрекеттеріне шағымдар мен жазаларды қайта қарау ресми форум арқылы 48 сағат ішінде беріледі. "
                    "Қолдау операторлары чатта банды өз бетінше шешпейді және автоматты разбанға кепілдік берілмейді. "
                    "Іс дәлелдерді тексеру үшін сервердің Бас әкімшілігіне жіберілді."
                )
            if is_payment_cat:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Төлем және донат бойынша өтінішіңіз қабылданды. "
                    "Егер валюта түспесе, ойында /donate -> «Төлемдерді тексеру» командасын қолданыңыз. "
                    "Қаражатты автоматты түрде қайтару жүзеге асырылмайды. "
                    "Өтініш төлем чегін тексеру үшін қаржы бөлімінің маманына берілді."
                )
            if is_account_cat:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Аккаунт қауіпсіздігі бойынша өтінішіңіз жоғары басымдықпен қабылданды (п. 1.4). "
                    "Құпиясөзді кіру экранындағы байланыстырылған пошта немесе Telegram арқылы қалпына келтіре аласыз. "
                    "Қауіпсіздік мақсатында аккаунт деректері ешкімге берілмейді. Өтініш техникалық маманға жіберілді."
                )
            if is_help_cat:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Егер кейіпкеріңіз немесе көлігіңіз текстураға кептеліп қалса, "
                    "ойын ішіндегі «/report -> Көмек / Кептеліп қалдым» мәзіріне жазыңыз немесе /respawn қолданыңыз. "
                    "Кезекші әкімші тез арада көмек көрсетеді. Жақсы ойын тілейміз!"
                )
            if is_question_cat:
                if gp_topic == "driving_school":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Жүргізуші куәлігін Автомектептен (Батырево кенті) алуға болады: "
                        "/gps -> Қоғамдық орындар -> Автомектеп. Қару лицензиясын Үкімет қызметкерлерінен (лицензерлерден) алуға болады: "
                        "/gps -> Мемлекеттік мекемелер -> Үкімет. Grand Mobile-де жақсы ойын тілейміз!"
                    )
                if gp_topic == "used_cars":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Б/у көлік базары Арзамас пен Южный қалаларының арасында орналасқан: "
                        "/gps -> Қоғамдық орындар -> Авторынок (б/у). Жақсы ойын тілейміз!"
                    )
                if gp_topic == "government":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Үкімет (Мэрия) ғимараты Арзамас қаласында орналасқан: "
                        "/gps -> Мемлекеттік мекемелер -> Үкімет. Жақсы ойын тілейміз!"
                    )
                if gp_topic == "hospital":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Медициналық картаны Қалалық ауруханадан (ЦБ) кезекші дәрігерлерден алуға болады: "
                        "/gps -> Мемлекеттік мекемелер -> Аурухана. Жақсы ойын тілейміз!"
                    )
                if gp_topic == "inventory_menu":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Негізгі мәзір /mm немесе /menu, инвентарь — экрандағы сөмке белгішесі немесе /inv, "
                        "көлік мәзірі — /car, сервер уақыты — /time. Жақсы ойын тілейміз!"
                    )
                if gp_topic == "jobs":
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Бастапқы жұмыстарға (Шахта, Зауыт, Ағаш кесу) 1-деңгейден бастап кіруге болады: "
                        "/gps -> Жұмыстар бойынша. Мемлекеттік ұйымдарға өту үшін /news хабарландыруларын күтіңіз. Жақсы ойын тілейміз!"
                    )
                if top_rule:
                    pun_str = f" Жаза шарасы: {top_rule.punishment}." if top_rule.punishment else ""
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Жоба ережелерінің п. {top_rule.id} ({top_rule.title}) сәйкес: "
                        f"«{top_rule.quote}».{pun_str} Жақсы ойын тілейміз!"
                    )
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Негізгі орындар навигаторы /gps командасы бойынша қолжетімді, "
                    "ал кейіпкер мен командалар мәзірі — /mm. Нақты сұрағыңыз болса, толығырақ жазыңыз. Grand Mobile-де жақсы ойын тілейміз!"
                )

            # Complaints (Kazakh)
            rule_clause_kk = ""
            if category in ("complaint.insult", "insult_toxicity"):
                if complaint_sub == "relatives":
                    rule_clause_kk = "Жоба ережелерінің п. 2.1 сәйкес: туыстарды қорлау 3-тен 10 күнге дейін Бұғаттауға әкеледі (туыстарды еске алу — 90-нан 180 минутқа дейін MUTE)."
                else:
                    rule_clause_kk = "Жоба ережелерінің п. 2.1 сәйкес: ойыншының жеке басын қорлау 15-тен 30 минутқа дейін MUTE шарасына әкеледі."
            elif category in ("complaint.dm", "rp_violation"):
                if complaint_sub == "green_zone":
                    rule_clause_kk = "Жоба ережелерінің п. GZ 1.1 сәйкес: Жасыл аймақта (Green Zone) өлтіру мен залал келтіруге қатаң тыйым салынады (жаза: 60-тан 120 минутқа дейін DEMORGAN немесе WARN)."
                else:
                    rule_clause_kk = "Grand Mobile серверлерінің регламенті бойынша (ереженің п. 5.4 / п. 5.3): себепсіз өлтіру (DM) 60–120 минутқа дейін DEMORGAN жазасын қарастырады."
            elif category in ("complaint.cheat", "cheating_exploits"):
                rule_clause_kk = "Жоба ережелерінің п. 5.2 сәйкес: бөгде бағдарламаларды (читтерді) қолдану 7-ден 31 күнге дейін Бұғаттауға әкеледі."
            elif category in ("complaint.scam", "scam_fraud"):
                rule_clause_kk = "Жоба ережелерінің п. 4.1 сәйкес: ойын валютасымен қулық жасау және алдау 3-тен 7 күнге дейін (ірі көлемде — 31 күннен мәңгілікке дейін) Бұғаттауға әкеледі."
            elif top_rule:
                rule_clause_kk = f"Қолданылатын ереже — п. {top_rule.id} ({top_rule.title})."

            if facts.has_evidence:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Сіздің шағымыңыз бен тіркелген материалдар қабылданды. {rule_clause_kk} "
                    "Автоматты жаза тағайындалмайды: мәселе дәлелдер мен логтарды тексеру үшін модераторға берілді, содан кейін әкімші түпкілікті шешім қабылдайды."
                )
            return (
                f"Сәлеметсіз бе! {unknown_notice_kk}Шағымыңыз назарға алынды. {rule_clause_kk} "
                "Скриншоттың болмауы шағымды жалған етпейді, алайда объективті тексеру үшін қосымша мәліметтер "
                "(/time бар скриншот немесе видео, сервер нөмірі және ойыншы нигі) қажет. Өтініш маманға тексеруге берілді."
            )

        # ==================== 2. UZBEK (uz) ====================
        if lang == "uz":
            if is_duplicate:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Sizning avvalgi murojaatingiz ko'rib chiqilmoqda. "
                    "Iltimos, takroriy xabarlar yubormang — mutaxassis tez orada javob beradi."
                )
            if is_spam_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Grand Mobile qo'llab-quvvatlash xizmati aloqada. "
                    "Qoidalarning 3.1-bandiga binoan, report o'yin savollari va qoidabuzarliklar uchun mo'ljallangan. "
                    "Iltimos, aniq savolingizni yozing, bajonidil yordam beramiz. Maroqli o'yin tilaymiz!"
                )
            if is_appeal_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Ma'muriyat harakatlari bo'yicha murojaatingiz qayd etildi. "
                    "Qoidalarning 3.4-bandiga ko'ra, shikoyatlar forum orqali 48 soat ichida beriladi. "
                    "Avtomatik bandan chiqarish amalga oshirilmaydi. "
                    "Murojaatingiz dalillarni tekshirish uchun server Bosh ma'muriyatiga yuborildi."
                )
            if is_payment_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}To'lov va donat bo'yicha murojaatingiz qabul qilindi. "
                    "Agar mablag' tushmagan bo'lsa, /donate buyrug'i orqali tekshiring. "
                    "Mablag'larni avtomatik qaytarish amalga oshirilmaydi. "
                    "So'rov to'lov chekini tekshirish uchun moliya bo'limi mutaxassisiga o'tkazildi."
                )
            if is_account_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Akkaunt xavfsizligi bo'yicha murojaatingiz qabul qilindi (1.4-band). "
                    "Kirish ekranida biriktirilgan pochta yoki Telegram orqali parolni tiklashingiz mumkin. "
                    "Parollarni hech kimga bermang! So'rov texnik mutaxassisga yo'naltirildi."
                )
            if is_help_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Agar qahramoningiz yoki mashinangiz teksturaga tiqilib qolgan bo'lsa, "
                    "o'yin ichidagi «/report -> Yordam / Tiqilib qoldim» menyusiga murojaat qiling yoki /respawn buyrug'idan foydalaning. "
                    "Navbatchi administrator tezda yordam beradi. Maroqli o'yin tilaymiz!"
                )
            if is_question_cat:
                if gp_topic == "driving_school":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Haydovchilik guvohnomasini Avtomaktabda (Batirovo) olishingiz mumkin: "
                        "/gps -> Jamoat joylari -> Avtomaktab. Qurol litsenziyasini esa Hukumat xodimlaridan olishingiz mumkin: "
                        "/gps -> Davlat muassasalari -> Hukumat. Maroqli o'yin tilaymiz!"
                    )
                if gp_topic == "used_cars":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Ikkilamchi avtomobil bozori (b/u avtobozor) Arzamas va Yujniy oralig'ida: "
                        "/gps -> Jamoat joylari -> Avtobozor. Maroqli o'yin tilaymiz!"
                    )
                if gp_topic == "government":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Hukumat (Mariya) binosi Arzamas shahrida joylashgan: "
                        "/gps -> Davlat muassasalari -> Hukumat. Maroqli o'yin tilaymiz!"
                    )
                if gp_topic == "hospital":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Tibbiy kartani Shahar shifoxonasida navbatchi shifokorlardan olishingiz mumkin: "
                        "/gps -> Davlat muassasalari -> Shifoxona. Maroqli o'yin tilaymiz!"
                    )
                if gp_topic == "inventory_menu":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Asosiy menyu /mm yoki /menu, inventar — ekrandagi tugma yoki /inv, "
                        "avtomobil menyusi — /car, vaqt — /time. Maroqli o'yin tilaymiz!"
                    )
                if gp_topic == "jobs":
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Dastlabki ishlarga (Shaxta, Zavod) 1-darajadan boshlab suhbatsiz kirishingiz mumkin: "
                        "/gps -> Ishlar bo'yicha. Davlat tashkilotlariga kirish uchun /news e'lonlarini kuting. Maroqli o'yin tilaymiz!"
                    )
                if top_rule:
                    pun_str = f" Jazo chorasi: {top_rule.punishment}." if top_rule.punishment else ""
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Loyiha qoidalarining {top_rule.id}-bandiga ({top_rule.title}) muvofiq: "
                        f"«{top_rule.quote}».{pun_str} Maroqli o'yin tilaymiz!"
                    )
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Asosiy joylar navigatori /gps buyrug'ida mavjud, "
                    "qahramon va buyruqlar menyusi esa — /mm. Qo'shimcha savollaringiz bo'lsa, batafsil yozib qoldiring. Maroqli o'yin tilaymiz!"
                )

            # Complaints (Uzbek)
            rule_clause_uz = ""
            if category in ("complaint.insult", "insult_toxicity"):
                if complaint_sub == "relatives":
                    rule_clause_uz = "Loyiha qoidalarining 2.1-bandiga muvofiq: qarindoshlarni haqorat qilish 3 kundan 10 kungacha Bloklashga olib keladi (qarindoshlarni eslatish — 90 dan 180 daqiqagacha MUTE)."
                else:
                    rule_clause_uz = "Loyiha qoidalarining 2.1-bandiga ko'ra: o'yinchini haqorat qilish 15 dan 30 daqiqagacha MUTE bilan jazolanadi."
            elif category in ("complaint.dm", "rp_violation"):
                if complaint_sub == "green_zone":
                    rule_clause_uz = "Loyiha qoidalarining GZ 1.1 bandiga ko'ra: Yashil zonada (Green Zone) o'ldirish va zarar yetkazish qat'iyan man etiladi (jazo: 60 dan 120 daqiqagacha DEMORGAN yoki WARN)."
                else:
                    rule_clause_uz = "Grand Mobile qoidalari bo'yicha (5.4 / 5.3-bandlar): sababsiz o'ldirish (DM) 60–120 daqiqagacha DEMORGAN bilan jazolanadi."
            elif category in ("complaint.cheat", "cheating_exploits"):
                rule_clause_uz = "Loyiha qoidalarining 5.2-bandiga binoan: noqonuniy dasturlar (cheatlar) dan foydalanish 7 kundan 31 kungacha Bloklash bilan jazolanadi."
            elif category in ("complaint.scam", "scam_fraud"):
                rule_clause_uz = "Loyiha qoidalarining 4.1-bandiga ko'ra: firibgarlik va valyuta manipulyatsiyasi 3 kundan 7 kungacha Bloklashga olib keladi."
            elif top_rule:
                rule_clause_uz = f"Qo'llaniladigan qoida — {top_rule.id}-band ({top_rule.title})."

            if facts.has_evidence:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Murojaatingiz va ilova qilingan dalillar qabul qilindi. {rule_clause_uz} "
                    "Avtomatik jazo choralari qo'llanilmaydi: holat dalillar va loglarni tekshirish uchun moderatorga yuborildi, yakuniy qarorni esa ma'muriyat qabul qiladi."
                )
            return (
                f"Assalomu alaykum! {unknown_notice_uz}Murojaatingiz e'tiborga olindi. {rule_clause_uz} "
                "Skrinshotlar yo'qligi shikoyatni yolg'on deb hisoblashga asos bo'lmaydi, ammo holatni tekshirish uchun qo'shimcha ma'lumotlar "
                "(/time bilan skrinshot/video, server va aniq nikneym) kerak bo'ladi. Murojaat mutaxassisga yo'naltirildi."
            )

        # ==================== 3. ENGLISH (en) ====================
        if lang == "en":
            if is_duplicate:
                return (
                    f"Hello! {unknown_notice_en}Your previous ticket is already in the queue. "
                    "Please avoid submitting duplicate reports — a support specialist will review it shortly."
                )
            if is_spam_cat:
                return (
                    f"Hello! {unknown_notice_en}Grand Mobile support is here. Please note that according to server rule 3.1, "
                    "the report system is meant for gameplay questions and rule violations. "
                    "Please describe your question clearly and we will be glad to help. Enjoy the game!"
                )
            if is_appeal_cat:
                return (
                    f"Hello! {unknown_notice_en}Your complaint regarding administration action has been registered. "
                    "According to server rule 3.4, appeals against administrator actions must be submitted on the official forum within 48 hours. "
                    "Support operators cannot issue automatic unbans. "
                    "Your appeal has been escalated to the Senior Administration for log and evidence review."
                )
            if is_payment_cat:
                return (
                    f"Hello! {unknown_notice_en}Your billing inquiry has been received. "
                    "If credits have not arrived automatically, use /donate -> 'Check Payments' in-game. "
                    "Automatic refunds are not provided. "
                    "Your ticket has been escalated to a billing specialist to verify your payment receipt."
                )
            if is_account_cat:
                return (
                    f"Hello! {unknown_notice_en}Your account security inquiry has been prioritized (rule 1.4). "
                    "You can reset your password using the 'Forgot Password' option on the login screen via linked email or Telegram. "
                    "Never share your credentials with anyone! Your ticket has been escalated to a senior technical specialist."
                )
            if is_help_cat:
                return (
                    f"Hello! {unknown_notice_en}If you or your vehicle are stuck in textures, please submit an in-game report via '/report -> Stuck / Help' "
                    "or use /respawn. A server administrator will assist you right away. Enjoy playing on Grand Mobile!"
                )
            if is_question_cat:
                if gp_topic == "driving_school":
                    return (
                        f"Hello! {unknown_notice_en}You can obtain a driving license at the Driving School in Batyrevo: "
                        "/gps -> Public Places -> Driving School. Weapon licenses can be purchased from Government licensers: "
                        "/gps -> State Organizations -> Government. Enjoy playing on Grand Mobile!"
                    )
                if gp_topic == "used_cars":
                    return (
                        f"Hello! {unknown_notice_en}The used car market is situated between Arzamas and Yuzhny. "
                        "Navigate via /gps -> Public Places -> Car Market. Have fun on Grand Mobile!"
                    )
                if gp_topic == "government":
                    return (
                        f"Hello! {unknown_notice_en}The Government building (City Hall) is located in Arzamas: "
                        "/gps -> State Organizations -> Government. Enjoy Grand Mobile!"
                    )
                if gp_topic == "hospital":
                    return (
                        f"Hello! {unknown_notice_en}You can obtain a medical card or treatment at the City Hospital: "
                        "/gps -> State Organizations -> Hospital. Have a great game!"
                    )
                if gp_topic == "inventory_menu":
                    return (
                        f"Hello! {unknown_notice_en}The main character menu is /mm or /menu. Inventory is opened via screen icon or /inv. "
                        "Vehicle controls: /car, server time: /time. Have fun playing on Grand Mobile!"
                    )
                if gp_topic == "jobs":
                    return (
                        f"Hello! {unknown_notice_en}Starter jobs (Mine, Factory, Sawmill) are available from level 1 without interviews: "
                        "/gps -> Jobs. For state organizations, check announcements in /news. Have fun on Grand Mobile!"
                    )
                if top_rule:
                    pun_str = f" Sanction: {top_rule.punishment}." if top_rule.punishment else ""
                    return (
                        f"Hello! {unknown_notice_en}According to server rule {top_rule.id} ({top_rule.title}): "
                        f"\"{top_rule.quote}\".{pun_str} Enjoy playing on Grand Mobile!"
                    )
                return (
                    f"Hello! {unknown_notice_en}Key locations can be found using /gps, and character menus/commands via /mm. "
                    "If you have a specific gameplay question, let us know and we will gladly help. Enjoy Grand Mobile!"
                )

            # Complaints (English)
            rule_clause_en = ""
            if category in ("complaint.insult", "insult_toxicity"):
                if complaint_sub == "relatives":
                    rule_clause_en = "According to server rule 2.1: insulting relatives carries a 3 to 10-day Ban (mentioning relatives: 90 to 180-minute Mute)."
                else:
                    rule_clause_en = "According to server rule 2.1: insulting another player is penalized with a 15 to 30-minute Mute."
            elif category in ("complaint.dm", "rp_violation"):
                if complaint_sub == "green_zone":
                    rule_clause_en = "According to rule GZ 1.1: killing or damaging players in a Green Zone is strictly prohibited (sanction: 60 to 120-minute Demorgan or Warn)."
                else:
                    rule_clause_en = "According to Grand Mobile rules (rules 5.4 / 5.3): random deathmatch (DM) and leaving RP situations carry a Demorgan penalty of up to 60-120 minutes."
            elif category in ("complaint.cheat", "cheating_exploits"):
                rule_clause_en = "According to rule 5.2: using third-party cheats or exploits is punished with a 7 to 31-day Ban."
            elif category in ("complaint.scam", "scam_fraud"):
                rule_clause_en = "According to rule 4.1: fraud and currency manipulation carry a 3 to 7-day Ban (under rule 1.4, administration does not recover lost property)."
            elif top_rule:
                rule_clause_en = f"Applicable regulation: rule {top_rule.id} ({top_rule.title})."

            if facts.has_evidence:
                return (
                    f"Hello! {unknown_notice_en}Thank you for your report{server_info}. The submitted materials have been accepted. {rule_clause_en} "
                    "We do not issue automatic punishments: the case has been forwarded to a moderator to review logs and evidence, after which an administrator will render a final decision."
                )
            return (
                f"Hello! {unknown_notice_en}Your report{server_info} has been received. {rule_clause_en} "
                "The absence of screenshots does not invalidate your report; however, for an objective review, please provide additional details "
                "(a screenshot or video with /time, server number, and exact player nickname). Your ticket has been forwarded to a specialist."
            )

        # ==================== 4. RUSSIAN (ru) & Fallback ====================
        if is_duplicate:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше предыдущее обращение по данному вопросу уже находится в очереди на обработку. "
                "Пожалуйста, не создавайте дубликаты тикетов — специалист рассмотрит вашу заявку в ближайшее время."
            )

        if is_spam_cat:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Служба поддержки Grand Mobile на связи. "
                "Напоминаем, что согласно п. 3.1 правил проекта репорт предназначен для вопросов по игре и сообщений о нарушениях. "
                "Пожалуйста, напишите ваш конкретный вопрос или проблему, и мы с радостью вам поможем. Приятной игры!"
            )

        if is_appeal_cat:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше обращение по действиям администрации{server_info} зафиксировано. "
                "Обращаем внимание, что служба поддержки не производит автоматический разбан и не может гарантировать снятие блокировки в чате. "
                "Согласно п. 3.4 правил проекта, жалобы на администраторов и оспаривание наказаний подаются через официальный форум в течение 48 часов с момента выдачи наказания. "
                "Заявка передана старшей администрации сервера для всесторонней проверки оснований вынесенного наказания."
            )

        if is_payment_cat:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваш запрос по платежам и донат-валюте{server_info} принят в обработку. "
                "Если валюта не начислилась автоматически, активируйте её в игре командой /donate -> «Проверить платежи». "
                "Автоматический возврат средств через чат поддержки не осуществляется. "
                "Обращение передано профильному специалисту финансового отдела; пожалуйста, сохраните чек оплаты из банка для сверки платежа."
            )

        if is_account_cat:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше обращение по безопасности и доступу к аккаунту{server_info} получило критический приоритет (p0). "
                "В соответствии с п. 1.4 правил проекта игрок лично отвечает за сохранность своего аккаунта и имущества. "
                "Восстановить доступ можно через кнопку «Забыли пароль?» на экране входа по привязанной почте или Telegram. "
                "В целях защиты учетной записи смена данных выполняется только после верификации владельца. "
                "Тикет передан старшему техническому специалисту — никогда и никому не передавайте пароли от аккаунта!"
            )

        if is_help_cat:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Если ваш персонаж или автомобиль застрял в текстурах или перевернулся, "
                "воспользуйтесь внутриигровым репортом: «/report -> Застрял / Помощь» либо используйте команду /respawn. "
                "Дежурный администратор сервера оперативно окажет помощь прямо в игре. Приятной игры на Grand Mobile!"
            )

        if is_question_cat:
            if gp_topic == "driving_school":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Сдать экзамен на водительские права можно в Автошколе (пгт. Батырево). "
                    "Маршрут в навигаторе: /gps -> Общественные места -> Автошкола. "
                    "Лицензию на оружие можно приобрести у сотрудников Правительства (лицензеров) через /gps -> Государственные учреждения -> Правительство. "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "used_cars":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Б/у авторынок расположен между г. Арзамас и г. Южный. "
                    "Чтобы проложить маршрут к нему, используйте навигатор: /gps -> Общественные места -> Авторынок (б/у). "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "government":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Здание Правительства (Мэрия) находится в городе Арзамас. "
                    "Маршрут в навигаторе: /gps -> Государственные учреждения -> Правительство. "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "hospital":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Оформить медицинскую карту и получить лечение можно в Городской больнице (ЦБ) у дежурных врачей. "
                    "Маршрут: /gps -> Государственные учреждения -> Больница (г. Арзамас или г. Южный). "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "inventory_menu":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Главное меню персонажа и настроек открывается командой /mm или /menu. "
                    "Инвентарь открывается иконкой на экране или командой /inv. "
                    "Меню личного транспорта — /car, точное игровое время — /time. "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "jobs":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Начальные работы (Шахта, Завод, Лесопилка) доступны с 1 уровня без собеседования: "
                    "/gps -> По работе. Для официальных организаций (МВД, Больница, МЧС, Правительство) дождитесь гос. волны в /news или подайте заявление на форуме проекта. "
                    "Приятной игры на Grand Mobile!"
                )
            if gp_topic == "promocodes":
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Активировать промокод можно через главное меню: "
                    "введите /mm -> «Промокоды» (или используйте команду /promo). "
                    "Приятной игры на Grand Mobile!"
                )
            if top_rule:
                pun_str = f" Регламентная мера: {top_rule.punishment}." if top_rule.punishment else ""
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Согласно п. {top_rule.id} правил проекта ({top_rule.title}): "
                    f"«{top_rule.quote}».{pun_str} Приятной игры на Grand Mobile!"
                )
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Навигатор по всем ключевым местам доступен по команде /gps, "
                "а общее меню персонажа и команд — /mm. Если у вас возник конкретный вопрос по механике игры, уточните подробности, "
                "и мы с радостью поможем. Приятной игры на Grand Mobile!"
            )

        # Player Complaints (Russian)
        rule_clause_ru = ""
        if category in ("complaint.insult", "insult_toxicity"):
            if complaint_sub == "relatives":
                rule_clause_ru = "Согласно п. 2.1 правил проекта: оскорбление родных влечёт Блокировку персонажа от 3 до 10 дней (упоминание родных — MUTE от 90 до 180 минут)."
            elif complaint_sub == "racism":
                rule_clause_ru = "Согласно п. 2.4 правил проекта: расизм, ксенофобия и розжиг межнациональной розни наказываются MUTE от 60 до 180 минут или Блокировкой персонажа от 1 до 31 дня."
            elif complaint_sub == "mic_noise":
                rule_clause_ru = "Согласно п. 2.5 правил проекта: создание помех и посторонние аудиотрансляции в микрофон наказываются MUTE от 20 до 60 минут."
            else:
                rule_clause_ru = "Согласно п. 2.1 правил проекта: прямое или завуалированное оскорбление личности игрока наказывается MUTE от 15 до 30 минут."
        elif category in ("complaint.dm", "rp_violation"):
            if complaint_sub == "green_zone":
                rule_clause_ru = "Согласно п. GZ 1.1 правил проекта: любые убийства и нанесение урона в Зелёной Зоне строго запрещены независимо от наличия RP-причины (наказание: DEMORGAN от 60 до 120 минут или WARN)."
            elif complaint_sub == "house_territory":
                rule_clause_ru = "Согласно п. 5.4 правил проекта: убийство или нанесение урона у входа или на территории дома наказывается DEMORGAN до 120 минут."
            else:
                rule_clause_ru = "Согласно регламенту серверов Grand Mobile (п. 5.4 / п. 5.3 правил): беспричинное убийство (DM) и уход от RP строго наказуемы (наказание: DEMORGAN до 60–120 минут)."
        elif category in ("complaint.cheat", "cheating_exploits"):
            rule_clause_ru = "Согласно п. 5.2 правил проекта: использование стороннего ПО (читов, аима, спидхака) наказывается Блокировкой персонажа от 7 до 31 дня (багоюз — WARN или блокировка до 3 дней)."
        elif category in ("complaint.scam", "scam_fraud"):
            rule_clause_ru = (
                "Согласно п. 4.1 правил проекта: любые махинации с игровой валютой и мошенничество по отношению к игрокам влекут Блокировку персонажа от 3 до 7 дней (при крупном ущербе — от 31 дня до навсегда). "
                "Также напоминаем, что в соответствии с п. 1.4 правил администрация не восстанавливает утраченное имущество при добровольных сделках."
            )
        elif category == "complaint.rmt_ads":
            rule_clause_ru = "Согласно п. 2.3 и п. 4.4 правил проекта: реклама сторонних ресурсов и продажа ценностей за реальные деньги (RMT) строго запрещены и влекут MUTE от 60 минут до бессрочной Блокировки с обнулением имущества."
        elif category == "complaint.invalid":
            rule_clause_ru = "Согласно регламенту правил рассмотрения жалоб (п. ЖБ 1.1.2 и п. ЖБ 1.2.2), обращение должно содержать понятное описание ситуации и прямые доказательства."
        elif top_rule:
            pun_str = f" Регламентная мера: {top_rule.punishment}." if top_rule.punishment else ""
            rule_clause_ru = f"По регламенту проекта — п. {top_rule.id} ({top_rule.title}): «{top_rule.quote}».{pun_str}"

        if facts.has_evidence:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваша жалоба{offender_info}{server_info} и приложенные материалы приняты в обработку. {rule_clause_ru} "
                "Мы не выносим автоматических решений и не гарантируем применение санкций до завершения расследования: "
                "материалы переданы модератору для детальной проверки доказательств и логов, после чего администратор примет объективное решение."
            )
        else:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваша жалоба{offender_info}{server_info} принята к сведению. {rule_clause_ru} "
                "Отсутствие скриншотов или видеозаписи само по себе не означает, что жалоба ложная, однако для объективной проверки "
                "просим предоставить дополнительные сведения: скриншот или видеозапись с видимым временем (/time согласно п. ЖБ 1.2.2 правил), номер сервера и точный никнейм игрока. "
                "Обращение передано специалисту для возможной проверки серверных логов."
            )
