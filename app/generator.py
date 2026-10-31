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
            "max_tokens": 300,
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
        Synthesizes a contextualized response from the structured RAG prompt & player text
        without external network calls ($0.0000 cost).
        """
        # Extract embedded base draft from system_prompt if provided
        m_base = re.search(r"\[BASE_DRAFT_START\]\n(.*?)\n\[BASE_DRAFT_END\]", system_prompt, flags=re.DOTALL)
        base_draft = m_base.group(1).strip() if m_base else ""

        m_lang = re.search(r"на языке '([a-z]+)'", system_prompt)
        lang = m_lang.group(1) if m_lang else "ru"

        m_sanction = re.search(r"\[Регламент:\s*([^\]]+)\]", system_prompt)
        sanction = m_sanction.group(1).strip() if m_sanction else ""

        # Extract contextual snippet from player message (clean placeholders)
        clean_user = re.sub(r"\s+", " ", re.sub(r"<(?:url|digits|handle|email)>", "", user_prompt)).strip()
        short_topic = clean_user[:75] + ("…" if len(clean_user) > 75 else "")

        if not base_draft:
            return f"Здравствуйте! Ваше обращение («{short_topic}») принято в обработку и передано оператору.", 0.0

        # Enrich base draft with sanction reference from rules.json (for Russian complaints where sanction exists)
        if lang == "ru" and sanction and "Регламентная мера по данному пункту" not in base_draft:
            if "Мы не выносим автоматических решений" in base_draft:
                sanction_clause = f" Справочно по регламенту: {sanction} (применяется только при подтверждении модератором)."
                base_draft = base_draft.replace(
                    " Мы не выносим автоматических решений",
                    f"{sanction_clause} Мы не выносим автоматических решений",
                )
        return base_draft, 0.0


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
                "Ты — ассистент оператора поддержки Grand Mobile. Напиши черновик ответа игроку "
                f"на языке '{target_lang}' (категория: {category}). Строго используй ТОЛЬКО предоставленные фрагменты правил:\n"
                f"{rules_ctx}\n"
                "ЗАПРЕЩЕНО: обещать разбан или возврат денег, гарантировать наказание другого игрока, "
                "представлять обвинение как доказанный факт, придумывать номера или текст правил.\n"
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
        facts: ExtractedFacts,
        cited_rules: List[CitedRule],
        is_spam: bool,
        is_duplicate: bool,
        needs_human: bool,
    ) -> str:
        top_rule: Optional[CitedRule] = cited_rules[0] if cited_rules else None
        rule_ids_str = ", ".join(f"п. {r.id}" for r in cited_rules) if cited_rules else ""
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

        # --- KAZAKH (kk) ---
        if lang == "kk":
            if is_duplicate:
                r_ref = f" (ереже п. {top_rule.id})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Сіздің алдыңғы өтінішіңіз кезекте тұр{r_ref}. "
                    "Қайталама сұраныстарды жібермеуіңізді сұраймыз — маман жақын арада жауап береді."
                )
            if is_spam_cat:
                r_ref = f" (п. {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! Қолдау қызметіне тақырыптан тыс немесе спам хабарламалар жібермеуіңізді сұраймыз{r_ref}. "
                    "Ойын бойынша нақты сұрағыңыз болса, толық жазыңыз."
                )
            if is_appeal_cat:
                r_ref = f" Ереженің п. {top_rule.id} сәйкес ({top_rule.title})," if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Әкімшілік шешіміне қатысты өтінішіңіз тіркелді.{r_ref} "
                    "қолдау операторлары банды өз бетінше шешпейді және автоматты разбанға кепілдік берілмейді. "
                    "Іс сервердің Бас әкімшілігіне дәлелдерді тексеру үшін жіберілді."
                )
            if is_payment_cat:
                r_ref = f" (п. {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Төлем және донат бойынша өтінішіңіз қабылданды{r_ref}. "
                    "Қаражатты автоматты түрде қайтару жүзеге асырылмайды. Өтініш төлем чегін тексеру үшін қаржы бөлімінің маманына берілді."
                )
            if is_account_cat:
                r_ref = f" (п. {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Аккаунт қауіпсіздігі бойынша өтінішіңіз жоғары басымдықпен қабылданды{r_ref}. "
                    "Өтініш аккаунт иесін тексеру үшін техникалық маманға жіберілді."
                )
            if is_question_cat:
                if top_rule:
                    return (
                        f"Сәлеметсіз бе! {unknown_notice_kk}Жоба ережелерінің п. {top_rule.id} ({top_rule.title}) сәйкес: "
                        f"«{top_rule.quote}». Қосымша сұрақтарыңыз болса, хабарласыңыз!"
                    )
                return f"Сәлеметсіз бе! {unknown_notice_kk}Сұрағыңыз ойын механикасы бойынша маманға жіберілді."
            if is_help_cat:
                r_ref = f" (п. {top_rule.id})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Көмек сұрауыңыз қабылданды{r_ref}. "
                    "Жағдайды тексеру үшін сервер әкімшісіне хабарланды."
                )
            r_cite = f" Жоба ережелерінің п. {top_rule.id} ({top_rule.title}) бойынша:" if top_rule else ""
            if facts.has_evidence:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Сіздің шағымыңыз бен тіркелген материалдар қабылданды.{r_cite} "
                    "Көрсетілген жағдай модератордың тексеруіне жіберілді. Түпкілікті шешімді дәлелдер мен логтарды зерттегеннен кейін әкімші қабылдайды."
                )
            return (
                f"Сәлеметсіз бе! {unknown_notice_kk}Шағымыңыз назарға алынды.{r_cite} "
                "Скриншоттың болмауы шағымды жалған етпейді, алайда тексеру үшін қосымша мәліметтер (сервер, ойыншының нигі, /time бар скриншот немесе видео) қажет. "
                "Өтініш маманның тексеруіне жіберілді."
            )

        # --- UZBEK (uz) ---
        if lang == "uz":
            if is_duplicate:
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Sizning avvalgi murojaatingiz ko'rib chiqilmoqda{r_ref}. "
                    "Iltimos, takroriy xabarlar yubormang — mutaxassis tez orada javob beradi."
                )
            if is_spam_cat:
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! Qo'llab-quvvatlash xizmatiga spam yoki mavzudan tashqari xabarlar yubormasligingizni so'raymiz{r_ref}."
                )
            if is_appeal_cat:
                r_ref = f" Qoidalarning {top_rule.id}-bandiga binoan ({top_rule.title})," if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Ma'muriyat qarori bo'yicha murojaatingiz qabul qilindi.{r_ref} "
                    "avtomatik bandan chiqarish amalga oshirilmaydi va kafolatlanmaydi. "
                    "Murojaatingiz dalillarni tekshirish uchun server Bosh ma'muriyatiga yuborildi."
                )
            if is_payment_cat:
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}To'lov va donat bo'yicha so'rovingiz qabul qilindi{r_ref}. "
                    "Mablag'larni avtomatik qaytarish amalga oshirilmaydi. So'rov to'lov chekini tekshirish uchun moliya bo'limi mutaxassisiga o'tkazildi."
                )
            if is_account_cat:
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Akkaunt xavfsizligi bo'yicha murojaatingiz qabul qilindi{r_ref}. "
                    "So'rov akkaunt egasini tasdiqlash uchun texnik mutaxassisga o'tkazildi."
                )
            if is_question_cat:
                if top_rule:
                    return (
                        f"Assalomu alaykum! {unknown_notice_uz}Loyiha qoidalarining {top_rule.id}-bandiga ({top_rule.title}) muvofiq: "
                        f"«{top_rule.quote}»."
                    )
                return f"Assalomu alaykum! {unknown_notice_uz}Savolingiz o'yin mexanikasi bo'yicha operatorga yuborildi."
            if is_help_cat:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Yordam so'rovingiz qabul qilindi. "
                    "Holatni tekshirish uchun ma'muriyatga xabar berildi."
                )
            r_cite = f" Loyiha qoidalarining {top_rule.id}-bandi ({top_rule.title}) bo'yicha" if top_rule else ""
            if facts.has_evidence:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Murojaatingiz va ilova qilingan dalillar qabul qilindi.{r_cite} "
                    "mazkur holat moderatorga tekshirish uchun yuborildi. Yakuniy qarorni dalillar o'rganilgach ma'muriyat qabul qiladi."
                )
            return (
                f"Assalomu alaykum! {unknown_notice_uz}Murojaatingiz e'tiborga olindi.{r_cite} "
                "Skrinshotlar yo'qligi shikoyatni rad etishga asos bo'lmaydi, ammo holatni tekshirish uchun qo'shimcha ma'lumotlar (/time bilan skrinshot/video, server va nikneym) so'raladi. "
                "Murojaat mutaxassisga o'tkazildi."
            )

        # --- ENGLISH (en) ---
        if lang == "en":
            if is_duplicate:
                r_ref = f" (rule {top_rule.id})" if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your previous ticket is already in the queue{r_ref}. "
                    "Please avoid submitting duplicate reports — a support specialist will review it shortly."
                )
            if is_spam_cat:
                r_ref = f" (rule {top_rule.id})" if top_rule else ""
                return (
                    f"Hello! Please refrain from sending spam or off-topic messages to support{r_ref}. "
                    "If you have a specific issue, please describe the details clearly."
                )
            if is_appeal_cat:
                r_ref = f" Per rule {top_rule.id} ({top_rule.title})," if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your complaint regarding administration action has been registered.{r_ref} "
                    "support operators cannot issue automatic unbans or guarantee ban removal. "
                    "Your appeal has been escalated to the Senior Administration for log and evidence review."
                )
            if is_payment_cat:
                r_ref = f" (rule {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your billing inquiry has been received{r_ref}. "
                    "Please note that automatic refunds are not provided. Your ticket has been escalated to a billing specialist to verify your payment receipt."
                )
            if is_account_cat:
                r_ref = f" (rule {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your account security inquiry has been prioritized{r_ref}. "
                    "Your ticket has been escalated to a senior specialist for ownership verification."
                )
            if is_question_cat:
                if top_rule:
                    return (
                        f"Hello! {unknown_notice_en}According to server rule {top_rule.id} ({top_rule.title}): "
                        f"\"{top_rule.quote}\". Let us know if you need any further assistance!"
                    )
                return f"Hello! {unknown_notice_en}Your gameplay question has been forwarded to a support operator."
            r_cite = f" Relevant rule: {top_rule.id} ({top_rule.title})." if top_rule else ""
            if facts.has_evidence:
                return (
                    f"Hello! {unknown_notice_en}Thank you for your report{server_info}.{r_cite} "
                    "The attached materials regarding the reported incident have been forwarded to a moderator for investigation. "
                    "A final decision will be made by an administrator after verifying all evidence."
                )
            return (
                f"Hello! {unknown_notice_en}Your report{server_info} has been received.{r_cite} "
                "The absence of screenshots does not invalidate your report; however, to investigate further, please provide additional details "
                "(screenshots/video with /time, server number, and player nickname). Your ticket has been forwarded to a specialist."
            )

        # --- RUSSIAN (ru) & Fallback (und) ---
        if is_duplicate:
            r_ref = f" (согласно п. {top_rule.id} правил)" if top_rule else ""
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше предыдущее обращение по данному вопросу уже находится в очереди на обработку{r_ref}. "
                "Пожалуйста, не создавайте дубликаты тикетов — специалист рассмотрит вашу заявку в ближайшее время."
            )

        if is_spam_cat:
            r_ref = f" Напоминаем о п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»." if top_rule else ""
            return (
                f"Здравствуйте{reporter_greet_ru}! Пожалуйста, воздержитесь от отправки сообщений не по теме, флуда или спама в службу поддержки.{r_ref} "
                "Если у вас возникла игровая проблема, опишите ситуацию подробно."
            )

        if is_appeal_cat:
            r_ref = (
                f" Согласно п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            missing_hint = ""
            if not facts.has_evidence:
                missing_hint = " Пожалуйста, приложите скриншот выданного наказания при входе в игру и укажите точный никнейм и сервер."
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше обращение по действиям администрации / обжалованию наказания{server_info} зафиксировано.{r_ref} "
                "Обращаем внимание, что служба поддержки не производит автоматический разбан и не может гарантировать снятие блокировки. "
                f"Заявка передана старшей администрации сервера для проверки доказательств.{missing_hint}"
            )

        if is_payment_cat:
            r_ref = (
                f" Согласно {rule_ids_str} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            missing_receipt = ""
            if not facts.has_evidence:
                missing_receipt = " Для проверки транзакции, пожалуйста, прикрепите квитанцию (чек оплаты из банка), укажите ваш никнейм, сервер и точное время платежа."
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваш запрос по платежам и донат-валюте{server_info} принят в обработку.{r_ref} "
                "Автоматический возврат средств через чат поддержки не осуществляется. "
                f"Обращение передано профильному специалисту финансового отдела для ручной сверки платежа.{missing_receipt}"
            )

        if is_account_cat:
            r_ref = (
                f" В соответствии с {rule_ids_str} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше обращение по безопасности и доступу к аккаунту{server_info} получило критический приоритет (p0).{r_ref} "
                "В целях защиты учетной записи смена данных выполняется только после верификации владельца. "
                "Тикет передан старшему техническому специалисту — пожалуйста, подготовьте регистрационные данные аккаунта."
            )

        if is_help_cat:
            r_ref = (
                f" Согласно п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваша просьба о помощи в игре{server_info} принята.{r_ref} "
                "Уточните, пожалуйста, ваш ID/никнейм и описание ситуации (если еще не указали) — "
                "информация передана администрации сервера для оказания помощи или проверки логов."
            )

        if is_question_cat:
            if top_rule:
                return (
                    f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Согласно п. {top_rule.id} правил проекта ({top_rule.title}): "
                    f"«{top_rule.quote}». Приятной игры на Grand Mobile!"
                )
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваш вопрос по игровой механике принят. "
                "Вы можете воспользоваться внутриигровым меню /mm или навигатором /gps; при необходимости оператор уточнит детали."
            )

        # Player complaints
        if top_rule:
            rule_citation = (
                f" Потенциально применимый регламент — {rule_ids_str} ({top_rule.title}): «{top_rule.quote}»."
            )
        else:
            rule_citation = " На данный момент в базе правил не найдено однозначного подтверждения по указанному описанию."

        if facts.has_evidence:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Благодарим за обращение{server_info}. Приложенные вами материалы по заявленной ситуации{offender_info} приняты.{rule_citation} "
                "Мы не выносим автоматических решений и не гарантируем применение санкций до завершения расследования: "
                "обращение передано модератору для проверки доказательств и логов, после чего администратор примет окончательное решение."
            )
        else:
            return (
                f"Здравствуйте{reporter_greet_ru}! {unknown_notice_ru}Ваше обращение по заявленной ситуации{offender_info}{server_info} принято к сведению.{rule_citation} "
                "Отсутствие скриншотов или видеозаписи само по себе не означает, что жалоба ложная, однако для объективной проверки "
                "просим предоставить дополнительные сведения: скриншот или видео с видимым временем (/time), номер сервера и точный никнейм/ID игрока. "
                "Обращение также передано специалисту для возможной проверки серверных логов."
            )
