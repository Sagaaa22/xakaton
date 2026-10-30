"""
Draft response generator for Grand Mobile Support Copilot.
Combines optional LLM generation (with strict RAG context & safety verification)
and a deterministic rule-grounded multilingual template generator ('ru', 'kk', 'uz', 'en').
"""
from typing import Callable, List, Optional, Set, Tuple
import httpx

from app.config import settings
from app.safety import SafetyGuard
from app.schemas import CitedRule, ExtractedFacts


class LLMClient:
    """Optional OpenAI-compatible LLM client with timeout and cost estimation."""

    def __init__(
        self,
        enabled: bool = False,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 3.5,
        custom_handler: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ) -> None:
        self.enabled = enabled and (bool(api_key) or custom_handler is not None)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.custom_handler = custom_handler

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """
        Call LLM to generate a draft response.
        Returns (draft_text, cost_usd). Raises RuntimeError on network/API failure.
        """
        if self.custom_handler is not None:
            return self.custom_handler(system_prompt, user_prompt)

        if not self.enabled or not self.api_key:
            raise RuntimeError("LLM provider is disabled or API key is not configured")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 280,
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
        intent: Optional[str] = None,
        admin_request_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Generate safe operator draft response and return (draft_text, cost_usd).
        """
        target_lang = language if language in ("ru", "kk", "uz", "en") else "ru"

        # 1. Try LLM if enabled
        if self.llm_client and self.llm_client.enabled:
            rules_ctx = "\n".join(
                f"- Пункт {r.id} ({r.title}): {r.quote} [Регламент: {r.punishment}]"
                for r in cited_rules
            ) or "Подтвержденных пунктов правил не найдено."
            sys_prompt = (
                "Ты — ассистент оператора поддержки Grand Mobile. Напиши черновик ответа игроку "
                f"на языке '{target_lang}'. Строго используй ТОЛЬКО предоставленные фрагменты правил:\n"
                f"{rules_ctx}\n"
                "ЗАПРЕЩЕНО: обещать разбан или возврат денег, гарантировать наказание другого игрока, "
                "представлять обвинение как доказанный факт, придумывать номера или текст правил."
            )
            try:
                llm_draft, cost = self.llm_client.generate(sys_prompt, player_text)
                is_safe, violations = SafetyGuard.validate_draft(
                    llm_draft,
                    valid_rule_ids=valid_rule_ids,
                    unknown_user_rules=facts.unknown_mentioned_rules,
                )
                if is_safe:
                    reasons.append("Черновик сгенерирован через LLM и успешно прошел проверку безопасности")
                    return llm_draft, cost
                else:
                    reasons.append(
                        f"Ответ LLM отклонен фильтром безопасности ({', '.join(violations)}); использован безопасный шаблон"
                    )
            except Exception as exc:
                reasons.append(
                    f"LLM недоступна или вернула ошибку ({type(exc).__name__}) — выполнен автоматический переход на локальный генератор"
                )

        # 2. Deterministic rule-grounded multilingual template generator
        draft = self._build_template_draft(
            lang=target_lang,
            category=category,
            facts=facts,
            cited_rules=cited_rules,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
            needs_human=needs_human,
            intent=intent,
            admin_request_id=admin_request_id,
        )
        return draft, 0.0

    def _build_template_draft(
        self,
        lang: str,
        category: str,
        facts: ExtractedFacts,
        cited_rules: List[CitedRule],
        is_spam: bool,
        is_duplicate: bool,
        needs_human: bool,
        intent: Optional[str] = None,
        admin_request_id: Optional[str] = None,
    ) -> str:
        top_rule: Optional[CitedRule] = cited_rules[0] if cited_rules else None
        rule_ids_str = ", ".join(f"п. {r.id}" for r in cited_rules) if cited_rules else ""
        server_info = f" (сервер #{facts.server})" if facts.server else ""
        offender_info = f" в отношении игрока {facts.offender_nickname}" if facts.offender_nickname else ""

        # Notice about unknown rules mentioned by player
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

        # Direct call for live human administrator
        if intent == "admin_request":
            ar_ref = f" #{admin_request_id}" if admin_request_id else ""
            if lang == "kk":
                r_ref = f" (п. {top_rule.id})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Тірі әкімшіні қосу туралы сұранысыңыз{ar_ref} тіркелді{r_ref} "
                    "және Telegram арқылы кезекші әкімшілерге жіберілді. Күйі: әкімшінің қосылуын күтуде. "
                    "Маман жақын арада диалогқа қосылады."
                )
            if lang == "uz":
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Jonli administratorni ulash bo'yicha so'rovingiz{ar_ref} qabul qilindi{r_ref} "
                    "va Telegram orqali navbatchi administratorlarga yuborildi. Holat: administrator ulanishi kutilmoqda. "
                    "Mutaxassis tez orada suhbatga qo'shiladi."
                )
            if lang == "en":
                r_ref = f" (rule {top_rule.id})" if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your request to connect with a live administrator{ar_ref} has been created{r_ref} "
                    "and sent to the on-duty admin team via Telegram. Status: waiting for an administrator to join. "
                    "A specialist will connect to your ticket shortly."
                )
            r_ref = f" (регламент п. {top_rule.id})" if top_rule else ""
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваш запрос на подключение живого администратора{ar_ref}{server_info} "
                f"зарегистрирован{r_ref} и передан дежурным администраторам в Telegram. "
                "Статус обращения: ожидание подключения администратора. Специалист подключится к диалогу в ближайшее время."
            )

        # --- KAZAKH (kk) ---
        if lang == "kk":
            if is_duplicate:
                r_ref = f" (ереже п. {top_rule.id})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Сіздің алдыңғы өтінішіңіз кезекте тұр{r_ref}. "
                    "Қайталама сұраныстарды жібермеуіңізді сұраймыз — маман жақын арада жауап береді."
                )
            if is_spam or category == "spam_offtopic":
                r_ref = f" (п. {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! Қолдау қызметіне тақырыптан тыс немесе спам хабарламалар жібермеуіңізді сұраймыз{r_ref}. "
                    "Ойын бойынша нақты сұрағыңыз болса, толық жазыңыз."
                )
            if category == "ban_appeal":
                r_ref = f" Ереженің п. {top_rule.id} сәйкес ({top_rule.title})," if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Бұғаттауды алып тастау туралы өтінішіңіз тіркелді.{r_ref} "
                    "қолдау операторлары банды өз бетінше шешпейді және автоматты разбанға кепілдік берілмейді. "
                    "Іс сервердің Бас әкімшілігіне дәлелдерді тексеру үшін жіберілді."
                )
            if category == "billing_refund":
                r_ref = f" (п. {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Төлем және донат бойынша өтінішіңіз қабылданды{r_ref}. "
                    "Қаражатты автоматты түрде қайтару жүзеге асырылмайды. Өтініш төлем чегін тексеру үшін қаржы бөлімінің маманына берілді."
                )
            if category == "gameplay_question" and top_rule:
                return (
                    f"Сәлеметсіз бе! {unknown_notice_kk}Жоба ережелерінің п. {top_rule.id} ({top_rule.title}) сәйкес: "
                    f"«{top_rule.quote}». Қосымша сұрақтарыңыз болса, хабарласыңыз!"
                )
            # Complaints (insult, rp, scam, cheats)
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
            if is_spam or category == "spam_offtopic":
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! Qo'llab-quvvatlash xizmatiga spam yoki mavzudan tashqari xabarlar yubormasligingizni so'raymiz{r_ref}."
                )
            if category == "ban_appeal":
                r_ref = f" Qoidalarning {top_rule.id}-bandiga binoan ({top_rule.title})," if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Blokirovkani bekor qilish bo'yicha murojaatingiz qabul qilindi.{r_ref} "
                    "avtomatik bandan chiqarish amalga oshirilmaydi va kafolatlanmaydi. "
                    "Murojaatingiz dalillarni tekshirish uchun server Bosh ma'muriyatiga yuborildi."
                )
            if category == "billing_refund":
                r_ref = f" ({top_rule.id}-band)" if top_rule else ""
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}To'lov va donat bo'yicha so'rovingiz qabul qilindi{r_ref}. "
                    "Mablag'larni avtomatik qaytarish amalga oshirilmaydi. So'rov to'lov chekini tekshirish uchun moliya bo'limi mutaxassisiga o'tkazildi."
                )
            if category == "gameplay_question" and top_rule:
                return (
                    f"Assalomu alaykum! {unknown_notice_uz}Loyiha qoidalarining {top_rule.id}-bandiga ({top_rule.title}) muvofiq: "
                    f"«{top_rule.quote}»."
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
            if is_spam or category == "spam_offtopic":
                r_ref = f" (rule {top_rule.id})" if top_rule else ""
                return (
                    f"Hello! Please refrain from sending spam or off-topic messages to support{r_ref}. "
                    "If you have a specific issue, please describe the details clearly."
                )
            if category == "ban_appeal":
                r_ref = f" Per rule {top_rule.id} ({top_rule.title})," if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your punishment appeal has been registered.{r_ref} "
                    "support operators cannot issue automatic unbans or guarantee ban removal. "
                    "Your appeal has been escalated to the Senior Administration for log and evidence review."
                )
            if category == "billing_refund":
                r_ref = f" (rule {top_rule.id}: {top_rule.title})" if top_rule else ""
                return (
                    f"Hello! {unknown_notice_en}Your billing inquiry has been received{r_ref}. "
                    "Please note that automatic refunds are not provided. Your ticket has been escalated to a billing specialist to verify your payment receipt."
                )
            if category == "gameplay_question" and top_rule:
                return (
                    f"Hello! {unknown_notice_en}According to server rule {top_rule.id} ({top_rule.title}): "
                    f"\"{top_rule.quote}\". Let us know if you need any further assistance!"
                )
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
                f"Здравствуйте! {unknown_notice_ru}Ваше предыдущее обращение по данному вопросу уже находится в очереди на обработку{r_ref}. "
                "Пожалуйста, не создавайте дубликаты тикетов — специалист рассмотрит вашу заявку в ближайшее время."
            )

        if is_spam or category == "spam_offtopic":
            r_ref = f" Напоминаем о п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»." if top_rule else ""
            return (
                f"Здравствуйте! Пожалуйста, воздержитесь от отправки сообщений не по теме, флуда или спама в службу поддержки.{r_ref} "
                "Если у вас возникла игровая проблема, опишите ситуацию подробно."
            )

        if category == "ban_appeal":
            r_ref = (
                f" Согласно п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            missing_hint = ""
            if not facts.has_evidence:
                missing_hint = " Пожалуйста, приложите скриншот окна блокировки при входе в игру и укажите точный никнейм и сервер."
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваше обращение по вопросу обжалования наказания{server_info} зафиксировано.{r_ref} "
                "Обращаем внимание, что служба поддержки не производит автоматический разбан и не может гарантировать снятие блокировки. "
                f"Заявка передана старшей администрации сервера для проверки доказательств.{missing_hint}"
            )

        if category == "billing_refund":
            r_ref = (
                f" Согласно {rule_ids_str} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            missing_receipt = ""
            if not facts.has_evidence:
                missing_receipt = " Для проверки транзакции, пожалуйста, прикрепите квитанцию (чек оплаты из банка), укажите ваш никнейм, сервер и точное время платежа."
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваш запрос по платежам и донат-валюте{server_info} принят в обработку.{r_ref} "
                "Автоматический возврат средств через чат поддержки не осуществляется. "
                f"Обращение передано профильному специалисту финансового отдела для ручной сверки платежа.{missing_receipt}"
            )

        if category == "account_recovery":
            r_ref = (
                f" В соответствии с {rule_ids_str} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваше обращение по безопасности и восстановлению доступа к аккаунту{server_info} получило критический приоритет.{r_ref} "
                "В целях защиты учетной записи смена данных выполняется только после верификации владельца. "
                "Тикет передан старшему техническому специалисту — пожалуйста, подготовьте регистрационные данные аккаунта."
            )

        if category == "technical_bug":
            r_ref = (
                f" Согласно п. {top_rule.id} правил ({top_rule.title}): «{top_rule.quote}»."
                if top_rule
                else ""
            )
            return (
                f"Здравствуйте! {unknown_notice_ru}Спасибо за сообщение о технической проблеме{server_info}.{r_ref} "
                "Уточните, пожалуйста, точное время происшествия, ваш никнейм и модель устройства (если еще не указали). "
                "Информация передана специалистам для проверки технических логов сервера."
            )

        if category == "gameplay_question":
            if top_rule:
                return (
                    f"Здравствуйте! {unknown_notice_ru}Согласно п. {top_rule.id} правил проекта ({top_rule.title}): "
                    f"«{top_rule.quote}». Приятной игры на Grand Mobile!"
                )
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваш вопрос передан оператору поддержки для уточнения информации по игровым механикам."
            )

        # Player complaints: insult_toxicity, rp_violation, scam_fraud, cheating_exploits
        if top_rule:
            rule_citation = (
                f" Потенциально применимый регламент — {rule_ids_str} ({top_rule.title}): «{top_rule.quote}»."
            )
        else:
            rule_citation = " На данный момент в базе правил не найдено однозначного подтверждения по указанному описанию."

        if facts.has_evidence:
            return (
                f"Здравствуйте! {unknown_notice_ru}Благодарим за обращение{server_info}. Приложенные вами материалы по заявленной ситуации{offender_info} приняты.{rule_citation} "
                "Мы не выносим автоматических решений и не гарантируем применение санкций до завершения расследования: "
                "обращение передано модератору для проверки доказательств и логов, после чего администратор примет окончательное решение."
            )
        else:
            return (
                f"Здравствуйте! {unknown_notice_ru}Ваше обращение по заявленной ситуации{offender_info}{server_info} принято к сведению.{rule_citation} "
                "Отсутствие скриншотов или видеозаписи само по себе не означает, что жалоба ложная, однако для объективной проверки "
                "просим предоставить дополнительные сведения: скриншот или видео с видимым временем (/time), номер сервера и точный никнейм/ID игрока. "
                "Обращение также передано специалисту для возможной проверки серверных логов."
            )
