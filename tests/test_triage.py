"""
Comprehensive test suite for Grand Mobile Support Copilot (Decentrathon 6.0).
Covers all required scenarios from Section 8:
1. Regular gameplay/rules question;
2. Insult complaint with evidence;
3. Complaint without sufficient evidence;
4. Refund / billing and unban / appeal requests;
5. Unknown rule ID referenced by player;
6. Spam and duplicate tickets;
7. Multilingual ('ru', 'kk', 'uz', 'en', 'und') and mixed-language tickets;
8. Empty or malformed requests (HTTP 422 validation);
9. LLM error / unavailability and unsafe LLM output fallback;
10. JSON schema compliance, real rule ID verification, and unconfirmed violation handling.
"""
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, Optional, Tuple

from app.cli import compute_macro_f1, run_predict
from app.generator import DraftGenerator, LLMClient
from app.main import app, triage_service
from app.safety import SafetyGuard
from app.schemas import MessageItem, TriageRequest, TriageResponse
from app.triage import TriageService


def call_asgi(
    asgi_app,
    method: str,
    path: str,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, Any], bytes]:
    """In-process ASGI test caller (compatible with restricted sandboxes without loopback sockets)."""
    if body is None:
        raw_body = b""
    elif isinstance(body, (bytes, bytearray)):
        raw_body = bytes(body)
    elif isinstance(body, str):
        raw_body = body.encode("utf-8")
    else:
        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")

    if "?" in path:
        path, query = path.split("?", 1)
    else:
        query = ""
    extra_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query.encode("utf-8"),
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw_body)).encode("ascii")),
        ] + extra_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    messages_in = [{"type": "http.request", "body": raw_body, "more_body": False}]
    messages_out = []

    async def receive():
        return messages_in.pop(0) if messages_in else {"type": "http.disconnect"}

    async def send(msg):
        messages_out.append(msg)

    coro = asgi_app(scope, receive, send)
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass

    status = next(m["status"] for m in messages_out if m["type"] == "http.response.start")
    resp_bytes = b"".join(m.get("body", b"") for m in messages_out if m["type"] == "http.response.body")
    try:
        parsed_json = json.loads(resp_bytes.decode("utf-8"))
    except Exception:
        parsed_json = {}
    return status, parsed_json, resp_bytes


class TestGrandMobileSupportCopilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = TriageService()
        cls.valid_rule_ids = cls.service.rag.valid_rule_ids
        cls.valid_categories = cls.service.taxonomy.valid_categories
        cls.valid_priorities = cls.service.taxonomy.valid_priorities
        cls._weights_path = Path("models/classifier_weights.json")
        cls._orig_weights = cls._weights_path.read_text(encoding="utf-8") if cls._weights_path.exists() else None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._orig_weights and cls._weights_path:
            cls._weights_path.write_text(cls._orig_weights, encoding="utf-8")

    def setUp(self) -> None:
        self.service.extractor.reset_cache()
        triage_service.extractor.reset_cache()
        from app.fraud_detector import fraud_detector
        fraud_detector.reset()

    def _assert_valid_schema_and_rules(self, data: Dict[str, Any]) -> TriageResponse:
        """Verify output strictly conforms to TriageResponse schema, taxonomy.json, and rules.json."""
        parsed = TriageResponse(**data)
        self.assertIn(parsed.language, {"ru", "kk", "uz", "en", "und"})
        self.assertIn(parsed.category, self.valid_categories)
        self.assertIn(parsed.priority, self.valid_priorities)
        self.assertGreaterEqual(parsed.confidence, 0.0)
        self.assertLessEqual(parsed.confidence, 1.0)
        self.assertIsInstance(parsed.reasons, list)
        self.assertTrue(len(parsed.reasons) > 0)
        self.assertTrue(len(parsed.draft_response.strip()) > 10)

        # Every returned rule ID must exist in rules.json
        for rid in parsed.rule_ids:
            self.assertIn(
                rid,
                self.valid_rule_ids,
                f"Returned rule_id '{rid}' does not exist in rules.json!",
            )
        for cited in parsed.cited_rules:
            self.assertIn(
                cited.id,
                self.valid_rule_ids,
                f"Cited rule '{cited.id}' does not exist in rules.json!",
            )
            self.assertTrue(len(cited.quote.strip()) > 0)

        # Safety guardrail check on draft response
        is_safe, violations = SafetyGuard.validate_draft(
            parsed.draft_response,
            valid_rule_ids=self.valid_rule_ids,
            unknown_user_rules=parsed.facts.unknown_mentioned_rules,
        )
        self.assertTrue(is_safe, f"Draft failed safety validation: {violations}")
        return parsed

    # 1. Обычный вопрос
    def test_01_regular_gameplay_question(self) -> None:
        status, data, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "question",
                "messages": [
                    {
                        "role": "player",
                        "text": "Подскажите пожалуйста, какой командой открыть навигатор GPS и показать лицензии?",
                        "attachments": 0,
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        res = self._assert_valid_schema_and_rules(data)
        self.assertEqual(res.language, "ru")
        self.assertEqual(res.category, "gameplay_question")
        self.assertEqual(res.priority, "low")
        self.assertIn("8.2", res.rule_ids)
        self.assertFalse(res.needs_human)
        self.assertIn("/gps", res.draft_response)

    # 2. Жалоба на оскорбление (с доказательствами)
    def test_02_insult_complaint_with_evidence(self) -> None:
        status, data, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок Ivan_Grozny оскорблял меня в чате на 5 сервере, скриншоты есть",
                        "attachments": 2,
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        res = self._assert_valid_schema_and_rules(data)
        self.assertEqual(res.language, "ru")
        self.assertEqual(res.category, "insult_toxicity")
        self.assertEqual(res.priority, "medium")
        self.assertTrue(res.facts.has_evidence)
        self.assertEqual(res.facts.attachments_count, 2)
        self.assertEqual(res.facts.server, 5)
        self.assertIn("Ivan_Grozny", res.facts.nicknames)
        self.assertIn("2.1", res.rule_ids)
        # Final punishment decision must be left to a human moderator
        self.assertTrue(res.needs_human)

    # 3. Обращение без достаточных доказательств
    def test_03_complaint_without_sufficient_evidence(self) -> None:
        status, data, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок Maksim_Killer убил меня без причины (ДМ) и оскорбил в чате, но скриншотов нет",
                        "attachments": 0,
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        res = self._assert_valid_schema_and_rules(data)
        self.assertFalse(res.facts.has_evidence)
        self.assertEqual(res.facts.attachments_count, 0)
        self.assertTrue(res.needs_human)
        self.assertIn("7.2", res.rule_ids)
        # Draft must request additional details and NOT claim the complaint is false
        lower_draft = res.draft_response.lower()
        self.assertTrue(
            "дополнительные сведения" in lower_draft or "скриншот" in lower_draft
        )
        self.assertIn("не означает, что жалоба ложная", lower_draft)

    # 4. Обращение о возврате денег или разбане
    def test_04_refund_and_unban_requests(self) -> None:
        # 4a. Billing / Refund request
        status_ref, data_ref, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Верните пожалуйста деньги 1500 рублей за донат на 3 сервере, ник Alex_Mercer, чек приложил",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status_ref, 200)
        res_ref = self._assert_valid_schema_and_rules(data_ref)
        self.assertEqual(res_ref.category, "billing_refund")
        self.assertEqual(res_ref.priority, "high")
        self.assertTrue(res_ref.needs_human)
        self.assertTrue(any(r in res_ref.rule_ids for r in ("6.1", "6.2")))
        # Must NOT promise refund
        self.assertNotIn("мы вернем деньги", res_ref.draft_response.lower())

        # 4b. Unban / Appeal request
        status_ban, data_ban, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "appeal",
                "messages": [
                    {
                        "role": "player",
                        "text": "Меня забанили на 10 сервере по ошибке, ник Danil_Volkov, разбаньте пожалуйста!",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status_ban, 200)
        res_ban = self._assert_valid_schema_and_rules(data_ban)
        self.assertEqual(res_ban.category, "ban_appeal")
        self.assertEqual(res_ban.priority, "high")
        self.assertTrue(res_ban.needs_human)
        self.assertIn("7.1", res_ban.rule_ids)
        # Must NOT promise unban
        self.assertNotIn("мы вас разбаним", res_ban.draft_response.lower())

    # 5. Неизвестный пункт правил
    def test_05_unknown_rule_id(self) -> None:
        status, data, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок Tommy_Vercetti нарушил правило 99.9 и п. 14.88 в чате на 2 сервере!",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        res = self._assert_valid_schema_and_rules(data)
        self.assertIn("99.9", res.facts.unknown_mentioned_rules)
        self.assertIn("14.88", res.facts.unknown_mentioned_rules)
        # Unknown rule IDs must NEVER appear in verified rule_ids
        self.assertNotIn("99.9", res.rule_ids)
        self.assertNotIn("14.88", res.rule_ids)
        self.assertTrue(res.needs_human)
        self.assertIn("99.9", res.draft_response)
        self.assertIn("отсутствует", res.draft_response.lower())

    # 6. Спам и повторные обращения (дубликаты)
    def test_06_spam_and_duplicate_tickets(self) -> None:
        # 6a. Spam flood
        status_spam, data_spam, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "question",
                "messages": [
                    {
                        "role": "player",
                        "text": "ааааааааааа 1111111111 дайте миллион виртов просто так пж",
                        "attachments": 0,
                    }
                ],
            },
        )
        self.assertEqual(status_spam, 200)
        res_spam = self._assert_valid_schema_and_rules(data_spam)
        self.assertTrue(res_spam.is_spam)
        self.assertEqual(res_spam.category, "spam_offtopic")
        self.assertEqual(res_spam.priority, "low")

        # 6b. Duplicate submission of the exact same ticket
        payload_dup = {
            "kind": "complaint",
            "messages": [
                {
                    "role": "player",
                    "text": "Игрок Kirill_Major задавил меня машиной (ДБ) на 7 сервере, видео есть",
                    "attachments": 1,
                }
            ],
        }
        status_1, data_1, _ = call_asgi(app, "POST", "/triage", payload_dup)
        status_2, data_2, _ = call_asgi(app, "POST", "/triage", payload_dup)
        self.assertEqual(status_1, 200)
        self.assertEqual(status_2, 200)
        res_1 = self._assert_valid_schema_and_rules(data_1)
        res_2 = self._assert_valid_schema_and_rules(data_2)
        self.assertFalse(res_1.is_duplicate)
        self.assertTrue(res_2.is_duplicate)
        self.assertIn("очереди", res_2.draft_response.lower())

    # 7. Смешение языков и мультиязычность (ru, kk, uz, en, und)
    def test_07_mixed_languages_and_multilingual(self) -> None:
        # 7a. Mixed Russian + English
        status_mix, data_mix, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Hello support, player Dark_Stalker killed me without reason на 5 сервере, оскорбил в чате и убежал, please check screenshot",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status_mix, 200)
        res_mix = self._assert_valid_schema_and_rules(data_mix)
        self.assertIn(res_mix.language, {"ru", "en"})
        self.assertTrue(res_mix.facts.is_mixed_language)
        self.assertGreaterEqual(len(res_mix.facts.detected_languages), 2)

        # 7b. Kazakh (kk)
        req_kk = TriageRequest(
            kind="complaint",
            messages=[
                MessageItem(
                    role="player",
                    text="Сәлем! 3 серверде Baha_Kazakh деген ойыншы мені чатта балағаттады, скриншот бар",
                    attachments=1,
                )
            ],
        )
        res_kk = self.service.analyze(req_kk)
        self.assertEqual(res_kk.language, "kk")
        self.assertEqual(res_kk.category, "insult_toxicity")
        self.assertIn("Сәлеметсіз бе", res_kk.draft_response)

        # 7c. Uzbek (uz)
        req_uz = TriageRequest(
            kind="appeal",
            messages=[
                MessageItem(
                    role="player",
                    text="Salom, mening Sardor_Uz akkauntimni 4-serverda xato ban qilishdi, iltimos bandan chiqaring, skrinshot bor",
                    attachments=1,
                )
            ],
        )
        res_uz = self.service.analyze(req_uz)
        self.assertEqual(res_uz.language, "uz")
        self.assertEqual(res_uz.category, "ban_appeal")
        self.assertIn("Assalomu alaykum", res_uz.draft_response)

        # 7d. Undetermined language (symbols/numbers only)
        req_und = TriageRequest(
            kind="question",
            messages=[MessageItem(role="player", text="1234567890 ??? !!!", attachments=0)],
        )
        res_und = self.service.analyze(req_und)
        self.assertEqual(res_und.language, "und")

    # 8. Пустой или некорректный запрос
    def test_08_empty_or_invalid_requests(self) -> None:
        # 8a. Empty messages list
        st1, err1, _ = call_asgi(app, "POST", "/triage", {"kind": "complaint", "messages": []})
        self.assertEqual(st1, 422)
        self.assertEqual(err1.get("error"), "validation_error")

        # 8b. Whitespace-only message text
        st2, err2, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {"kind": "complaint", "messages": [{"role": "player", "text": "   \n\t  "}]},
        )
        self.assertEqual(st2, 422)
        self.assertEqual(err2.get("error"), "validation_error")

        # 8c. Missing messages field
        st3, err3, _ = call_asgi(app, "POST", "/triage", {"kind": "complaint"})
        self.assertEqual(st3, 422)

    # 9. Ошибка модели или недоступность LLM + защита от галлюцинаций LLM
    def test_09_llm_error_and_unsafe_output_fallback(self) -> None:
        # 9a. LLM raises ConnectionError / Timeout -> service must not crash and must return safe fallback
        def failing_llm_handler(sys_prompt: str, user_prompt: str):
            raise ConnectionError("Simulated LLM endpoint connection timeout")

        failing_llm = LLMClient(enabled=True, custom_handler=failing_llm_handler)
        svc_with_failing_llm = TriageService(llm_client=failing_llm)

        req = TriageRequest(
            kind="complaint",
            messages=[
                MessageItem(
                    role="player",
                    text="Игрок Artem_Boss оскорблял меня в чате на 2 сервере, скриншоты есть",
                    attachments=1,
                )
            ],
        )
        res = svc_with_failing_llm.analyze(req)
        self._assert_valid_schema_and_rules(res.dict())
        self.assertTrue(
            any("LLM недоступна или вернула ошибку" in r for r in res.reasons)
        )

        # 9b. LLM returns unsafe draft (promising unban/refund & hallucinating rule 88.8) -> SafetyGuard blocks it
        def unsafe_llm_handler(sys_prompt: str, user_prompt: str):
            return (
                "Здравствуйте! Игрок действительно нарушил п. 88.8 правил! Мы вас разбаним и вернем деньги прямо сейчас, а нарушитель получит бан!",
                0.0002,
            )

        unsafe_llm = LLMClient(enabled=True, custom_handler=unsafe_llm_handler)
        svc_with_unsafe_llm = TriageService(llm_client=unsafe_llm)
        res_safe = svc_with_unsafe_llm.analyze(req)
        self._assert_valid_schema_and_rules(res_safe.dict())
        self.assertNotIn("88.8", res_safe.draft_response)
        self.assertNotIn("мы вас разбаним", res_safe.draft_response.lower())
        self.assertTrue(
            any("Ответ LLM отклонен фильтром безопасности" in r for r in res_safe.reasons)
        )

    # 10. Поведение при отсутствии подтверждений в rules.json и проверка CLI/UI эндпоинтов
    def test_10_unconfirmed_violation_and_endpoints(self) -> None:
        # Vague message with no matching rule confirmation
        req = TriageRequest(
            kind="complaint",
            messages=[
                MessageItem(
                    role="player",
                    text="На сервере вчера случилась странная непонятная ситуация возле дерева",
                    attachments=0,
                )
            ],
        )
        res = self.service.analyze(req)
        self.assertTrue(res.needs_human)

        # Check /health, /api/tickets, /api/analytics, /api/escalate, and / (UI)
        st_h, data_h, _ = call_asgi(app, "GET", "/health")
        self.assertEqual(st_h, 200)
        self.assertEqual(data_h["status"], "ok")
        self.assertGreater(data_h["rules_loaded"], 0)

        st_t, data_t, _ = call_asgi(app, "GET", "/api/tickets")
        self.assertEqual(st_t, 200)
        self.assertGreater(data_t["count"], 0)

        first_tid = data_t["tickets"][0]["id"]
        st_e, data_e, _ = call_asgi(
            app,
            "POST",
            "/api/escalate",
            {"ticket_id": first_tid, "reason": "Тестовая эскалация"},
        )
        self.assertEqual(st_e, 200)
        self.assertEqual(data_e["status"], "escalated")

        st_a, data_a, _ = call_asgi(app, "GET", "/api/analytics")
        self.assertEqual(st_a, 200)
        self.assertGreaterEqual(data_a["senior_escalated_count"], 1)

        st_ui, _, ui_bytes = call_asgi(app, "GET", "/")
        self.assertEqual(st_ui, 200)
        ui_html = ui_bytes.decode("utf-8")
        self.assertIn("Список обращений", ui_html)
        self.assertIn("Нужна проверка человеком", ui_html)
        self.assertIn("Скопировать ответ", ui_html)
        self.assertIn("Перегенерировать", ui_html)
        self.assertIn("Передать старшему специалисту", ui_html)

    # 11. Проверка пакетного CLI predict и метрик
    def test_11_batch_predict_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "pred_test.jsonl"
            in_file = Path(__file__).resolve().parent.parent / "examples" / "sample_tickets.jsonl"
            summary = run_predict(in_file, out_file)
            self.assertEqual(summary["tickets_processed"], 10)
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 10)
            for line in lines:
                obj = json.loads(line)
                self._assert_valid_schema_and_rules(obj)

        macro_f1, _ = compute_macro_f1(
            ["high", "medium", "low", "critical"],
            ["high", "medium", "low", "critical"],
        )
        self.assertEqual(macro_f1, 1.0)


    # 12. Telegram-уведомления, распределение между админами, кнопки и напоминания
    def test_12_telegram_notifications_callbacks_reminders_and_distribution(self) -> None:
        # Escalated complaint should automatically trigger Telegram notification and assign an admin
        st_1, data_1, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок ScammerPro [ID: 1042] обманул меня при обмене имущества на 1 сервере на 5 млн!",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(st_1, 200)
        res_1 = self._assert_valid_schema_and_rules(data_1)
        self.assertIsNotNone(res_1.telegram_notification)
        self.assertTrue(res_1.telegram_notification.get("assigned_admin_id"))
        btns = [
            b["text"]
            for row in res_1.telegram_notification.get("inline_keyboard", [])
            for b in row
        ]
        self.assertTrue(any("Принять жалобу" in t for t in btns))
        self.assertTrue(any("Открыть жалобу" in t for t in btns))

        # Check /api/telegram/status and /api/telegram/notifications
        st_tg, data_tg, _ = call_asgi(app, "GET", "/api/telegram/status")
        self.assertEqual(st_tg, 200)
        self.assertGreaterEqual(data_tg["total_notifications"], 1)
        self.assertTrue(len(data_tg["admins"]) >= 2)

        st_nl, data_nl, _ = call_asgi(app, "GET", "/api/telegram/notifications")
        self.assertEqual(st_nl, 200)
        self.assertGreaterEqual(data_nl["count"], 1)
        latest_tid = res_1.telegram_notification["ticket_id"]

        # Test SLA Reminder check (max_age_minutes=0 forces reminder for unaccepted tickets)
        st_rem, data_rem, _ = call_asgi(
            app, "POST", "/api/telegram/check-reminders?max_age_minutes=0"
        )
        self.assertEqual(st_rem, 200)
        self.assertGreaterEqual(data_rem["reminders_sent"], 1)

        # Test inline button callback "Принять жалобу"
        st_cb, data_cb, _ = call_asgi(
            app,
            "POST",
            "/api/telegram/callback",
            {"callback_data": f"accept_ticket:{latest_tid}", "admin_id": "senior_1", "admin_name": "Admin_Nurlan"},
        )
        self.assertEqual(st_cb, 200)
        self.assertTrue(data_cb["ok"])
        self.assertEqual(data_cb["notification"]["accepted_by_name"], "Admin_Nurlan")

    # 13. Многоязычный ИИ (сленг, опечатки, смешение языков) и прямой вызов администратора
    def test_13_multilingual_slang_typos_and_direct_admin_call(self) -> None:
        # 13a. Typos, gaming slang, and inverted word order
        st_slang, data_slang, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "адмен памагите пж чел Vanya_Cheat с читами спидхак и аимбот убил без причины дм на 2 серве фрапс есть",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(st_slang, 200)
        res_slang = self._assert_valid_schema_and_rules(data_slang)
        self.assertEqual(res_slang.category, "cheating_exploits")
        self.assertEqual(res_slang.subcategory, "cheats")
        self.assertTrue(len(res_slang.facts.slang_or_typos_normalized) > 0)

        # 13b. Direct live admin call in Russian, Kazakh, Uzbek, and English
        direct_calls = [
            ("ru", "Позовите пожалуйста живого администратора срочно, нужен человек!", "Здравствуйте"),
            ("kk", "Тірі админ шақырыңызшы өтініш, маған оператор керек!", "Сәлеметсіз бе"),
            ("uz", "Iltimos jonli admin chaqiring, menga haqiqiy operator kerak!", "Assalomu alaykum"),
            ("en", "Please connect me to a live human admin right now!", "Hello"),
        ]
        last_ar_id = None
        for expected_lang, text, expected_greeting in direct_calls:
            st_dc, data_dc, _ = call_asgi(
                app,
                "POST",
                "/triage",
                {
                    "kind": "question",
                    "messages": [{"role": "player", "text": text, "attachments": 0}],
                },
            )
            self.assertEqual(st_dc, 200)
            res_dc = self._assert_valid_schema_and_rules(data_dc)
            self.assertEqual(res_dc.language, expected_lang)
            self.assertEqual(res_dc.intent, "admin_request")
            self.assertTrue(res_dc.needs_human)
            self.assertIsNotNone(res_dc.admin_request_id)
            self.assertEqual(res_dc.admin_request_status, "WAITING")
            self.assertIn(expected_greeting, res_dc.draft_response)
            last_ar_id = res_dc.admin_request_id

        # Lifecycle of direct admin request: WAITING -> ACCEPTED -> IN_PROGRESS -> COMPLETED
        action_body = {"request_id": last_ar_id, "admin_id": "senior_1", "admin_name": "Admin_Dias"}
        st_acc, data_acc, _ = call_asgi(
            app, "POST", f"/api/admin-requests/{last_ar_id}/accept", action_body
        )
        self.assertEqual(st_acc, 200)
        self.assertEqual(data_acc["admin_request"]["status"], "ACCEPTED")

        st_start, data_start, _ = call_asgi(
            app, "POST", f"/api/admin-requests/{last_ar_id}/start", action_body
        )
        self.assertEqual(st_start, 200)
        self.assertEqual(data_start["admin_request"]["status"], "IN_PROGRESS")

        st_comp, data_comp, _ = call_asgi(
            app, "POST", f"/api/admin-requests/{last_ar_id}/complete", action_body
        )
        self.assertEqual(st_comp, 200)
        self.assertEqual(data_comp["admin_request"]["status"], "COMPLETED")

    # 14. Confidence Score, автоматическая обработка простых вопросов и детальные подкатегории
    def test_14_confidence_score_subcategories_and_auto_handling(self) -> None:
        # Simple gameplay question -> auto_handled=True, needs_human=False, high confidence_pct
        st_q, data_q, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "question",
                "messages": [
                    {
                        "role": "player",
                        "text": "Какой командой открыть навигатор /gps и меню персонажа /mm?",
                        "attachments": 0,
                    }
                ],
            },
        )
        self.assertEqual(st_q, 200)
        res_q = self._assert_valid_schema_and_rules(data_q)
        self.assertTrue(res_q.auto_handled)
        self.assertFalse(res_q.needs_human)
        self.assertGreater(res_q.confidence_pct, 50.0)

        # Subcategory detection across PG, MG, Family Insult, Bug Abuse
        subcat_cases = [
            ("Игрок Oleg_Hero один пошел против пятерых вооруженных (ПГ / PowerGaming) на 3 сервере", "pg"),
            ("Игрок Maks_Info использовал ООС информацию из дискорда в РП чате (МГ / MetaGaming)", "mg"),
            ("Игрок Toxic_Kid оскорбил родных и упомянул маму в голосовом чате на 1 сервере", "family_insult"),
            ("Игрок Bug_Master использовал багоюз и дюп виртов через текстуры гаража", "bug_abuse"),
        ]
        for msg_text, expected_subcat in subcat_cases:
            req = TriageRequest(
                kind="complaint",
                messages=[MessageItem(role="player", text=msg_text, attachments=1)],
            )
            res = self.service.analyze(req)
            self.assertEqual(res.subcategory, expected_subcat)
            self.assertTrue(len(res.subcategory_name) > 2)

    # 15. Защита от фейковых жалоб (противоречия, риск фрода, запрет авто-наказаний)
    def test_15_antifraud_contradictions_and_no_auto_punishment(self) -> None:
        st_f, data_f, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок Fake_Target обманул на 100кк на 2 сервере, скриншот приложил, но скриншотов нет, забаньте его срочно без проверки!",
                        "attachments": 0,
                    }
                ],
                "evidence_files": [
                    {
                        "filename": "proof.png",
                        "ocr_text": "Grand Mobile Server #9 | Player: Other_Guy [ID: 105]",
                    }
                ],
            },
        )
        self.assertEqual(st_f, 200)
        res_f = self._assert_valid_schema_and_rules(data_f)
        self.assertGreater(res_f.fraud_risk_score, 0)
        self.assertTrue(len(res_f.fraud_reasons) > 0)

        st_fs, data_fs, _ = call_asgi(app, "GET", "/api/fraud/stats")
        self.assertEqual(st_fs, 200)
        self.assertIn("stats", data_fs)
        self.assertTrue(data_fs.get("automated_punishment_blocked"))

        st_susp, data_susp, _ = call_asgi(app, "GET", "/api/fraud/suspicious")
        self.assertEqual(st_susp, 200)
        self.assertTrue(data_susp.get("automated_punishment_blocked"))

    # 16. Поиск повторных нарушителей по игровому ID и никнейму + статус игрового API
    def test_16_repeat_offenders_by_id_and_nickname_and_game_api(self) -> None:
        # Search by numeric in-game ID (142 -> Ivan_Grozny)
        st_id, data_id, _ = call_asgi(app, "GET", "/api/player-history/search?q=142")
        self.assertEqual(st_id, 200)
        self.assertTrue(data_id["found"])
        self.assertEqual(data_id["nickname"], "Ivan_Grozny")
        self.assertEqual(data_id["player_id"], 142)
        self.assertTrue(data_id["is_repeat_offender"])
        self.assertGreater(data_id["confirmed_violations"], 0)
        self.assertIn("unconfirmed_complaints", data_id)
        self.assertIn("punishment_history", data_id)
        self.assertIn("game_api_connected", data_id)

        # Search by nickname (ScammerPro -> player_id 1042)
        st_nick, data_nick, _ = call_asgi(app, "GET", "/api/player-history/search?q=ScammerPro")
        self.assertEqual(st_nick, 200)
        self.assertTrue(data_nick["found"])
        self.assertEqual(data_nick["player_id"], 1042)

        # List all repeat offenders
        st_ro, data_ro, _ = call_asgi(app, "GET", "/api/repeat-offenders")
        self.assertEqual(st_ro, 200)
        self.assertGreaterEqual(data_ro["count"], 3)
        self.assertIn("api_status", data_ro)

    # 17. Анализ доказательств (JPG, PNG, MP4, WEBM, OCR, сверка с текстом жалобы)
    def test_17_evidence_analyzer_jpg_png_mp4_webm_ocr_and_comparison(self) -> None:
        import base64
        import struct

        # Construct valid minimal PNG header (800x600) and MP4 ftyp header
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\rIHDR"
            + struct.pack(">II", 800, 600)
            + b"\x08\x02\x00\x00\x00"
        )
        mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + (b"\x00" * 512)

        st_ev, data_ev, _ = call_asgi(
            app,
            "POST",
            "/api/evidence/analyze",
            {
                "complaint_text": "На 5 сервере игрок ScammerPro [ID: 777] обманул меня на 2500000 руб при сделке",
                "server": 5,
                "offender_nickname": "ScammerPro",
                "offender_id": 777,
                "files": [
                    {
                        "filename": "screen_trade.png",
                        "content_base64": base64.b64encode(png_bytes).decode("ascii"),
                        "ocr_text": "Grand Mobile Server 5 | /time 18:45 | Игрок ScammerPro [777] предложил обмен 2500000 руб",
                    },
                    {
                        "filename": "fraps_capture.mp4",
                        "content_base64": base64.b64encode(mp4_bytes).decode("ascii"),
                        "ocr_text": "Server #5 /time 18:46 ID: 777 ScammerPro",
                    },
                ],
            },
        )
        self.assertEqual(st_ev, 200)
        self.assertEqual(data_ev["files_count"], 2)
        self.assertTrue(data_ev["has_image"])
        self.assertTrue(data_ev["has_video"])
        self.assertEqual(data_ev["files"][0]["resolution"], "800x600")
        self.assertIn("ScammerPro", data_ev["extracted_nicknames"])
        self.assertIn(777, data_ev["extracted_player_ids"])
        self.assertEqual(data_ev["extracted_servers"], [5])
        self.assertTrue(data_ev["has_time_stamp"])
        self.assertTrue(data_ev["is_consistent"])

        # Contradictory evidence (complaint claims Server 2, screenshot shows Server 9)
        st_bad, data_bad, _ = call_asgi(
            app,
            "POST",
            "/api/evidence/analyze",
            {
                "complaint_text": "На 2 сервере игрок Ivan_Test убил меня",
                "server": 2,
                "offender_nickname": "Ivan_Test",
                "files": [
                    {
                        "filename": "wrong_server.jpg",
                        "ocr_text": "Grand Mobile Server 9 | Other_Player [999]",
                    }
                ],
            },
        )
        self.assertEqual(st_bad, 200)
        self.assertFalse(data_bad["is_consistent"])
        self.assertGreater(len(data_bad["contradictions"]), 0)

    # 18. AI Admin Assistant, фильтры Dashboard и Feedback Loop
    def test_18_ai_admin_assistant_dashboard_filters_and_feedback_loop(self) -> None:
        st_t, data_t, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Игрок Cheat_Master [ID: 666] летал на машине с читами (телепорт) на 4 сервере, видео есть",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(st_t, 200)
        res_t = self._assert_valid_schema_and_rules(data_t)
        self.assertIsNotNone(res_t.ai_summary)
        self.assertTrue(res_t.ai_summary["final_decision_by_human"])
        self.assertTrue(len(res_t.ai_summary.get("similar_resolved_cases", [])) > 0)
        self.assertTrue(len(res_t.ai_summary.get("decision_recommendations", [])) > 0)
        self.assertTrue(len(res_t.ai_summary.get("response_variants", {})) > 0)

        # Test Dashboard filters
        st_flt, data_flt, _ = call_asgi(
            app, "GET", "/api/tickets?category=cheating_exploits&subcategory=cheats"
        )
        self.assertEqual(st_flt, 200)
        self.assertGreaterEqual(data_flt["count"], 1)
        target_tid = data_flt["tickets"][0]["id"]

        # Submit Admin Feedback Loop entry with correction
        st_fb, data_fb, _ = call_asgi(
            app,
            "POST",
            "/api/feedback",
            {
                "ticket_id": target_tid,
                "admin_id": "senior_1",
                "admin_name": "Admin_Alikhan",
                "feedback_type": "wrong_category",
                "ai_category": "rp_violation",
                "correct_category": "cheating_exploits",
                "ai_priority": "high",
                "correct_priority": "critical",
                "note": "Подтвержден спидхак и телепорт по видео",
            },
        )
        self.assertEqual(st_fb, 200)
        self.assertEqual(data_fb["status"], "recorded")
        self.assertTrue(data_fb["feedback"].get("applied_to_model"))

        st_fbl, data_fbl, _ = call_asgi(app, "GET", "/api/feedback")
        self.assertEqual(st_fbl, 200)
        self.assertGreaterEqual(data_fbl["stats"]["total_feedback"], 1)

        # Verify Workspace endpoints (/api/ws/meta, /api/ws/tickets, /api/ws/analytics)
        ws_headers = {"x-demo-user": "senior_1"}
        st_wsm, data_wsm, _ = call_asgi(app, "GET", "/api/ws/meta", headers=ws_headers)
        self.assertEqual(st_wsm, 200)
        self.assertGreater(len(data_wsm["subcategories"]), 5)

        st_wst, data_wst, _ = call_asgi(app, "GET", "/api/ws/tickets?scope=all", headers=ws_headers)
        self.assertEqual(st_wst, 200)
        self.assertGreater(data_wst["count"], 0)

        st_wsa, data_wsa, _ = call_asgi(app, "GET", "/api/ws/analytics", headers=ws_headers)
        self.assertEqual(st_wsa, 200)
        self.assertIn("ai_effectiveness", data_wsa)


if __name__ == "__main__":
    unittest.main()

