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

    def setUp(self) -> None:
        self.service.extractor.reset_cache()
        triage_service.extractor.reset_cache()

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


if __name__ == "__main__":
    unittest.main()

