"""
Comprehensive test suite for Grand Mobile Support Copilot (Decentrathon 6.0).
Covers all required scenarios from Section 8 and Admin Workspace RBAC:
1. Regular gameplay/mechanics question (`report.question`);
2. Insult complaint with evidence (`complaint.insult`);
3. Complaint without sufficient evidence (`ЖБ 1.2.2` / `ЖБ 1.1.2`);
4. Payment / donate (`report.payment`) and admin appeal (`complaint.admin`) requests;
5. Unknown rule ID referenced by player;
6. Spam / offtopic (`report.offtopic`) and duplicate tickets;
7. Multilingual ('ru', 'kk', 'uz', 'en', 'und') and mixed-language tickets;
8. Empty or malformed requests (HTTP 422 validation);
9. LLM error / unavailability and unsafe LLM output fallback;
10. JSON schema compliance, real rule ID verification, and unconfirmed violation handling;
11. Batch CLI predict and evaluation metrics;
12. Role-based access control (`junior_admin` vs `senior_admin`) on backend `/api/ws/*`.
"""
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, Optional, Tuple

from app.cli import compute_macro_f1, run_predict
from app.generator import LLMClient
from app.main import app, triage_service, workspace_store
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
        ]
        + extra_headers,
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
                "kind": "report",
                "messages": [
                    {
                        "role": "player",
                        "text": "Подскажите пожалуйста, какой командой открыть навигатор GPS и где купить лицензию?",
                        "attachments": 0,
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        res = self._assert_valid_schema_and_rules(data)
        self.assertEqual(res.language, "ru")
        self.assertEqual(res.category, "report.question")
        self.assertEqual(res.priority, "p3")
        self.assertFalse(res.needs_human)

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
        self.assertEqual(res.category, "complaint.insult")
        self.assertEqual(res.priority, "p2")
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
        self.assertTrue(any(r in res.rule_ids for r in ("ЖБ 1.2.2", "ЖБ 1.1.2")))
        # Draft must request additional details and NOT claim the complaint is false
        lower_draft = res.draft_response.lower()
        self.assertTrue(
            "дополнительные сведения" in lower_draft or "скриншот" in lower_draft
        )
        self.assertIn("не означает, что жалоба ложная", lower_draft)

    # 4. Обращение о возврате денег / платежах или разбане / жалобе на администратора
    def test_04_refund_and_unban_requests(self) -> None:
        # 4a. Billing / Payment request
        status_ref, data_ref, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "report",
                "messages": [
                    {
                        "role": "player",
                        "text": "Не пришёл донат 1500 рублей на 3 сервере, ник Alex_Mercer, чек оплаты приложил, верните пожалуйста деньги",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status_ref, 200)
        res_ref = self._assert_valid_schema_and_rules(data_ref)
        self.assertEqual(res_ref.category, "report.payment")
        self.assertEqual(res_ref.priority, "p0")
        self.assertTrue(res_ref.needs_human)
        self.assertNotIn("мы вернем деньги", res_ref.draft_response.lower())

        # 4b. Unban / Admin complaint request
        status_ban, data_ban, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "complaint",
                "messages": [
                    {
                        "role": "player",
                        "text": "Администратор забанил меня на 10 сервере просто так, не согласен с баном, ник Danil_Volkov, разбаньте пожалуйста!",
                        "attachments": 1,
                    }
                ],
            },
        )
        self.assertEqual(status_ban, 200)
        res_ban = self._assert_valid_schema_and_rules(data_ban)
        self.assertEqual(res_ban.category, "complaint.admin")
        self.assertEqual(res_ban.priority, "p1")
        self.assertTrue(res_ban.needs_human)
        self.assertIn("3.4", res_ban.rule_ids)
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
        self.assertNotIn("99.9", res.rule_ids)
        self.assertNotIn("14.88", res.rule_ids)
        self.assertTrue(res.needs_human)
        self.assertIn("99.9", res.draft_response)
        self.assertIn("отсутствует", res.draft_response.lower())

    # 6. Спам и повторные обращения (дубликаты)
    def test_06_spam_and_duplicate_tickets(self) -> None:
        # 6a. Spam flood in report
        status_spam, data_spam, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {
                "kind": "report",
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
        self.assertEqual(res_spam.category, "report.offtopic")
        self.assertEqual(res_spam.priority, "p3")
        self.assertIn("3.1", res_spam.rule_ids)

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
        self.assertEqual(res_kk.category, "complaint.insult")
        self.assertIn("Сәлеметсіз бе", res_kk.draft_response)

        # 7c. Uzbek (uz)
        req_uz = TriageRequest(
            kind="complaint",
            messages=[
                MessageItem(
                    role="player",
                    text="Salom, mening Sardor_Uz akkauntimni 4-serverda admin nohaq ban qildi, iltimos bandan chiqaring, skrinshot bor",
                    attachments=1,
                )
            ],
        )
        res_uz = self.service.analyze(req_uz)
        self.assertEqual(res_uz.language, "uz")
        self.assertEqual(res_uz.category, "complaint.admin")
        self.assertIn("Assalomu alaykum", res_uz.draft_response)

        # 7d. Undetermined language (symbols/numbers only)
        req_und = TriageRequest(
            kind="report",
            messages=[MessageItem(role="player", text="1234567890 ??? !!!", attachments=0)],
        )
        res_und = self.service.analyze(req_und)
        self.assertEqual(res_und.language, "und")

    # 8. Пустой или некорректный запрос
    def test_08_empty_or_invalid_requests(self) -> None:
        st1, err1, _ = call_asgi(app, "POST", "/triage", {"kind": "complaint", "messages": []})
        self.assertEqual(st1, 422)
        self.assertEqual(err1.get("error"), "validation_error")

        st2, err2, _ = call_asgi(
            app,
            "POST",
            "/triage",
            {"kind": "complaint", "messages": [{"role": "player", "text": "   \n\t  "}]},
        )
        self.assertEqual(st2, 422)
        self.assertEqual(err2.get("error"), "validation_error")

        st3, err3, _ = call_asgi(app, "POST", "/triage", {"kind": "complaint"})
        self.assertEqual(st3, 422)

    # 9. Ошибка модели или недоступность LLM + защита от галлюцинаций LLM
    def test_09_llm_error_and_unsafe_output_fallback(self) -> None:
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
            ["p1", "p2", "p3", "p0"],
            ["p1", "p2", "p3", "p0"],
        )
        self.assertEqual(macro_f1, 1.0)


class TestWorkspaceRBAC(unittest.TestCase):
    """Server-side RBAC, workflow lifecycle, and safety tests for `/api/ws/*`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.valid_rule_ids = triage_service.rag.valid_rule_ids
        cls.valid_categories = triage_service.taxonomy.valid_categories
        cls.valid_priorities = triage_service.taxonomy.valid_priorities
        workspace_store.persist_path = None
        workspace_store.reset()

    def setUp(self) -> None:
        workspace_store.reset()

    def _call(self, user: str, method: str, path: str, body: Optional[Any] = None):
        return call_asgi(app, method, path, body, headers={"X-Demo-User": user})

    def test_01_authentication_and_demo_mode(self) -> None:
        st, _, _ = call_asgi(app, "GET", "/api/ws/me")
        self.assertEqual(st, 401)

        st, me_jr, _ = self._call("junior_1", "GET", "/api/ws/me")
        self.assertEqual(st, 200)
        self.assertEqual(me_jr["role"], "junior_admin")
        self.assertTrue(me_jr["demo_mode"])
        self.assertFalse(me_jr["can_manage_settings"])
        self.assertFalse(me_jr["can_review"])
        self.assertFalse(me_jr["can_edit_rules"])

        st, me_sr, _ = self._call("senior_1", "GET", "/api/ws/me")
        self.assertEqual(st, 200)
        self.assertEqual(me_sr["role"], "senior_admin")
        self.assertTrue(me_sr["can_manage_settings"])
        self.assertTrue(me_sr["can_review"])

    def test_02_junior_cannot_access_senior_endpoints_or_others_tickets(self) -> None:
        st, err, _ = self._call("junior_1", "GET", "/api/ws/tickets?scope=escalations")
        self.assertEqual(st, 403)
        self.assertEqual(err["detail"]["error"], "forbidden")

        st, _, _ = self._call("junior_1", "GET", "/api/ws/settings")
        self.assertEqual(st, 403)
        st, _, _ = self._call("junior_1", "PUT", "/api/ws/settings", {"junior_can_view_unassigned": False})
        self.assertEqual(st, 403)

        other_tid = next(t["id"] for t in workspace_store.tickets.values() if t.get("assignee") == "junior_2")
        st, _, _ = self._call("junior_1", "GET", f"/api/ws/tickets/{other_tid}")
        self.assertEqual(st, 403)

        my_tid = next(
            t["id"] for t in workspace_store.tickets.values()
            if t.get("assignee") == "junior_1" and t["status"] == "in_progress"
        )
        st, _, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{my_tid}/assign", {"assignee_id": "junior_2"})
        self.assertEqual(st, 403)
        for act in ("approve", "return", "close"):
            st, _, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{my_tid}/review", {"action": act, "basis": "тестовое основание"})
            self.assertEqual(st, 403)

    def test_03_junior_take_request_info_escalate_and_senior_return_approve(self) -> None:
        unassigned_tid = next(
            t["id"] for t in workspace_store.tickets.values()
            if t.get("assignee") is None and t["status"] == "new"
        )
        st, t, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/take")
        self.assertEqual(st, 200)
        self.assertEqual(t["assignee"], "junior_1")
        self.assertEqual(t["status"], "in_progress")

        workspace_store.tickets[unassigned_tid]["triage"]["needs_human"] = True
        st, _, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/send", {"text": "Здравствуйте! Проверим."})
        self.assertEqual(st, 403)

        st, t, _ = self._call(
            "junior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/request-info",
            {"text": "Здравствуйте! Пришлите, пожалуйста, скриншот с /time и номер сервера."},
        )
        self.assertEqual(st, 200)
        self.assertEqual(t["status"], "awaiting_player")

        st, t, _ = self._call(
            "junior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/escalate",
            {"reason": "Требуется проверка логов старшим администратором"},
        )
        self.assertEqual(st, 200)
        self.assertEqual(t["status"], "escalated")

        st, _, _ = self._call("junior_1", "PUT", f"/api/ws/tickets/{unassigned_tid}/draft", {"text": "Новый текст"})
        self.assertEqual(st, 403)

        st, t, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/review",
            {"action": "return", "comment": "Уточните точное время и сервер у игрока"},
        )
        self.assertEqual(st, 200)
        self.assertEqual(t["status"], "returned")

        st, t, _ = self._call(
            "junior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/escalate",
            {"reason": "Данные уточнены, передаю повторно старшему"},
        )
        self.assertEqual(st, 200)
        self.assertEqual(t["status"], "escalated")

        # Senior cannot confirm punishment without human evidence verification and valid rule IDs
        st, _, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/review",
            {
                "action": "approve",
                "decision": "punishment_confirmed",
                "basis": "ИИ написал, что игрок виновен",
                "evidence_verified": False,
                "rule_ids": ["2.1"],
            },
        )
        self.assertEqual(st, 422)

        # Senior cannot cite non-existent rule ID
        st, _, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/review",
            {
                "action": "approve",
                "decision": "punishment_confirmed",
                "basis": "Проверены скриншоты и логи чата сервера",
                "evidence_verified": True,
                "evidence_reference": "логи чата сервера #5 за 14:20",
                "rule_ids": ["99.9"],
            },
        )
        self.assertEqual(st, 422)

        # Senior approves after verifying evidence with real rule ID
        st, t, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{unassigned_tid}/review",
            {
                "action": "approve",
                "decision": "punishment_confirmed",
                "basis": "Лично проверены логи чата и скриншот с /time",
                "evidence_verified": True,
                "evidence_reference": "логи чата сервера #5, скриншот #1",
                "rule_ids": ["2.1"],
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(t["status"], "resolved")
        self.assertTrue(t["review"]["evidence_verified"])

    def test_04_safety_filter_blocks_unsafe_send_and_assistant_works(self) -> None:
        my_tid = next(
            t["id"] for t in workspace_store.tickets.values()
            if t.get("assignee") == "junior_1" and t["status"] == "in_progress"
        )
        st, _, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{my_tid}/send",
            {"text": "Здравствуйте! Мы вас разбаним и вернем деньги по п. 77.7!"},
        )
        self.assertEqual(st, 422)

        st, res, _ = self._call(
            "junior_1", "POST", "/api/ws/assistant/analyze",
            {"text": "Игрок Oleg_Smirnov оскорблял меня в чате на 3 сервере, скрин есть", "kind": "complaint", "attachments": 1},
        )
        self.assertEqual(st, 200)
        self.assertTrue(res["rules_found"])
        self.assertTrue(all(v["exists"] for v in res["rule_verification"]))

        st, saved, _ = self._call(
            "junior_1", "POST", "/api/ws/assistant/save",
            {
                "text": "Игрок Oleg_Smirnov оскорблял меня в чате на 3 сервере, скрин есть",
                "kind": "complaint",
                "attachments": 1,
                "draft": res["triage"]["draft_response"],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(saved["id"], workspace_store.tickets)

    def test_05_audit_and_analytics_scoping(self) -> None:
        st, jr_audit, _ = self._call("junior_1", "GET", "/api/ws/audit")
        self.assertEqual(st, 200)
        self.assertEqual(jr_audit["scope"], "own")
        self.assertTrue(all(e["actor_id"] == "junior_1" for e in jr_audit["entries"]))

        st, sr_audit, _ = self._call("senior_1", "GET", "/api/ws/audit")
        self.assertEqual(st, 200)
        self.assertEqual(sr_audit["scope"], "team")

        st, jr_an, _ = self._call("junior_1", "GET", "/api/ws/analytics")
        self.assertIsNone(jr_an["team"])
        st, sr_an, _ = self._call("senior_1", "GET", "/api/ws/analytics")
        self.assertIsNotNone(sr_an["team"])

    def test_06_full_report_workflow_unread_and_notifications(self) -> None:
        """
        End-to-end verification of ЗАДАЧА 3, ЗАДАЧА 4, ЗАДАЧА 5:
        1. Incoming report arrives (POST /api/ws/tickets).
        2. AI automatically classifies category, language, priority, and retrieves verified rules.
        3. Junior admin lists unread reports, opens ticket (marks as read, logs view_ticket).
        4. Junior checks draft with verified rules, takes ticket, edits draft.
        5. Junior escalates to Senior with mandatory reason.
        6. Senior sees new escalation notification, history, and AI recommendations.
        7. Senior approves with evidence verification and real rule ID.
        8. Full ticket action history is recorded.
        9. Non-destructive migration preserves state.
        """
        # Step 1: New report arrives
        payload = {
            "text": "Игрок Denis_Voronov упомянул родных и оскорблял в голосовом чате на 2 сервере, есть скриншот /time",
            "kind": "complaint",
            "attachments": 1,
            "reported_nick": "Denis_Voronov",
        }
        st, created, _ = self._call("junior_1", "POST", "/api/ws/tickets", payload)
        self.assertEqual(st, 200)
        tid = created["id"]
        self.assertEqual(created["status"], "new")
        self.assertEqual(created["kind"], "complaint")
        self.assertTrue(created["unread"])

        # Step 2: AI determined category, language, and priority
        triage = created["triage"]
        self.assertEqual(triage["language"], "ru")
        self.assertIn(triage["category"], {"complaint.insult", "complaint.other"})
        self.assertIn(triage["priority"], {"p1", "p0", "p2"})
        self.assertTrue(len(triage["rule_ids"]) > 0)
        for rid in triage["rule_ids"]:
            self.assertIn(rid, self.valid_rule_ids)

        # Step 3: Junior filters unread tickets and opens the report
        st, unread_list, _ = self._call("junior_1", "GET", "/api/ws/tickets?unread=true")
        self.assertEqual(st, 200)
        self.assertTrue(any(t["id"] == tid for t in unread_list["tickets"]))

        st, opened, _ = self._call("junior_1", "GET", f"/api/ws/tickets/{tid}")
        self.assertEqual(st, 200)
        # Opened ticket is now marked read for junior_1
        self.assertFalse(opened["unread"])
        self.assertIn("junior_1", workspace_store.tickets[tid]["read_by"])
        # History contains ticket_created and view_ticket
        hist_actions = [e["action"] for e in opened["history"]]
        self.assertIn("ticket_created", hist_actions)
        self.assertIn("view_ticket", hist_actions)

        # Step 4: Junior takes ticket and saves draft
        st, taken, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{tid}/take")
        self.assertEqual(st, 200)
        self.assertEqual(taken["status"], "in_progress")
        self.assertEqual(taken["assignee"], "junior_1")

        new_draft = "Здравствуйте! Ваша жалоба на Denis_Voronov принята к рассмотрению. Ожидайте вердикта."
        st, drafted, _ = self._call("junior_1", "PUT", f"/api/ws/tickets/{tid}/draft", {"text": new_draft})
        self.assertEqual(st, 200)
        self.assertEqual(drafted["draft"], new_draft)

        # Step 5: Junior escalates to Senior
        esc_reason = "Сложный случай: требуется проверка логов голосового чата и аудиозаписи старшим"
        st, escalated, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{tid}/escalate", {"reason": esc_reason})
        self.assertEqual(st, 200)
        self.assertEqual(escalated["status"], "escalated")
        self.assertEqual(escalated["escalation"]["reason"], esc_reason)

        # Step 6: Senior sees escalation notification
        st, sr_an, _ = self._call("senior_1", "GET", "/api/ws/analytics")
        self.assertEqual(st, 200)
        self.assertGreaterEqual(sr_an["new_escalations_count"], 1)
        esc_notifs = [n for n in sr_an["notifications"] if n["ticket_id"] == tid]
        self.assertTrue(len(esc_notifs) > 0)
        self.assertTrue(esc_notifs[0]["unseen"])

        # Step 7: Senior opens ticket and reviews recommendations
        st, sr_view, _ = self._call("senior_1", "GET", f"/api/ws/tickets/{tid}")
        self.assertEqual(st, 200)
        self.assertEqual(sr_view["junior_response"], new_draft)
        self.assertIn("senior_1", sr_view["escalation"]["seen_by"])

        # Step 8: Senior approves decision with evidence verification & valid rule IDs
        rule_to_cite = triage["rule_ids"][0] if triage["rule_ids"] else "2.6"
        st, approved, _ = self._call(
            "senior_1", "POST", f"/api/ws/tickets/{tid}/review",
            {
                "action": "approve",
                "decision": "punishment_confirmed",
                "basis": "Лично прослушана аудиозапись и сверены логи сервера #2 за 16:30",
                "evidence_verified": True,
                "evidence_reference": "аудиозапись игрока и логи сервера #2",
                "rule_ids": [rule_to_cite],
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(approved["status"], "resolved")
        self.assertTrue(approved["review"]["evidence_verified"])

        # Step 9: History persistence check
        st, hist, _ = self._call("senior_1", "GET", f"/api/ws/tickets/{tid}/history")
        self.assertEqual(st, 200)
        actions = [e["action"] for e in hist]
        self.assertIn("ticket_created", actions)
        self.assertIn("view_ticket", actions)
        self.assertIn("take", actions)
        self.assertIn("save_draft", actions)
        self.assertIn("escalate", actions)
        self.assertIn("approve", actions)

        # Step 10: Toggle read/unread endpoint
        st, unread_toggle, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{tid}/read", {"unread": True})
        self.assertEqual(st, 200)
        self.assertTrue(unread_toggle["unread"])
        st, read_toggle, _ = self._call("junior_1", "POST", f"/api/ws/tickets/{tid}/read", {"unread": False})
        self.assertEqual(st, 200)
        self.assertFalse(read_toggle["unread"])


if __name__ == "__main__":
    unittest.main()
