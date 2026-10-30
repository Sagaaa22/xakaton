"""
Telegram Admin Notification, Inline Button Handling, SLA Reminder & Load Distribution Engine
for Grand Mobile Support Copilot.

Features:
1. Automatic notifications for tickets requiring human admin intervention (`needs_human=True`)
   and direct live admin calls (`admin_request`).
2. Detailed formatted payload: ticket ID, player, offender, server, category/subcategory,
   priority, AI confidence, fraud risk score, repeat offender alert, cited rules, and AI summary.
3. Interactive Inline Keyboard buttons:
   - «✅ Принять жалобу» (callback_data: `accept_ticket:<id>` or `accept_ar:<id>`)
   - «🔗 Открыть жалобу» (URL link to Operator Dashboard ticket card)
4. SLA Reminder system for tickets/calls waiting too long without being accepted.
5. Smart Admin Load Balancing (distributes tickets across available junior/senior admins
   based on role specialization and current active workload).
6. Real Telegram Bot API integration via `httpx` when `TELEGRAM_BOT_TOKEN` is set in `.env`,
   with automatic local outbox persistence so notifications & button callbacks work seamlessly
   both with a live Telegram bot and in local/demo environments.
"""
from datetime import datetime, timedelta, timezone
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
import uuid

import httpx

from app.config import settings

logger = logging.getLogger("grand_mobile_copilot.telegram")


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_dt().isoformat(timespec="seconds")


PRIORITY_EMOJI = {
    "critical": "🚨 КРИТИЧЕСКИЙ",
    "high": "🔴 ВЫСОКИЙ",
    "medium": "🟡 СРЕДНИЙ",
    "low": "🟢 НИЗКИЙ",
}

SENIOR_REQUIRED_CATEGORIES = {
    "ban_appeal",
    "scam_fraud",
    "billing_refund",
    "account_recovery",
    "cheating_exploits",
}


class TelegramAdminNotifier:
    """
    Manages Telegram notifications, admin workload distribution, inline button callbacks,
    and overdue ticket reminders.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._notifications: List[Dict[str, Any]] = []
        self._ticket_notifications: Dict[str, str] = {}  # ticket_id -> notification_id
        self._ar_notifications: Dict[str, str] = {}      # admin_request_id -> notification_id

        # Admin roster synchronized with workspace DEMO_USERS + custom Telegram chat IDs from .env
        self._admins: Dict[str, Dict[str, Any]] = {
            "junior_1": {
                "id": "junior_1",
                "name": "Айдар К.",
                "role": "junior_admin",
                "role_label": "Младший администратор",
                "telegram_username": "@aidar_gm_mod",
                "telegram_chat_id": settings.telegram_admin_ids.get("junior_1", ""),
                "online": True,
                "assigned_count": 0,
                "accepted_count": 0,
                "active_tickets": [],
                "specialties": ["insult_toxicity", "rp_violation", "technical_bug", "gameplay_question"],
            },
            "junior_2": {
                "id": "junior_2",
                "name": "Мария С.",
                "role": "junior_admin",
                "role_label": "Младший администратор",
                "telegram_username": "@maria_gm_mod",
                "telegram_chat_id": settings.telegram_admin_ids.get("junior_2", ""),
                "online": True,
                "assigned_count": 0,
                "accepted_count": 0,
                "active_tickets": [],
                "specialties": ["insult_toxicity", "rp_violation", "technical_bug", "gameplay_question"],
            },
            "senior_1": {
                "id": "senior_1",
                "name": "Ерлан Б.",
                "role": "senior_admin",
                "role_label": "Старший администратор",
                "telegram_username": "@erlan_gm_senior",
                "telegram_chat_id": settings.telegram_admin_ids.get("senior_1", ""),
                "online": True,
                "assigned_count": 0,
                "accepted_count": 0,
                "active_tickets": [],
                "specialties": [
                    "ban_appeal",
                    "scam_fraud",
                    "cheating_exploits",
                    "billing_refund",
                    "account_recovery",
                ],
            },
        }
        self._rr_index = 0
        self._on_ticket_accepted_cb: Optional[Callable[[str, str, str], Any]] = None
        self._on_ar_accepted_cb: Optional[Callable[[str, str, str], Any]] = None

    def register_callbacks(
        self,
        on_ticket_accepted: Optional[Callable[[str, str, str], Any]] = None,
        on_ar_accepted: Optional[Callable[[str, str, str], Any]] = None,
    ) -> None:
        """Register hooks to update WorkspaceStore and AdminRequestStore when admin clicks Accept."""
        if on_ticket_accepted is not None:
            self._on_ticket_accepted_cb = on_ticket_accepted
        if on_ar_accepted is not None:
            self._on_ar_accepted_cb = on_ar_accepted

    def register_on_accept_ticket(self, cb: Callable[[str, str, str], Any]) -> None:
        self._on_ticket_accepted_cb = cb

    def register_on_accept_admin_request(self, cb: Callable[[str, str, str], Any]) -> None:
        self._on_ar_accepted_cb = cb

    def mark_ticket_accepted(self, ticket_id: str, admin_id: str, admin_name: str) -> None:
        with self._lock:
            notif_id = self._ticket_notifications.get(ticket_id)
            if notif_id:
                for n in self._notifications:
                    if n["id"] == notif_id and n["status"] == "pending":
                        n["status"] = "accepted"
                        n["accepted_by_id"] = admin_id
                        n["accepted_by_name"] = admin_name
                        n["accepted_at"] = _now_iso()
                        n["updated_at"] = _now_iso()
            self._record_admin_acceptance(admin_id, ticket_id)

    @property
    def is_live_bot_configured(self) -> bool:
        return bool(settings.telegram_enabled and settings.telegram_bot_token and settings.telegram_chat_id)

    # ---------------- Admin Load Balancing & Distribution ----------------

    def select_admin_for_ticket(
        self,
        category: str,
        priority: str,
        is_direct_call: bool = False,
    ) -> Dict[str, Any]:
        """
        Distribute complaint among online administrators using role rules + Least-Loaded balancing.
        - Critical priority or senior-only categories prefer `senior_admin` unless overloaded.
        - Standard complaints distribute evenly across `junior_admin`s by active load.
        """
        with self._lock:
            online_admins = [a for a in self._admins.values() if a.get("online", True)]
            if not online_admins:
                online_admins = list(self._admins.values())

            needs_senior = (
                priority == "critical"
                or category in SENIOR_REQUIRED_CATEGORIES
            )

            if needs_senior:
                seniors = [a for a in online_admins if a["role"] == "senior_admin"]
                if seniors:
                    chosen = min(seniors, key=lambda a: (len(a["active_tickets"]), a["assigned_count"]))
                    return dict(chosen)

            # For standard tickets or direct calls: pick least-loaded admin (preferring juniors first)
            juniors = [a for a in online_admins if a["role"] == "junior_admin"]
            candidates = juniors if juniors else online_admins
            # Sort by active ticket count, then total assigned count, then round-robin tie-breaker
            chosen = min(
                candidates,
                key=lambda a: (len(a["active_tickets"]), a["assigned_count"]),
            )
            return dict(chosen)

    def _record_admin_assignment(self, admin_id: str, ref_id: str) -> None:
        with self._lock:
            adm = self._admins.get(admin_id)
            if adm:
                adm["assigned_count"] += 1
                if ref_id not in adm["active_tickets"]:
                    adm["active_tickets"].append(ref_id)

    def _record_admin_acceptance(self, admin_id: str, ref_id: str) -> None:
        with self._lock:
            for aid, adm in self._admins.items():
                if aid == admin_id:
                    adm["accepted_count"] += 1
                    if ref_id not in adm["active_tickets"]:
                        adm["active_tickets"].append(ref_id)
                else:
                    if ref_id in adm["active_tickets"]:
                        adm["active_tickets"].remove(ref_id)

    def release_admin_ticket(self, ref_id: str) -> None:
        """Remove ticket/call from active workload when resolved or closed."""
        with self._lock:
            for adm in self._admins.values():
                if ref_id in adm["active_tickets"]:
                    adm["active_tickets"].remove(ref_id)

    # ---------------- Notification Formatting & Sending ----------------

    def notify_ticket_escalation(
        self,
        ticket_id: str,
        triage_data: Optional[Dict[str, Any]] = None,
        player_text: str = "",
        force: bool = False,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically send Telegram notification when a complaint requires admin attention.
        Accepts either `triage_data` dict or keyword arguments (`category`, `priority`, etc.).
        """
        if not settings.telegram_enabled:
            return None

        if triage_data is None:
            triage_data = {
                "needs_human": True,
                "category": kwargs.get("category", "rp_violation"),
                "category_name": kwargs.get("category_name", "Жалоба"),
                "subcategory_name": kwargs.get("subcategory"),
                "priority": kwargs.get("priority", "high"),
                "confidence": kwargs.get("confidence", 0.75),
                "language": kwargs.get("language", "ru"),
                "rule_ids": kwargs.get("rule_ids", []),
                "fraud_risk_score": kwargs.get("fraud_risk_score", 0),
                "needs_human_reason": kwargs.get("needs_human_reason"),
                "facts": {
                    "reporter_nickname": kwargs.get("player_nickname"),
                    "reporter_id": kwargs.get("player_id"),
                    "offender_nickname": kwargs.get("offender_nickname"),
                    "offender_id": kwargs.get("offender_id"),
                    "server": kwargs.get("server"),
                    "has_evidence": kwargs.get("has_evidence", True),
                    "attachments_count": kwargs.get("attachments_count", 1),
                },
            }
            if not player_text and kwargs.get("ai_summary_text"):
                player_text = str(kwargs["ai_summary_text"])

        needs_human = bool(triage_data.get("needs_human", True))
        if not needs_human and not force:
            return None

        with self._lock:
            if ticket_id in self._ticket_notifications and not force:
                existing_id = self._ticket_notifications[ticket_id]
                for n in self._notifications:
                    if n["id"] == existing_id:
                        return dict(n)

        category = str(triage_data.get("category") or "unknown")
        category_name = str(triage_data.get("category_name") or category)
        subcat_name = str(triage_data.get("subcategory_name") or "")
        priority = str(triage_data.get("priority") or "medium")
        confidence_pct = int(round(float(triage_data.get("confidence", 0.0)) * 100))
        language = str(triage_data.get("language") or "ru").upper()
        facts = triage_data.get("facts") or {}
        if hasattr(facts, "dict"):
            facts = facts.dict()

        reporter = facts.get("reporter_nickname") or "Не указан"
        offender = facts.get("offender_nickname") or "Не указан"
        offender_id = facts.get("offender_id") or (
            facts.get("player_ids")[0] if facts.get("player_ids") else None
        )
        server = facts.get("server")
        server_str = f"Сервер #{server}" if server is not None else "Не указан"
        has_ev = "Да" if facts.get("has_evidence") else "Нет"
        rule_ids = triage_data.get("rule_ids") or []
        rules_str = ", ".join(f"п. {r}" for r in rule_ids) if rule_ids else "Требуется уточнение"
        fraud_score = int(triage_data.get("fraud_risk_score") or 0)
        needs_reason = str(triage_data.get("needs_human_reason") or "Требуется проверка администратором")

        assigned_admin = self.select_admin_for_ticket(category=category, priority=priority)
        self._record_admin_assignment(assigned_admin["id"], ticket_id)

        cat_display = f"{category_name} ({subcat_name})" if subcat_name else category_name
        offender_display = f"{offender} (ID: {offender_id})" if offender_id else offender
        snippet = (player_text or "").strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."

        lines = [
            f"🔔 <b>НОВАЯ ЖАЛОБА ТРЕБУЕТ ВМЕШАТЕЛЬСТВА</b>",
            f"━━━━━━━━━━━━━━━━━━",
            f"🆔 <b>Тикет:</b> <code>{ticket_id}</code> | 🌐 <b>Язык:</b> {language}",
            f"⚡ <b>Приоритет:</b> {PRIORITY_EMOJI.get(priority, priority.upper())}",
            f"📂 <b>Категория:</b> {cat_display}",
            f"🎮 <b>Сервер:</b> {server_str}",
            f"👤 <b>Заявитель:</b> <code>{reporter}</code>",
            f"🎯 <b>Нарушитель:</b> <code>{offender_display}</code>",
            f"📎 <b>Доказательства:</b> {has_ev} (вложений: {facts.get('attachments_count', 0)})",
            f"📜 <b>Правила Grand Mobile:</b> {rules_str}",
            f"🤖 <b>Confidence ИИ:</b> {confidence_pct}%",
        ]
        if fraud_score >= 40:
            lines.append(f"⚠️ <b>Риск фейка (Anti-Fraud):</b> {fraud_score}/100 (проверьте достоверность!)")
        ph = triage_data.get("player_history")
        if isinstance(ph, dict) and ph.get("found") and ph.get("confirmed_violations", 0) > 0:
            lines.append(
                f"🔁 <b>Повторный нарушитель:</b> подтверждённых нарушений — {ph.get('confirmed_violations')} "
                f"(всего жалоб: {ph.get('total_complaints', 0)})"
            )
        if snippet:
            lines.append(f"💬 <b>Текст:</b> <i>«{snippet}»</i>")
        lines.extend([
            f"📌 <b>Причина вызова:</b> {needs_reason}",
            f"👨‍💻 <b>Распределено на:</b> {assigned_admin['name']} ({assigned_admin['telegram_username']})",
        ])
        text_html = "\n".join(lines)

        open_url = f"{settings.dashboard_base_url}/#junior/{ticket_id}"
        inline_keyboard = [
            [
                {"text": "✅ Принять жалобу", "callback_data": f"accept_ticket:{ticket_id}"},
                {"text": "🔗 Открыть жалобу", "url": open_url},
            ]
        ]

        return self._dispatch_notification(
            kind="ticket_escalation",
            ref_id=ticket_id,
            ticket_id=ticket_id,
            title=f"Жалоба {ticket_id} — {cat_display}",
            priority=priority,
            category=category,
            category_name=cat_display,
            player_nickname=reporter if reporter != "Не указан" else None,
            offender_nickname=offender if offender != "Не указан" else None,
            server=server,
            confidence_pct=confidence_pct,
            fraud_risk_score=fraud_score,
            assigned_admin=assigned_admin,
            text_html=text_html,
            inline_keyboard=inline_keyboard,
            open_url=open_url,
        )

    def notify_direct_admin_call(
        self,
        admin_request: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Send high-priority Telegram notification when a player explicitly requests a live human admin.
        """
        if not settings.telegram_enabled:
            return None

        req_id = str(admin_request.get("id") or f"ar_{uuid.uuid4().hex[:8]}")
        ticket_id = admin_request.get("ticket_id")
        player = admin_request.get("player_nickname") or "Игрок"
        server = admin_request.get("server")
        server_str = f"Сервер #{server}" if server is not None else "Не указан"
        lang = str(admin_request.get("language") or "ru").upper()
        summary = (admin_request.get("context_summary") or "Игрок просит подключить живого администратора").strip()
        if len(summary) > 200:
            summary = summary[:197] + "..."

        assigned_admin = self.select_admin_for_ticket(
            category="admin_request",
            priority="high",
            is_direct_call=True,
        )
        self._record_admin_assignment(assigned_admin["id"], req_id)

        lines = [
            f"🆘 <b>ПРЯМОЙ ВЫЗОВ ЖИВОГО АДМИНИСТРАТОРА</b>",
            f"━━━━━━━━━━━━━━━━━━",
            f"🆔 <b>Запрос:</b> <code>{req_id}</code>" + (f" (Тикет: <code>{ticket_id}</code>)" if ticket_id else ""),
            f"👤 <b>Игрок:</b> <code>{player}</code> | 🌐 <b>Язык:</b> {lang}",
            f"🎮 <b>Сервер:</b> {server_str}",
            f"💬 <b>Сообщение игрока:</b> <i>«{summary}»</i>",
            f"⏳ <b>Статус:</b> Ожидает подключения администратора",
            f"👨‍💻 <b>Назначен дежурный:</b> {assigned_admin['name']} ({assigned_admin['telegram_username']})",
        ]
        text_html = "\n".join(lines)

        open_url = (
            f"{settings.dashboard_base_url}/#junior/{ticket_id}"
            if ticket_id
            else f"{settings.dashboard_base_url}/#admin-calls"
        )
        inline_keyboard = [
            [
                {"text": "✅ Принять жалобу", "callback_data": f"accept_ar:{req_id}"},
                {"text": "🔗 Открыть жалобу", "url": open_url},
            ]
        ]

        return self._dispatch_notification(
            kind="direct_admin_call",
            ref_id=req_id,
            ticket_id=ticket_id,
            title=f"Прямой вызов администратора ({req_id}) от {player}",
            priority="high",
            category="admin_request",
            category_name="Прямой вызов администратора",
            player_nickname=player,
            offender_nickname=None,
            server=server,
            confidence_pct=98,
            fraud_risk_score=0,
            assigned_admin=assigned_admin,
            text_html=text_html,
            inline_keyboard=inline_keyboard,
            open_url=open_url,
        )

    def _dispatch_notification(
        self,
        kind: str,
        ref_id: str,
        ticket_id: Optional[str],
        title: str,
        priority: str,
        category: str,
        category_name: str,
        player_nickname: Optional[str],
        offender_nickname: Optional[str],
        server: Any,
        confidence_pct: int,
        fraud_risk_score: int,
        assigned_admin: Dict[str, Any],
        text_html: str,
        inline_keyboard: List[List[Dict[str, str]]],
        open_url: str,
    ) -> Dict[str, Any]:
        notif_id = f"tg_{uuid.uuid4().hex[:8]}"
        created_at = _now_iso()

        # Determine target chat_id (specific admin's telegram_chat_id or global TELEGRAM_CHAT_ID)
        target_chat_id = (
            assigned_admin.get("telegram_chat_id")
            or settings.telegram_chat_id
            or "local_dashboard_outbox"
        )

        delivered_via_api = False
        telegram_message_id: Optional[int] = None
        delivery_mode = "local_outbox"
        api_error: Optional[str] = None

        if settings.telegram_bot_token and target_chat_id != "local_dashboard_outbox":
            try:
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                payload = {
                    "chat_id": target_chat_id,
                    "text": text_html,
                    "parse_mode": "HTML",
                    "reply_markup": {"inline_keyboard": inline_keyboard},
                }
                with httpx.Client(timeout=4.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("ok"):
                        delivered_via_api = True
                        delivery_mode = "telegram_bot_api"
                        telegram_message_id = data.get("result", {}).get("message_id")
            except Exception as exc:
                api_error = str(exc)
                logger.warning("Telegram API send failed (saved to local outbox): %s", exc)

        record: Dict[str, Any] = {
            "id": notif_id,
            "kind": kind,
            "ref_id": ref_id,
            "ticket_id": ticket_id,
            "title": title,
            "priority": priority,
            "category": category,
            "category_name": category_name,
            "player_nickname": player_nickname,
            "offender_nickname": offender_nickname,
            "server": server,
            "confidence_pct": confidence_pct,
            "fraud_risk_score": fraud_risk_score,
            "assigned_admin_id": assigned_admin["id"],
            "assigned_admin_name": assigned_admin["name"],
            "assigned_admin_tg": assigned_admin["telegram_username"],
            "target_chat_id": target_chat_id,
            "text_html": text_html,
            "inline_keyboard": inline_keyboard,
            "open_url": open_url,
            "status": "pending",  # pending -> accepted -> resolved
            "accepted_by_id": None,
            "accepted_by_name": None,
            "accepted_at": None,
            "created_at": created_at,
            "updated_at": created_at,
            "reminders_sent": 0,
            "last_reminder_at": None,
            "delivered_via_api": delivered_via_api,
            "delivery_mode": delivery_mode,
            "telegram_message_id": telegram_message_id,
            "api_error": api_error,
        }

        with self._lock:
            self._notifications.append(record)
            if kind == "ticket_escalation" and ticket_id:
                self._ticket_notifications[ticket_id] = notif_id
            elif kind == "direct_admin_call":
                self._ar_notifications[ref_id] = notif_id

        logger.info(
            "telegram_notification_sent id=%s kind=%s ref=%s admin=%s mode=%s",
            notif_id,
            kind,
            ref_id,
            assigned_admin["id"],
            delivery_mode,
        )
        return dict(record)

    # ---------------- Button Callback Handling («Принять жалобу») ----------------

    def handle_callback(
        self,
        callback_data: str,
        admin_id: Optional[str] = None,
        admin_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process inline button click:
        - `accept_ticket:<ticket_id>`
        - `accept_ar:<request_id>`
        Updates notification state, admin workload, and invokes Workspace / AdminRequest hooks.
        """
        if not callback_data or ":" not in callback_data:
            raise ValueError("Некорректный формат callback_data")

        action_prefix, target_id = callback_data.split(":", 1)
        target_id = target_id.strip()

        with self._lock:
            notif: Optional[Dict[str, Any]] = None
            for n in reversed(self._notifications):
                if n["ref_id"] == target_id or n.get("ticket_id") == target_id:
                    notif = n
                    break

            resolved_admin_id = admin_id or (notif["assigned_admin_id"] if notif else "junior_1")
            adm_obj = self._admins.get(resolved_admin_id) or self._admins.get("junior_1")
            resolved_admin_name = admin_name or (adm_obj["name"] if adm_obj else "Администратор")

            if notif:
                notif["status"] = "accepted"
                notif["accepted_by_id"] = resolved_admin_id
                notif["accepted_by_name"] = resolved_admin_name
                notif["accepted_at"] = _now_iso()
                notif["updated_at"] = _now_iso()

            self._record_admin_acceptance(resolved_admin_id, target_id)

        # Trigger hooks into WorkspaceStore / AdminRequestStore
        hook_result = None
        if action_prefix == "accept_ticket" and self._on_ticket_accepted_cb:
            try:
                hook_result = self._on_ticket_accepted_cb(target_id, resolved_admin_id, resolved_admin_name)
            except Exception as exc:
                logger.warning("Hook on_ticket_accepted failed: %s", exc)
        elif action_prefix == "accept_ar" and self._on_ar_accepted_cb:
            try:
                hook_result = self._on_ar_accepted_cb(target_id, resolved_admin_id, resolved_admin_name)
            except Exception as exc:
                logger.warning("Hook on_ar_accepted failed: %s", exc)

        return {
            "ok": True,
            "status": "accepted",
            "action": action_prefix,
            "target_id": target_id,
            "accepted_by_id": resolved_admin_id,
            "accepted_by_name": resolved_admin_name,
            "notification": dict(notif) if notif else None,
            "hook_result": hook_result,
        }

    # ---------------- SLA Reminders for Unaccepted Complaints ----------------

    def check_and_send_reminders(
        self,
        max_wait_minutes: Optional[int] = None,
        max_age_minutes: Optional[int] = None,
        force_all_pending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Check all pending notifications that have not been accepted yet.
        If waiting time >= `max_wait_minutes` (or `force_all_pending=True`), send a reminder
        and reassign if needed.
        """
        eff_minutes = max_wait_minutes if max_wait_minutes is not None else max_age_minutes
        if eff_minutes == 0:
            force_all_pending = True
        threshold_min = (
            eff_minutes
            if eff_minutes is not None
            else settings.telegram_reminder_minutes
        )
        now = _now_dt()
        reminders: List[Dict[str, Any]] = []

        with self._lock:
            pending_items = [n for n in self._notifications if n["status"] == "pending"]

        for n in pending_items:
            try:
                created_dt = datetime.fromisoformat(n["created_at"])
            except Exception:
                created_dt = now - timedelta(minutes=threshold_min + 1)

            waited_minutes = max(1, int((now - created_dt).total_seconds() // 60))
            if not force_all_pending and waited_minutes < threshold_min:
                continue

            with self._lock:
                n["reminders_sent"] = int(n.get("reminders_sent", 0)) + 1
                n["last_reminder_at"] = _now_iso()
                n["updated_at"] = _now_iso()
                rem_num = n["reminders_sent"]

            reminder_text = (
                f"⏰ <b>НАПОМИНАНИЕ #{rem_num}: ЖАЛОБА ОЖИДАЕТ РАССМОТРЕНИЯ!</b>\n"
                f"🆔 <b>Объект:</b> <code>{n['ref_id']}</code> ({n['category_name']})\n"
                f"⚡ <b>Приоритет:</b> {PRIORITY_EMOJI.get(n['priority'], n['priority'])}\n"
                f"⌛ <b>Время ожидания:</b> {waited_minutes} мин. (порог SLA: {threshold_min} мин.)\n"
                f"👨‍💻 <b>Назначен:</b> {n['assigned_admin_name']} ({n['assigned_admin_tg']})\n"
                f"Пожалуйста, нажмите «✅ Принять жалобу» или откройте карточку!"
            )

            # Attempt live Telegram reminder if configured
            if settings.telegram_bot_token and n["target_chat_id"] != "local_dashboard_outbox":
                try:
                    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                    with httpx.Client(timeout=4.0) as client:
                        client.post(
                            url,
                            json={
                                "chat_id": n["target_chat_id"],
                                "text": reminder_text,
                                "parse_mode": "HTML",
                                "reply_markup": {"inline_keyboard": n["inline_keyboard"]},
                            },
                        )
                except Exception as exc:
                    logger.warning("Failed to send Telegram reminder via API: %s", exc)

            rem_entry = {
                "notification_id": n["id"],
                "ref_id": n["ref_id"],
                "ticket_id": n.get("ticket_id"),
                "priority": n["priority"],
                "assigned_admin_id": n["assigned_admin_id"],
                "assigned_admin_name": n["assigned_admin_name"],
                "waited_minutes": waited_minutes,
                "reminders_sent": rem_num,
                "reminder_text": reminder_text,
                "sent_at": _now_iso(),
            }
            reminders.append(rem_entry)
            logger.info("telegram_reminder_sent ref=%s count=%d", n["ref_id"], rem_num)

        return reminders

    # ---------------- Queries & Dashboard Stats ----------------

    def list_notifications(
        self,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        admin_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._notifications)
        if status:
            items = [n for n in items if n["status"] == status]
        if kind:
            items = [n for n in items if n["kind"] == kind]
        if admin_id:
            items = [n for n in items if n["assigned_admin_id"] == admin_id]
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return [dict(n) for n in items[:limit]]

    def get_admins_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    **adm,
                    "active_load": len(adm["active_tickets"]),
                    "active_tickets": list(adm["active_tickets"]),
                }
                for adm in self._admins.values()
            ]

    def get_status_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._notifications)
            pending = sum(1 for n in self._notifications if n["status"] == "pending")
            accepted = sum(1 for n in self._notifications if n["status"] == "accepted")
            reminders_total = sum(int(n.get("reminders_sent", 0)) for n in self._notifications)
            direct_calls = sum(1 for n in self._notifications if n["kind"] == "direct_admin_call")

        return {
            "enabled": settings.telegram_enabled,
            "live_bot_configured": self.is_live_bot_configured,
            "delivery_mode": "telegram_bot_api" if self.is_live_bot_configured else "local_outbox_simulation",
            "reminder_minutes_threshold": settings.telegram_reminder_minutes,
            "dashboard_base_url": settings.dashboard_base_url,
            "total_notifications": total,
            "notifications_total": total,
            "pending_notifications": pending,
            "accepted_notifications": accepted,
            "direct_call_notifications": direct_calls,
            "total_reminders_sent": reminders_total,
            "admins": self.get_admins_status(),
            "config_hint": (
                "Telegram Bot API активен."
                if self.is_live_bot_configured
                else "Уведомления работают в локальном режиме (доступны в панели и API). Для отправки в реальный Telegram укажите TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в .env."
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        return self.get_status_summary()


# Singleton instance
telegram_notifier = TelegramAdminNotifier()
