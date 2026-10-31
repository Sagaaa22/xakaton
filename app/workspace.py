"""
Operator workspace for Grand Mobile Support Copilot.

- Demo users and roles (`junior_admin`, `senior_admin`).
- Ticket lifecycle (statuses, assignment, drafts, escalation, senior review).
- Server-side permission checks for EVERY mutating request (single source of truth: `check_action`).
- Per-ticket history and global audit log (who, what, when, on what basis).

DEMO MODE WARNING:
The acting user is taken from the `X-Demo-User` request header. The server resolves the role
from its own user table (the client cannot claim a role), but there is NO authentication:
anyone can send any demo user id. This is a demonstration of role-based access control,
NOT secure authorization. Replace `get_current_user` with a real identity provider before production.
"""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.safety import SafetyGuard
from app.schemas import TriageRequest

JUNIOR = "junior_admin"
SENIOR = "senior_admin"

DEMO_USERS: Dict[str, Dict[str, str]] = {
    "junior_1": {"id": "junior_1", "name": "Айдар К.", "role": JUNIOR},
    "junior_2": {"id": "junior_2", "name": "Мария С.", "role": JUNIOR},
    "senior_1": {"id": "senior_1", "name": "Ерлан Б.", "role": SENIOR},
}
ROLE_LABELS = {JUNIOR: "Младший администратор", SENIOR: "Старший администратор"}

STATUSES: Dict[str, str] = {
    "new": "Новое",
    "in_progress": "В работе",
    "awaiting_player": "Ожидает данных игрока",
    "escalated": "Эскалировано старшему",
    "returned": "Возвращено на доработку",
    "resolved": "Решено",
    "closed": "Закрыто",
}
FINAL_STATUSES = {"resolved", "closed"}
WORKING_STATUSES = {"in_progress", "returned", "awaiting_player"}
PRIORITY_RANK = {
    "p0": 4, "p1": 3, "p2": 2, "p3": 1,
    "critical": 4, "high": 3, "medium": 2, "low": 1,
}

ACTIONS = [
    "take", "assign", "retriage", "save_draft", "request_info",
    "send_reply", "escalate", "approve", "return", "close",
]
ACTION_LABELS = {
    "ticket_created": "Поступил новый репорт",
    "ticket_imported": "Импортирован из датасета",
    "view_ticket": "Открыл репорт",
    "take": "Взять в работу",
    "assign": "Переназначить",
    "retriage": "Перегенерировать анализ",
    "save_draft": "Сохранить черновик",
    "request_info": "Запросить данные у игрока",
    "send_reply": "Отправить ответ",
    "escalate": "Передать старшему специалисту",
    "approve": "Одобрить решение",
    "return": "Вернуть на доработку",
    "close": "Закрыть",
    "assistant_saved": "Сохранено через AI Assistant",
}

DECISIONS: Dict[str, str] = {
    "no_violation": "Нарушение не подтверждено",
    "punishment_confirmed": "Нарушение подтверждено человеком (наказание по регламенту)",
    "forwarded": "Передано в профильный отдел (финансы / техподдержка / ГА)",
    "answered": "Дан ответ игроку",
    "info_requested": "Запрошены дополнительные данные",
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "junior_can_view_unassigned": True,
    "junior_can_send_without_review": True,
}

# Ground-truth label/outcome fields from datasets must never leak into triage input/facts
LABEL_KEYS = {"category", "priority", "label", "is_synthetic_demo", "verdict", "punishment", "violation"}
STATE_VERSION = 5
URGENT_PRIORITIES = {"p0", "p1", "critical", "high"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _player_text(ticket: Dict[str, Any]) -> str:
    return "\n".join(
        m.get("text", "")
        for m in ticket.get("messages", [])
        if str(m.get("role", "player")).lower() in ("player", "user", "client", "reporter")
    )


def _trigrams(text: str) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


def _forbidden(message: str) -> HTTPException:
    return HTTPException(status_code=403, detail={"error": "forbidden", "message": message})


def _not_found(message: str = "Тикет не найден") -> HTTPException:
    return HTTPException(status_code=404, detail={"error": "not_found", "message": message})


def _bad_request(message: str, extra: Optional[Dict[str, Any]] = None) -> HTTPException:
    detail = {"error": "invalid_action", "message": message}
    if extra:
        detail.update(extra)
    return HTTPException(status_code=422, detail=detail)


class WorkspaceStore:
    """Thread-safe in-memory ticket store with optional JSON persistence."""

    def __init__(self, triage_service, seed_path: Optional[Path], persist_path: Optional[Path]) -> None:
        self.svc = triage_service
        self.seed_path = seed_path
        self.persist_path = persist_path
        self.lock = threading.RLock()
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self.audit: List[Dict[str, Any]] = []
        self.settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self._load_or_seed()

    # ---------- persistence ----------
    def _save(self) -> None:
        if not self.persist_path:
            return
        payload = {
            "version": STATE_VERSION,
            "tickets": self.tickets,
            "audit": self.audit,
            "settings": self.settings,
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.persist_path)

    def _load_or_seed(self) -> None:
        if self.persist_path and self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                raw_tickets = data.get("tickets")
                if isinstance(raw_tickets, dict) and raw_tickets:
                    self.tickets = raw_tickets
                    self.audit = data.get("audit", [])
                    self.settings = {**DEFAULT_SETTINGS, **data.get("settings", {})}
                    if data.get("version") != STATE_VERSION or self._needs_migration():
                        self._migrate_state(data.get("version", 1))
                        self._save()
                    return
            except Exception:
                pass
        self.reset()

    def _needs_migration(self) -> bool:
        for t in self.tickets.values():
            if "read_by" not in t or "extra_meta" not in t:
                return True
            esc = t.get("escalation")
            if isinstance(esc, dict) and "seen_by" not in esc:
                return True
        return False

    def _migrate_state(self, from_version: int) -> None:
        """Non-destructive migration preserving all existing tickets, audit entries, and settings."""
        for tid, t in self.tickets.items():
            if "read_by" not in t or not isinstance(t.get("read_by"), list):
                read_by: List[str] = []
                if t.get("assignee") and t.get("status") not in ("new", "returned"):
                    read_by.append(t["assignee"])
                t["read_by"] = read_by
            if "extra_meta" not in t or not isinstance(t.get("extra_meta"), dict):
                t["extra_meta"] = {}
            esc = t.get("escalation")
            if isinstance(esc, dict) and "seen_by" not in esc:
                esc["seen_by"] = []
            rev = t.get("review")
            if isinstance(rev, dict) and "seen_by" not in rev:
                rev["seen_by"] = []
            tr = t.get("triage") or {}
            # Migrate legacy priority codes if an old state v1-v3 was loaded
            legacy_prio = {"critical": "p0", "high": "p1", "medium": "p2", "low": "p3"}
            if tr.get("priority") in legacy_prio:
                tr["priority"] = legacy_prio[tr["priority"]]

    def reset(self) -> None:
        with self.lock:
            self.tickets = {}
            self.audit = []
            self.settings = dict(DEFAULT_SETTINGS)
            self._seed()
            self._save()

    # ---------- triage ----------
    def run_triage(
        self,
        kind: str,
        messages: List[Dict[str, Any]],
        ticket_id: Optional[str],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = extra_meta or {}
        req = TriageRequest(
            id=ticket_id,
            kind=kind or "complaint",
            messages=messages,
            images_count=meta.get("images_count"),
            reported_nick=meta.get("reported_nick"),
            rule_refs_raw=meta.get("rule_refs_raw"),
            region=meta.get("region"),
            regenerate=True,
        )
        return self.svc.analyze(req, record_cache=False).dict()

    def rule_verification(self, triage: Dict[str, Any]) -> List[Dict[str, Any]]:
        rag = self.svc.rag
        out = [{"id": rid, "exists": rag.is_valid_rule_id(rid), "source": "retrieved"} for rid in triage.get("rule_ids", [])]
        for rid in triage.get("facts", {}).get("unknown_mentioned_rules", []):
            out.append({"id": rid, "exists": rag.is_valid_rule_id(rid), "source": "mentioned_by_player"})
        return out

    def find_similar(self, text: str, exclude_id: Optional[str] = None, threshold: float = 0.55) -> List[Dict[str, Any]]:
        q = _trigrams(text)
        if not q:
            return []
        found = []
        for t in self.tickets.values():
            if t["id"] == exclude_id:
                continue
            other = _trigrams(_player_text(t))
            if not other:
                continue
            sim = len(q & other) / len(q | other)
            if sim >= threshold:
                found.append({"id": t["id"], "similarity": round(sim, 2), "status": t["status"],
                              "status_label": STATUSES.get(t["status"], t["status"])})
        found.sort(key=lambda x: x["similarity"], reverse=True)
        return found[:3]

    # ---------- seeding from provided (or demo) dataset ----------
    def _seed(self) -> None:
        if not self.seed_path or not Path(self.seed_path).exists():
            return
        lines = [ln for ln in Path(self.seed_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_raw = [json.loads(ln) for ln in lines[:1000]]
        complaints = [r for r in parsed_raw if r.get("kind") != "report"]
        reports = [r for r in parsed_raw if r.get("kind") == "report"]
        selected = (complaints[:25] + reports[:15]) if reports else parsed_raw[:40]
        base = datetime.now(timezone.utc) - timedelta(minutes=17 * len(selected))
        for i, raw in enumerate(selected[:40]):
            clean = {k: v for k, v in raw.items() if k not in LABEL_KEYS}
            tid = str(clean.get("id") or clean.get("ticket_id") or f"tkt_{i + 1:04d}")
            raw_msgs = clean.get("messages") or []
            messages = [
                m for m in raw_msgs
                if isinstance(m, dict) and str(m.get("role", "player")).lower() not in ("admin", "moderator", "support")
            ] or raw_msgs
            imgs = int(clean.get("images_count") or 0)
            if imgs > 0 and messages and not messages[0].get("attachments"):
                messages = [dict(messages[0], attachments=imgs)] + messages[1:]
            kind = clean.get("kind") or "complaint"
            extra_meta = {
                "images_count": imgs if "images_count" in clean else None,
                "reported_nick": clean.get("reported_nick"),
                "rule_refs_raw": clean.get("rule_refs_raw"),
                "region": clean.get("region"),
            }
            triage = self.run_triage(kind, messages, tid, extra_meta=extra_meta)
            created = clean.get("created_at") or (base + timedelta(minutes=17 * i)).isoformat(timespec="seconds")
            ticket = {
                "id": tid, "kind": kind, "created_at": created, "updated_at": created,
                "messages": messages, "triage": triage, "status": "new", "assignee": None,
                "created_by": None, "draft": triage["draft_response"], "escalation": None,
                "junior_response": None, "review": None,
                "extra_meta": extra_meta,
                "read_by": [],
                "is_synthetic_demo": bool(raw.get("is_synthetic_demo")),
            }
            self.tickets[tid] = ticket
            self._log(None, "ticket_imported", tid, {"source": Path(self.seed_path).name})

            # Deterministic demo distribution across roles and statuses
            slot = i % 6
            if slot == 1:
                self._seed_assign(ticket, "junior_1")
            elif slot == 2:
                self._seed_assign(ticket, "junior_2")
            elif slot == 4:
                self._seed_assign(ticket, "junior_1")
                reason = triage.get("needs_human_reason") or "Спорный случай, требуется решение старшего"
                ticket["status"] = "escalated"
                ticket["junior_response"] = ticket["draft"]
                ticket["escalation"] = {
                    "reason": reason,
                    "by": "junior_1",
                    "by_name": DEMO_USERS["junior_1"]["name"],
                    "at": created,
                    "seen_by": [],
                }
                self._log(DEMO_USERS["junior_1"], "escalate", tid, {"reason": reason})
            elif slot == 5:
                self._seed_assign(ticket, "junior_2")
                ticket["status"] = "awaiting_player"

    def _seed_assign(self, ticket: Dict[str, Any], user_id: str) -> None:
        ticket["assignee"] = user_id
        ticket["status"] = "in_progress"
        if user_id not in ticket.setdefault("read_by", []):
            ticket["read_by"].append(user_id)
        self._log(DEMO_USERS[user_id], "take", ticket["id"], {})

    # ---------- audit ----------
    def _log(self, actor: Optional[Dict[str, str]], action: str, ticket_id: Optional[str], details: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "id": uuid.uuid4().hex[:10],
            "ts": _now(),
            "ticket_id": ticket_id,
            "actor_id": actor["id"] if actor else "system",
            "actor_name": actor["name"] if actor else "Система",
            "actor_role": actor["role"] if actor else "system",
            "action": action,
            "details": details,
        }
        self.audit.append(entry)
        return entry

    # ---------- permissions (single source of truth) ----------
    def can_view(self, user: Dict[str, str], t: Dict[str, Any]) -> bool:
        if user["role"] == SENIOR:
            return True
        if t.get("assignee") == user["id"] or t.get("created_by") == user["id"]:
            return True
        return (
            t.get("assignee") is None
            and t["status"] in ("new", "returned")
            and bool(self.settings.get("junior_can_view_unassigned", True))
        )

    def check_action(self, user: Dict[str, str], t: Dict[str, Any], action: str) -> Optional[str]:
        """Return None if allowed, otherwise a human-readable denial reason."""
        if not self.can_view(user, t):
            return "Нет доступа к этому обращению"
        senior = user["role"] == SENIOR
        mine = t.get("assignee") == user["id"]
        st = t["status"]
        needs_human = bool(t.get("triage", {}).get("needs_human"))

        if action == "take":
            if t.get("assignee") is not None:
                return "Обращение уже назначено"
            if st not in ("new", "returned"):
                return "Взять можно только новое или возвращённое обращение"
            return None
        if action == "assign":
            if not senior:
                return "Переназначение доступно только старшему администратору"
            if st in FINAL_STATUSES:
                return "Обращение уже завершено"
            return None
        if action == "retriage":
            if not (mine or senior):
                return "Перегенерация доступна исполнителю или старшему"
            if st in FINAL_STATUSES:
                return "Обращение уже завершено"
            return None
        if action == "save_draft":
            if not (mine or senior):
                return "Редактировать черновик может только исполнитель или старший"
            if st in FINAL_STATUSES:
                return "Обращение уже завершено"
            if st == "escalated" and not senior:
                return "Обращение на рассмотрении у старшего — редактирование заблокировано"
            return None
        if action == "request_info":
            if not (mine or senior):
                return "Запрос данных доступен исполнителю или старшему"
            if st not in WORKING_STATUSES and not (senior and st == "escalated"):
                return "Сначала возьмите обращение в работу"
            return None
        if action == "send_reply":
            if st not in WORKING_STATUSES and not (senior and st == "escalated"):
                return "Сначала возьмите обращение в работу"
            if senior:
                return None
            if not mine:
                return "Отправлять ответ может только исполнитель"
            if needs_human:
                return "Обращение требует проверки человеком с полномочиями: запросите данные или передайте старшему"
            if not self.settings.get("junior_can_send_without_review", True):
                return "Отправка без проверки старшим отключена в настройках"
            return None
        if action == "escalate":
            if senior:
                return "Старший администратор принимает решение сам — используйте «Одобрить решение» или «Закрыть»"
            if not mine:
                return "Эскалировать может только исполнитель обращения"
            if st not in WORKING_STATUSES:
                return "Эскалировать можно только обращение в работе"
            return None
        if action in ("approve", "return", "close"):
            if not senior:
                return "Действие доступно только старшему администратору"
            if action == "approve" and st not in WORKING_STATUSES | {"escalated"}:
                return "Одобрить можно только обращение в работе или на эскалации"
            if action == "return":
                if st != "escalated":
                    return "Вернуть на доработку можно только эскалированное обращение"
                assignee = DEMO_USERS.get(t.get("assignee") or "")
                if not assignee or assignee["role"] != JUNIOR:
                    return "Нет младшего исполнителя, которому можно вернуть обращение"
            if action == "close" and st == "closed":
                return "Обращение уже закрыто"
            return None
        return "Неизвестное действие"

    def require(self, user: Dict[str, str], t: Dict[str, Any], action: str) -> None:
        reason = self.check_action(user, t, action)
        if reason:
            raise _forbidden(reason)

    # ---------- views ----------
    def summary(self, t: Dict[str, Any], user: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        tr = t.get("triage", {})
        assignee = DEMO_USERS.get(t.get("assignee") or "")
        text = _player_text(t)
        prio = tr.get("priority")
        read_by = t.get("read_by") or []
        unread = (user["id"] not in read_by) if user else False
        esc = t.get("escalation") if isinstance(t.get("escalation"), dict) else None
        escalation_unseen = bool(
            t["status"] == "escalated" and esc and user and user["id"] not in (esc.get("seen_by") or [])
        )
        return {
            "id": t["id"],
            "created_at": t["created_at"],
            "updated_at": t["updated_at"],
            "kind": t["kind"],
            "category": tr.get("category"),
            "category_name": tr.get("category_name"),
            "priority": prio,
            "is_urgent": prio in URGENT_PRIORITIES,
            "language": tr.get("language"),
            "status": t["status"],
            "status_label": STATUSES.get(t["status"], t["status"]),
            "assignee": t.get("assignee"),
            "assignee_name": assignee["name"] if assignee else None,
            "needs_human": bool(tr.get("needs_human")),
            "escalated": t["status"] == "escalated" or t.get("escalation") is not None,
            "escalation_unseen": escalation_unseen,
            "unread": unread,
            "confidence": tr.get("confidence"),
            "is_spam": tr.get("is_spam"),
            "is_duplicate": tr.get("is_duplicate"),
            "snippet": text[:160],
        }

    def detail(self, user: Dict[str, str], t: Dict[str, Any]) -> Dict[str, Any]:
        allowed, denied = [], {}
        for a in ACTIONS:
            reason = self.check_action(user, t, a)
            if reason is None:
                allowed.append(a)
            else:
                denied[a] = reason
        out = dict(t)
        out.update(self.summary(t, user))
        out["allowed_actions"] = allowed
        out["denied_actions"] = denied
        out["history"] = [e for e in self.audit if e["ticket_id"] == t["id"]]
        out["rule_verification"] = self.rule_verification(t.get("triage", {}))
        out["similar_tickets"] = self.find_similar(_player_text(t), exclude_id=t["id"])
        return out

    def get(self, ticket_id: str) -> Dict[str, Any]:
        t = self.tickets.get(ticket_id)
        if not t:
            raise _not_found()
        return t

    def touch(self, t: Dict[str, Any]) -> None:
        t["updated_at"] = _now()


# ====================== Request bodies ======================
class AssignBody(BaseModel):
    assignee_id: str


class TextBody(BaseModel):
    text: str = Field(..., min_length=1)


class EscalateBody(BaseModel):
    reason: str = Field(..., min_length=5, description="Причина эскалации (обязательно)")


class ReviewBody(BaseModel):
    action: str = Field(..., description="approve | return | close")
    decision: Optional[str] = Field(None, description="Ключ из DECISIONS (для approve)")
    basis: Optional[str] = Field(None, description="Основание решения (для approve/close)")
    comment: Optional[str] = Field(None, description="Комментарий исполнителю (для return)")
    evidence_verified: bool = Field(False, description="Человек лично проверил доказательства/логи")
    evidence_reference: Optional[str] = Field(None, description="Что именно проверено: логи, скриншоты, видео")
    rule_ids: List[str] = Field(default_factory=list, description="Пункты rules.json, на которых основано решение")
    reply_text: Optional[str] = Field(None, description="Финальный ответ игроку (необязательно)")


class AssistantAnalyzeBody(BaseModel):
    ticket_id: Optional[str] = None
    text: Optional[str] = None
    kind: str = "complaint"
    attachments: int = Field(0, ge=0, le=50)


class AssistantSaveBody(BaseModel):
    ticket_id: Optional[str] = None
    text: Optional[str] = None
    kind: str = "complaint"
    attachments: int = Field(0, ge=0, le=50)
    draft: str = Field(..., min_length=1)


class SettingsBody(BaseModel):
    junior_can_view_unassigned: Optional[bool] = None
    junior_can_send_without_review: Optional[bool] = None


class CreateTicketBody(BaseModel):
    text: str = Field(..., min_length=2, description="Текст обращения игрока")
    kind: str = Field("complaint", description="complaint | report")
    attachments: int = Field(0, ge=0, le=50)
    reported_nick: Optional[str] = None
    region: Optional[str] = None


class ReadStateBody(BaseModel):
    unread: bool = False


# ====================== Router ======================
def create_router(store: WorkspaceStore) -> APIRouter:
    router = APIRouter(prefix="/api/ws", tags=["workspace"])

    async def get_current_user(x_demo_user: Optional[str] = Header(None, alias="X-Demo-User")) -> Dict[str, str]:
        """DEMO MODE: identity from header, role resolved server-side. Not secure authentication."""
        if not x_demo_user:
            raise HTTPException(status_code=401, detail={"error": "unauthenticated", "message": "Не указан пользователь (заголовок X-Demo-User)"})
        user = DEMO_USERS.get(x_demo_user.strip())
        if not user:
            raise HTTPException(status_code=401, detail={"error": "unauthenticated", "message": "Неизвестный демо-пользователь"})
        return user

    def require_senior(user: Dict[str, str]) -> None:
        if user["role"] != SENIOR:
            raise _forbidden("Доступно только старшему администратору")

    def safety_check(text: str, t: Optional[Dict[str, Any]]) -> None:
        unknown = (t or {}).get("triage", {}).get("facts", {}).get("unknown_mentioned_rules", [])
        ok, violations = SafetyGuard.validate_draft(text, store.svc.rag.valid_rule_ids, unknown)
        if not ok:
            raise _bad_request(
                "Текст не прошёл проверку безопасности: обещания разбана/возврата, гарантии наказания, "
                "утверждение вины или несуществующие пункты правил запрещены",
                {"violations": violations},
            )

    # ---------- identity & reference data ----------
    @router.get("/me", summary="Текущий демо-пользователь и его права")
    async def me(user=Depends(get_current_user)):
        return {
            **user,
            "role_label": ROLE_LABELS[user["role"]],
            "demo_mode": True,
            "auth_notice": "DEMO MODE: роль выбирается без аутентификации и не является безопасной авторизацией",
            "can_manage_settings": user["role"] == SENIOR,
            "can_review": user["role"] == SENIOR,
            "can_edit_rules": False,
        }

    @router.get("/users", summary="Список демо-пользователей")
    async def users(user=Depends(get_current_user)):
        return [{**u, "role_label": ROLE_LABELS[u["role"]]} for u in DEMO_USERS.values()]

    @router.get("/meta", summary="Справочники: категории, приоритеты, статусы, решения")
    async def meta(user=Depends(get_current_user)):
        tax = store.svc.taxonomy
        return {
            "categories": [{"id": c.id, "name": c.name} for c in tax.categories.values()],
            "priorities": tax.priorities,
            "statuses": STATUSES,
            "decisions": DECISIONS,
            "actions": ACTION_LABELS,
            "languages": ["ru", "kk", "uz", "en", "und"],
        }

    @router.get("/rules", summary="База правил (только чтение, из rules.json)")
    async def rules(user=Depends(get_current_user)):
        rag = store.svc.rag
        return {
            "read_only": True,
            "source": rag.rules_path.name,
            "count": len(rag.rules_by_id),
            "rules": [
                {"id": r.id, "section": r.section, "category": r.category, "title": r.title,
                 "text": r.text, "punishment": r.punishment}
                for r in rag.rules_by_id.values()
            ],
        }

    # ---------- tickets ----------
    @router.get("/tickets", summary="Список тикетов с фильтрами и сортировкой")
    async def list_tickets(
        scope: str = Query("all", description="all | mine | queue | escalations"),
        status: Optional[str] = None,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        language: Optional[str] = None,
        assignee: Optional[str] = Query(None, description="id пользователя или 'none'"),
        escalated: Optional[bool] = None,
        unread: Optional[bool] = Query(None, description="Фильтр непрочитанных репортов"),
        urgent: Optional[bool] = Query(None, description="Фильтр срочных обращений (p0/p1)"),
        q: Optional[str] = None,
        sort: str = Query("created_at", description="created_at | priority | category | status | language | id | assignee"),
        order: str = Query("desc", description="asc | desc"),
        user=Depends(get_current_user),
    ):
        if scope == "escalations":
            require_senior(user)
        with store.lock:
            items = [t for t in store.tickets.values() if store.can_view(user, t)]
            if scope == "mine":
                items = [t for t in items if t.get("assignee") == user["id"]]
            elif scope == "queue":
                items = [t for t in items if t.get("assignee") is None and t["status"] in ("new", "returned")]
            elif scope == "escalations":
                items = [t for t in items if t["status"] == "escalated"]
            rows = [store.summary(t, user) for t in items]

        def f(rows, key, val):
            return [r for r in rows if r.get(key) == val] if val else rows

        rows = f(rows, "status", status)
        rows = f(rows, "category", category)
        rows = f(rows, "priority", priority)
        rows = f(rows, "language", language)
        if assignee == "none":
            rows = [r for r in rows if not r["assignee"]]
        elif assignee:
            rows = [r for r in rows if r["assignee"] == assignee]
        if escalated is not None:
            rows = [r for r in rows if r["escalated"] == escalated]
        if unread is not None:
            rows = [r for r in rows if bool(r.get("unread")) == unread]
        if urgent is not None:
            rows = [r for r in rows if bool(r.get("is_urgent")) == urgent]
        if q:
            ql = q.lower().strip()
            rows = [r for r in rows if ql in r["id"].lower() or ql in (r["snippet"] or "").lower()
                    or ql in (r["category_name"] or "").lower() or ql in (r["category"] or "").lower()]

        keyfn = {
            "priority": lambda r: (PRIORITY_RANK.get(r["priority"], 0), r["created_at"]),
            "category": lambda r: r["category"] or "",
            "status": lambda r: r["status"],
            "language": lambda r: r["language"] or "",
            "id": lambda r: r["id"],
            "assignee": lambda r: r["assignee_name"] or "",
        }.get(sort, lambda r: r["created_at"])
        rows.sort(key=keyfn, reverse=(order != "asc"))
        return {"count": len(rows), "tickets": rows}

    @router.post("/tickets", summary="Поступление нового репорта (с автоматическим ИИ-триажем)")
    async def create_ticket(body: CreateTicketBody, user=Depends(get_current_user)):
        text = (body.text or "").strip()
        if len(text) < 2:
            raise _bad_request("Введите текст обращения игрока (минимум 2 символа)")
        with store.lock:
            tid = f"tkt_{uuid.uuid4().hex[:6]}"
            messages = [{"role": "player", "text": text, "attachments": body.attachments}]
            extra_meta = {
                "images_count": body.attachments,
                "reported_nick": (body.reported_nick or "").strip() or None,
                "region": (body.region or "").strip() or None,
            }
            triage = store.run_triage(body.kind, messages, tid, extra_meta=extra_meta)
            now = _now()
            t = {
                "id": tid,
                "kind": body.kind or "complaint",
                "created_at": now,
                "updated_at": now,
                "messages": messages,
                "triage": triage,
                "status": "new",
                "assignee": None,
                "created_by": None,
                "draft": triage["draft_response"],
                "escalation": None,
                "junior_response": None,
                "review": None,
                "extra_meta": extra_meta,
                "read_by": [],
                "is_synthetic_demo": False,
            }
            store.tickets[tid] = t
            store._log(None, "ticket_created", tid, {
                "category": triage.get("category"),
                "priority": triage.get("priority"),
                "language": triage.get("language"),
                "rule_ids": triage.get("rule_ids", []),
                "needs_human": triage.get("needs_human"),
            })
            store._save()
            return store.detail(user, t)

    @router.get("/tickets/{ticket_id}", summary="Карточка тикета с доступными действиями")
    async def get_ticket(ticket_id: str, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            if not store.can_view(user, t):
                raise _forbidden("Нет доступа к этому обращению")
            changed = False
            read_by = t.setdefault("read_by", [])
            if user["id"] not in read_by:
                read_by.append(user["id"])
                changed = True
            if user["role"] == SENIOR and isinstance(t.get("escalation"), dict):
                seen_by = t["escalation"].setdefault("seen_by", [])
                if user["id"] not in seen_by:
                    seen_by.append(user["id"])
                    changed = True
            already_viewed = any(
                e["ticket_id"] == ticket_id and e["actor_id"] == user["id"] and e["action"] == "view_ticket"
                for e in store.audit
            )
            if not already_viewed:
                store._log(user, "view_ticket", ticket_id, {"role": user["role"]})
                changed = True
            if changed:
                store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/read", summary="Отметить репорт прочитанным или непрочитанным")
    async def mark_read(ticket_id: str, body: ReadStateBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            if not store.can_view(user, t):
                raise _forbidden("Нет доступа к этому обращению")
            read_by = t.setdefault("read_by", [])
            if body.unread:
                if user["id"] in read_by:
                    read_by.remove(user["id"])
            else:
                if user["id"] not in read_by:
                    read_by.append(user["id"])
            store._save()
            return store.detail(user, t)

    @router.get("/tickets/{ticket_id}/history", summary="История изменений тикета")
    async def ticket_history(ticket_id: str, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            if not store.can_view(user, t):
                raise _forbidden("Нет доступа к этому обращению")
            return [e for e in store.audit if e["ticket_id"] == ticket_id]

    @router.post("/tickets/{ticket_id}/take", summary="Взять тикет в работу (назначить себя)")
    async def take(ticket_id: str, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "take")
            t["assignee"] = user["id"]
            t["status"] = "in_progress"
            read_by = t.setdefault("read_by", [])
            if user["id"] not in read_by:
                read_by.append(user["id"])
            store.touch(t)
            store._log(user, "take", ticket_id, {})
            store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/assign", summary="Назначить исполнителя (только старший)")
    async def assign(ticket_id: str, body: AssignBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "assign")
            target = DEMO_USERS.get(body.assignee_id)
            if not target:
                raise _bad_request("Неизвестный исполнитель")
            prev = t.get("assignee")
            t["assignee"] = target["id"]
            if t["status"] == "new":
                t["status"] = "in_progress"
            read_by = t.setdefault("read_by", [])
            if target["id"] in read_by and target["id"] != user["id"]:
                read_by.remove(target["id"])
            store.touch(t)
            store._log(user, "assign", ticket_id, {"from": prev, "to": target["id"], "to_name": target["name"]})
            store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/triage", summary="Повторно запустить триаж ИИ")
    async def retriage(ticket_id: str, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "retriage")
            t["triage"] = store.run_triage(t["kind"], t["messages"], ticket_id, extra_meta=t.get("extra_meta"))
            t["draft"] = t["triage"]["draft_response"]
            store.touch(t)
            store._log(user, "retriage", ticket_id, {
                "category": t["triage"]["category"], "priority": t["triage"]["priority"],
                "rule_ids": t["triage"]["rule_ids"], "needs_human": t["triage"]["needs_human"],
            })
            store._save()
            return store.detail(user, t)

    @router.put("/tickets/{ticket_id}/draft", summary="Сохранить отредактированный черновик")
    async def save_draft(ticket_id: str, body: TextBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "save_draft")
            unknown = t["triage"].get("facts", {}).get("unknown_mentioned_rules", [])
            ok, violations = SafetyGuard.validate_draft(body.text, store.svc.rag.valid_rule_ids, unknown)
            t["draft"] = body.text
            store.touch(t)
            store._log(user, "save_draft", ticket_id, {"length": len(body.text), "safety_ok": ok, "violations": violations})
            store._save()
            out = store.detail(user, t)
            out["safety_warnings"] = violations
            return out

    @router.post("/tickets/{ticket_id}/request-info", summary="Запросить дополнительные данные у игрока")
    async def request_info(ticket_id: str, body: TextBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "request_info")
            safety_check(body.text, t)
            t["messages"].append({"role": "admin", "text": body.text, "attachments": 0, "author": user["id"], "at": _now(), "type": "info_request"})
            t["status"] = "awaiting_player"
            store.touch(t)
            store._log(user, "request_info", ticket_id, {"length": len(body.text)})
            store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/send", summary="Отправить ответ игроку (в рамках полномочий)")
    async def send_reply(ticket_id: str, body: TextBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "send_reply")
            safety_check(body.text, t)
            t["messages"].append({"role": "admin", "text": body.text, "attachments": 0, "author": user["id"], "at": _now(), "type": "reply"})
            t["draft"] = body.text
            t["status"] = "resolved"
            store.touch(t)
            store._log(user, "send_reply", ticket_id, {"length": len(body.text), "ai_needs_human": t["triage"].get("needs_human")})
            store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/escalate", summary="Передать старшему специалисту")
    async def escalate(ticket_id: str, body: EscalateBody, user=Depends(get_current_user)):
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, "escalate")
            t["status"] = "escalated"
            t["junior_response"] = t.get("draft")
            t["escalation"] = {
                "reason": body.reason.strip(),
                "by": user["id"],
                "by_name": user["name"],
                "at": _now(),
                "seen_by": [],
            }
            # Mark as unread for senior admins so it surfaces in unread & escalation notifications
            read_by = t.setdefault("read_by", [])
            for uid, u in DEMO_USERS.items():
                if u["role"] == SENIOR and uid in read_by:
                    read_by.remove(uid)
            store.touch(t)
            store._log(user, "escalate", ticket_id, {"reason": body.reason.strip()})
            store._save()
            return store.detail(user, t)

    @router.post("/tickets/{ticket_id}/review", summary="Решение старшего: approve | return | close")
    async def review(ticket_id: str, body: ReviewBody, user=Depends(get_current_user)):
        if body.action not in ("approve", "return", "close"):
            raise _bad_request("action должен быть approve, return или close")
        with store.lock:
            t = store.get(ticket_id)
            store.require(user, t, body.action)
            tr = t.get("triage", {})
            ai_snapshot = {
                "category": tr.get("category"), "priority": tr.get("priority"),
                "needs_human": tr.get("needs_human"), "confidence": tr.get("confidence"),
                "rule_ids": tr.get("rule_ids", []),
            }
            if isinstance(t.get("escalation"), dict):
                seen_by = t["escalation"].setdefault("seen_by", [])
                if user["id"] not in seen_by:
                    seen_by.append(user["id"])

            if body.action == "return":
                if not body.comment or len(body.comment.strip()) < 5:
                    raise _bad_request("Укажите комментарий для исполнителя (минимум 5 символов)")
                t["status"] = "returned"
                t["review"] = {
                    "action": "return",
                    "comment": body.comment.strip(),
                    "by": user["id"],
                    "by_name": user["name"],
                    "at": _now(),
                    "seen_by": [],
                }
                # Mark unread for the junior assignee when returned
                read_by = t.setdefault("read_by", [])
                if t.get("assignee") in read_by:
                    read_by.remove(t["assignee"])
                store._log(user, "return", ticket_id, {"comment": body.comment.strip(), "to": t.get("assignee")})

            elif body.action == "close":
                if not body.basis or len(body.basis.strip()) < 5:
                    raise _bad_request("Укажите основание закрытия (минимум 5 символов)")
                t["status"] = "closed"
                t["review"] = {"action": "close", "basis": body.basis.strip(), "by": user["id"], "by_name": user["name"], "at": _now(), "ai_snapshot": ai_snapshot}
                store._log(user, "close", ticket_id, {"basis": body.basis.strip()})

            else:  # approve
                if body.decision not in DECISIONS:
                    raise _bad_request("Выберите итоговое решение из списка", {"allowed": list(DECISIONS)})
                basis = (body.basis or "").strip()
                if len(basis) < 10:
                    raise _bad_request("Зафиксируйте основание решения (минимум 10 символов)")
                rag = store.svc.rag
                invalid = [r for r in body.rule_ids if not rag.is_valid_rule_id(r)]
                if invalid:
                    raise _bad_request("Указаны несуществующие пункты правил", {"invalid_rule_ids": invalid})
                if body.decision == "punishment_confirmed":
                    ref = (body.evidence_reference or "").strip()
                    if not body.evidence_verified or len(ref) < 5:
                        raise _bad_request(
                            "Нельзя подтвердить наказание только на основании анализа ИИ: отметьте, что доказательства "
                            "проверены лично, и укажите, что именно проверено (логи, скриншоты, видео)"
                        )
                    if not body.rule_ids:
                        raise _bad_request("Для подтверждения нарушения укажите хотя бы один пункт rules.json")
                if body.reply_text:
                    safety_check(body.reply_text, t)
                    t["messages"].append({"role": "admin", "text": body.reply_text, "attachments": 0, "author": user["id"], "at": _now(), "type": "reply"})
                    t["draft"] = body.reply_text
                t["status"] = "resolved"
                t["review"] = {
                    "action": "approve", "decision": body.decision, "decision_label": DECISIONS[body.decision],
                    "basis": basis, "evidence_verified": body.evidence_verified,
                    "evidence_reference": (body.evidence_reference or "").strip() or None,
                    "rule_ids": body.rule_ids, "by": user["id"], "by_name": user["name"], "at": _now(),
                    "ai_snapshot": ai_snapshot,
                    "note": "Анализ ИИ носил рекомендательный характер; решение принято человеком",
                }
                store._log(user, "approve", ticket_id, {
                    "decision": body.decision, "basis": basis, "rule_ids": body.rule_ids,
                    "evidence_verified": body.evidence_verified, "evidence_reference": body.evidence_reference,
                })
            store.touch(t)
            store._save()
            return store.detail(user, t)

    # ---------- AI Report Assistant ----------
    def _assistant_input(body, user) -> tuple:
        if body.ticket_id:
            t = store.get(body.ticket_id)
            if not store.can_view(user, t):
                raise _forbidden("Нет доступа к этому обращению")
            return t["kind"], t["messages"], t["id"], t
        text = (body.text or "").strip()
        if not text:
            raise _bad_request("Вставьте текст репорта или выберите тикет")
        return body.kind, [{"role": "player", "text": text, "attachments": body.attachments}], None, None

    @router.post("/assistant/analyze", summary="AI Report Assistant: анализ без сохранения")
    async def assistant_analyze(body: AssistantAnalyzeBody, user=Depends(get_current_user)):
        with store.lock:
            kind, messages, tid, t = _assistant_input(body, user)
            triage = store.run_triage(kind, messages, tid, extra_meta=(t or {}).get("extra_meta"))
            text = "\n".join(m.get("text", "") for m in messages if m.get("role", "player") == "player")
            similar = store.find_similar(text, exclude_id=tid)
        return {
            "ticket_id": tid,
            "kind": kind,
            "messages": messages,
            "triage": triage,
            "rule_verification": store.rule_verification(triage),
            "rules_found": bool(triage.get("rule_ids")),
            "similar_tickets": similar,
        }

    @router.post("/assistant/save", summary="AI Report Assistant: сохранить результат как тикет")
    async def assistant_save(body: AssistantSaveBody, user=Depends(get_current_user)):
        with store.lock:
            if body.ticket_id:
                t = store.get(body.ticket_id)
                store.require(user, t, "save_draft")
                t["triage"] = store.run_triage(t["kind"], t["messages"], t["id"], extra_meta=t.get("extra_meta"))
                t["draft"] = body.draft
                store.touch(t)
                store._log(user, "assistant_saved", t["id"], {"mode": "update_existing", "rule_ids": t["triage"]["rule_ids"]})
            else:
                text = (body.text or "").strip()
                if not text:
                    raise _bad_request("Вставьте текст репорта")
                tid = f"rep_{uuid.uuid4().hex[:6]}"
                messages = [{"role": "player", "text": text, "attachments": body.attachments}]
                triage = store.run_triage(body.kind, messages, tid)
                now = _now()
                t = {
                    "id": tid, "kind": body.kind, "created_at": now, "updated_at": now,
                    "messages": messages, "triage": triage, "status": "in_progress",
                    "assignee": user["id"], "created_by": user["id"], "draft": body.draft,
                    "escalation": None, "junior_response": None, "review": None,
                    "extra_meta": {"images_count": body.attachments},
                    "read_by": [user["id"]],
                    "is_synthetic_demo": False,
                }
                store.tickets[tid] = t
                store._log(user, "assistant_saved", tid, {"mode": "new_ticket", "rule_ids": triage["rule_ids"]})
            store._save()
            return store.detail(user, t)

    # ---------- analytics, audit, settings ----------
    @router.get("/analytics", summary="Аналитика (командная — только старшему)")
    async def analytics(user=Depends(get_current_user)):
        with store.lock:
            visible = [t for t in store.tickets.values() if store.can_view(user, t)]
            by_status: Dict[str, int] = {}
            by_category: Dict[str, int] = {}
            by_priority: Dict[str, int] = {}
            for t in visible:
                by_status[t["status"]] = by_status.get(t["status"], 0) + 1
                c = t["triage"].get("category")
                by_category[c] = by_category.get(c, 0) + 1
                p = t["triage"].get("priority")
                by_priority[p] = by_priority.get(p, 0) + 1
            open_items = [t for t in visible if t["status"] not in FINAL_STATUSES]
            unread_count = sum(1 for t in visible if user["id"] not in (t.get("read_by") or []))
            urgent_open_count = sum(1 for t in open_items if t["triage"].get("priority") in URGENT_PRIORITIES)

            notifications: List[Dict[str, Any]] = []
            new_escalations_count = 0
            for t in visible:
                esc = t.get("escalation") if isinstance(t.get("escalation"), dict) else None
                if t["status"] == "escalated" and esc:
                    unseen = user["id"] not in (esc.get("seen_by") or [])
                    if unseen:
                        new_escalations_count += 1
                    if user["role"] == SENIOR:
                        notifications.append({
                            "type": "escalation",
                            "ticket_id": t["id"],
                            "priority": t["triage"].get("priority"),
                            "category_name": t["triage"].get("category_name"),
                            "reason": esc.get("reason"),
                            "by_name": esc.get("by_name"),
                            "at": esc.get("at") or t["updated_at"],
                            "unseen": unseen,
                        })
                elif user["role"] == JUNIOR and t["status"] == "returned" and t.get("assignee") == user["id"]:
                    rev = t.get("review") or {}
                    unseen = user["id"] not in (t.get("read_by") or [])
                    notifications.append({
                        "type": "returned",
                        "ticket_id": t["id"],
                        "priority": t["triage"].get("priority"),
                        "category_name": t["triage"].get("category_name"),
                        "reason": rev.get("comment") or "Возвращено старшим на доработку",
                        "by_name": rev.get("by_name") or "Старший администратор",
                        "at": rev.get("at") or t["updated_at"],
                        "unseen": unseen,
                    })
            notifications.sort(key=lambda x: (not x["unseen"], x.get("at") or ""), reverse=False)

            result = {
                "visible_total": len(visible),
                "my_open": sum(1 for t in open_items if t.get("assignee") == user["id"]),
                "queue_unassigned": sum(1 for t in visible if t.get("assignee") is None and t["status"] in ("new", "returned")),
                "escalations_open": sum(1 for t in visible if t["status"] == "escalated"),
                "new_escalations_count": new_escalations_count,
                "unread_count": unread_count,
                "urgent_open_count": urgent_open_count,
                "critical_open": sum(1 for t in open_items if t["triage"].get("priority") in ("p0", "critical")),
                "needs_human_open": sum(1 for t in open_items if t["triage"].get("needs_human")),
                "notifications": notifications[:15],
                "by_status": by_status,
                "by_category": by_category,
                "by_priority": by_priority,
                "processing": {k: v for k, v in store.svc.get_analytics_summary().items()
                               if k in ("total_processed", "avg_processing_time_ms", "p95_processing_time_ms", "avg_cost_usd", "escalation_rate_pct")},
                "team": None,
            }
            if user["role"] == SENIOR:
                team = []
                for u in DEMO_USERS.values():
                    team.append({
                        "id": u["id"], "name": u["name"], "role": u["role"],
                        "open": sum(1 for t in store.tickets.values() if t.get("assignee") == u["id"] and t["status"] not in FINAL_STATUSES),
                        "resolved": sum(1 for t in store.tickets.values() if t.get("assignee") == u["id"] and t["status"] in FINAL_STATUSES),
                        "escalated_by": sum(1 for e in store.audit if e["actor_id"] == u["id"] and e["action"] == "escalate"),
                        "actions": sum(1 for e in store.audit if e["actor_id"] == u["id"]),
                    })
                result["team"] = team
            return result

    @router.get("/audit", summary="Журнал действий (младший видит только свои)")
    async def audit(
        limit: int = Query(200, ge=1, le=1000),
        action: Optional[str] = None,
        actor: Optional[str] = None,
        user=Depends(get_current_user),
    ):
        with store.lock:
            entries = list(store.audit)
        if user["role"] != SENIOR:
            entries = [e for e in entries if e["actor_id"] == user["id"]]
        elif actor:
            entries = [e for e in entries if e["actor_id"] == actor]
        if action:
            entries = [e for e in entries if e["action"] == action]
        entries.reverse()
        return {"scope": "team" if user["role"] == SENIOR else "own", "count": len(entries), "entries": entries[:limit]}

    @router.get("/settings", summary="Системные настройки (только старший)")
    async def get_settings(user=Depends(get_current_user)):
        require_senior(user)
        llm = store.svc.generator.llm_client
        return {
            "settings": store.settings,
            "system": {
                "demo_mode": True,
                "llm_enabled": bool(llm and llm.enabled),
                "llm_model": llm.model if llm and llm.enabled else None,
                "rules_source": store.svc.rag.rules_path.name,
                "rules_count": len(store.svc.rag.rules_by_id),
                "punishment_requires_human_evidence": True,
            },
        }

    @router.put("/settings", summary="Изменить системные настройки (только старший)")
    async def put_settings(body: SettingsBody, user=Depends(get_current_user)):
        require_senior(user)
        with store.lock:
            changes = {k: v for k, v in body.dict().items() if v is not None}
            store.settings.update(changes)
            store._log(user, "settings_changed", None, changes)
            store._save()
            return {"settings": store.settings}

    @router.post("/demo/reset", summary="Сбросить демо-данные (только старший)")
    async def demo_reset(user=Depends(get_current_user)):
        require_senior(user)
        store.reset()
        with store.lock:
            store._log(user, "demo_reset", None, {})
            store._save()
        return {"status": "reset", "tickets": len(store.tickets)}

    return router
