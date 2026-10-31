"""
Admin Request (direct call) tracking module for Grand Mobile Support Copilot.
Handles player requests for a live human admin with persistent storage.
Tracks statuses: WAITING -> ACCEPTED -> IN_PROGRESS -> COMPLETED | CANCELLED.

IMPORTANT: Players always have the right to request a live admin,
regardless of AI confidence or classification.
"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import uuid

from app.config import settings

logger = logging.getLogger("grand_mobile_copilot.admin_requests")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Status constants
STATUS_WAITING = "WAITING"
STATUS_ACCEPTED = "ACCEPTED"
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_CANCELLED = "CANCELLED"

STATUS_LABELS = {
    STATUS_WAITING: "⏳ Ожидает администратора",
    STATUS_ACCEPTED: "✅ Принят администратором",
    STATUS_IN_PROGRESS: "🔄 Администратор подключился",
    STATUS_COMPLETED: "✔️ Завершён",
    STATUS_CANCELLED: "❌ Отменён игроком",
}


class AdminRequestStore:
    """Thread-safe store for direct admin call requests with disk persistence."""

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.persist_path: Path = persist_path or (settings.root_dir / "data" / "admin_requests.json")
        self._load()

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._requests = raw
            elif isinstance(raw, list):
                self._requests = {r["id"]: r for r in raw if isinstance(r, dict) and "id" in r}
        except Exception as exc:
            logger.warning("Could not load admin requests: %s", exc)

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._requests, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.persist_path)
        except Exception as exc:
            logger.warning("Could not persist admin requests: %s", exc)

    def reset(self) -> None:
        with self._lock:
            self._requests = {}
            self._save()

    def create(
        self,
        player_nickname: Optional[str],
        server: Any,
        context_messages: List[Dict[str, Any]],
        ticket_id: Optional[str] = None,
        language: str = "ru",
    ) -> Dict[str, Any]:
        """Create a new admin request from a player."""
        request_id = f"ar_{uuid.uuid4().hex[:8]}"

        # Build context summary from messages
        context_texts = [
            m.get("text", "")
            for m in context_messages
            if m.get("role", "player") in ("player", "user", "client", "reporter")
            and m.get("text")
        ]
        context_summary = "\n".join(context_texts[-3:])[:500]  # Last 3 messages, max 500 chars

        request = {
            "id": request_id,
            "ticket_id": ticket_id,
            "player_nickname": player_nickname,
            "server": server,
            "language": language,
            "context_summary": context_summary,
            "context_messages": context_messages[-5:],  # Keep last 5 messages
            "status": STATUS_WAITING,
            "status_label": STATUS_LABELS[STATUS_WAITING],
            "created_at": _now(),
            "updated_at": _now(),
            "accepted_by": None,
            "accepted_at": None,
            "completed_at": None,
            "cancelled_at": None,
            "admin_note": None,
            "history": [
                {
                    "status": STATUS_WAITING,
                    "timestamp": _now(),
                    "actor": "player",
                    "note": "Игрок запросил живого администратора",
                }
            ],
        }

        with self._lock:
            self._requests[request_id] = request
            self._save()

        logger.info("admin_request_created id=%s player=%s server=%s", request_id, player_nickname, server)
        return dict(request)

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get admin request by ID."""
        with self._lock:
            return dict(self._requests[request_id]) if request_id in self._requests else None

    def accept(self, request_id: str, admin_id: str, admin_name: str) -> Optional[Dict[str, Any]]:
        """Admin accepts a request."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return None
            if req["status"] != STATUS_WAITING:
                return dict(req)  # Already accepted or finished

            req["status"] = STATUS_ACCEPTED
            req["status_label"] = STATUS_LABELS[STATUS_ACCEPTED]
            req["accepted_by"] = admin_id
            req["accepted_at"] = _now()
            req["updated_at"] = _now()
            req["history"].append({
                "status": STATUS_ACCEPTED,
                "timestamp": _now(),
                "actor": admin_id,
                "note": f"Принят администратором {admin_name}",
            })
            self._save()
            logger.info("admin_request_accepted id=%s admin=%s", request_id, admin_name)
            return dict(req)

    def start(self, request_id: str, admin_id: str, admin_name: str) -> Optional[Dict[str, Any]]:
        """Admin starts working on request (IN_PROGRESS state)."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return None

            req["status"] = STATUS_IN_PROGRESS
            req["status_label"] = STATUS_LABELS[STATUS_IN_PROGRESS]
            req["updated_at"] = _now()
            req["history"].append({
                "status": STATUS_IN_PROGRESS,
                "timestamp": _now(),
                "actor": admin_id,
                "note": f"Администратор {admin_name} подключился к обращению",
            })
            self._save()
            return dict(req)

    def complete(self, request_id: str, admin_id: str, admin_name: str, note: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Complete an admin request."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return None

            req["status"] = STATUS_COMPLETED
            req["status_label"] = STATUS_LABELS[STATUS_COMPLETED]
            req["completed_at"] = _now()
            req["updated_at"] = _now()
            if note:
                req["admin_note"] = note
            req["history"].append({
                "status": STATUS_COMPLETED,
                "timestamp": _now(),
                "actor": admin_id,
                "note": note or f"Обращение завершено администратором {admin_name}",
            })
            self._save()
            logger.info("admin_request_completed id=%s admin=%s", request_id, admin_name)
            return dict(req)

    def cancel(self, request_id: str, requester: str = "player") -> Optional[Dict[str, Any]]:
        """Cancel an admin request (by player or admin)."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return None
            if req["status"] in (STATUS_COMPLETED, STATUS_CANCELLED):
                return dict(req)

            req["status"] = STATUS_CANCELLED
            req["status_label"] = STATUS_LABELS[STATUS_CANCELLED]
            req["cancelled_at"] = _now()
            req["updated_at"] = _now()
            req["history"].append({
                "status": STATUS_CANCELLED,
                "timestamp": _now(),
                "actor": requester,
                "note": "Отменён" + (" игроком" if requester == "player" else f" администратором {requester}"),
            })
            self._save()
            return dict(req)

    def list_requests(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List all admin requests, optionally filtered by status."""
        with self._lock:
            items = list(self._requests.values())

        if status:
            items = [r for r in items if r["status"] == status]

        items.sort(key=lambda x: x["created_at"], reverse=True)
        return [dict(r) for r in items[:limit]]

    def get_waiting_count(self) -> int:
        """Return count of requests waiting for admin."""
        with self._lock:
            return sum(1 for r in self._requests.values() if r["status"] == STATUS_WAITING)

    def get_stats(self) -> Dict[str, Any]:
        """Return admin request statistics."""
        with self._lock:
            total = len(self._requests)
            by_status = {}
            for req in self._requests.values():
                s = req["status"]
                by_status[s] = by_status.get(s, 0) + 1

        return {
            "total": total,
            "waiting": by_status.get(STATUS_WAITING, 0),
            "accepted": by_status.get(STATUS_ACCEPTED, 0),
            "in_progress": by_status.get(STATUS_IN_PROGRESS, 0),
            "completed": by_status.get(STATUS_COMPLETED, 0),
            "cancelled": by_status.get(STATUS_CANCELLED, 0),
        }


# Singleton instance for the full application
admin_request_store = AdminRequestStore()
