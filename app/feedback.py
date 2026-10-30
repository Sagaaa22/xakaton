"""
Feedback Loop module for Grand Mobile Support Copilot.
Collects admin evaluations (correct vs error) and corrections to improve AI classification quality.
Persists feedback entries and connects to `TicketClassifier.apply_feedback` to update model weights.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional
import uuid

from app.config import settings

logger = logging.getLogger("grand_mobile_copilot.feedback")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


FEEDBACK_TYPES = [
    "correct_classification",  # AI was right
    "wrong_category",          # AI chose wrong category
    "wrong_priority",          # AI chose wrong priority
    "wrong_rule",              # AI cited wrong rule
    "wrong_evidence_analysis", # AI evidence analysis was wrong
    "wrong_confidence",        # AI confidence was overconfident / underconfident
    "intent_mismatch",         # AI detected wrong intent
]

FEEDBACK_TYPE_LABELS = {
    "correct_classification": "✅ ИИ правильно проанализировал обращение",
    "wrong_category": "❌ Ошибка категории",
    "wrong_priority": "❌ Ошибка приоритета",
    "wrong_rule": "❌ Неверно подобран пункт правил",
    "wrong_evidence_analysis": "❌ Ошибка анализа доказательств",
    "wrong_confidence": "⚠️ Неверная оценка уверенности (Confidence)",
    "intent_mismatch": "❌ Неверно определено намерение игрока",
}


class FeedbackStore:
    """
    Thread-safe storage for admin feedback on AI classifications with optional JSON persistence
    and live classifier weight update callback.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._feedback: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self.persist_path: Path = persist_path or (settings.root_dir / "data" / "feedback_store.json")
        self._model_updater: Optional[Callable[[str, str, Optional[str], Optional[str]], bool]] = None
        self._load()

    def register_model_updater(
        self,
        updater: Callable[[str, str, Optional[str], Optional[str]], bool],
    ) -> None:
        """Register classifier weight update function: (text, kind, correct_cat, correct_prio) -> bool."""
        self._model_updater = updater

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._feedback = raw
        except Exception:
            pass

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps(self._feedback, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def record(
        self,
        ticket_id: str,
        admin_id: str,
        admin_name: str,
        feedback_type: str,
        ai_category: str,
        correct_category: Optional[str],
        ai_priority: str,
        correct_priority: Optional[str],
        ai_rule_ids: List[str],
        correct_rule_ids: Optional[List[str]],
        note: Optional[str] = None,
        ticket_text: Optional[str] = None,
        apply_to_model: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Record admin feedback on an AI decision and optionally update classifier weights."""
        if feedback_type not in FEEDBACK_TYPES:
            feedback_type = "correct_classification"

        applied_to_classifier = False
        if apply_to_model and self._model_updater and ticket_text:
            target_cat = correct_category or (ai_category if feedback_type == "correct_classification" else None)
            target_prio = correct_priority or (ai_priority if feedback_type == "correct_classification" else None)
            if target_cat or target_prio:
                try:
                    applied_to_classifier = bool(
                        self._model_updater(ticket_text, "complaint", target_cat, target_prio)
                    )
                except Exception as exc:
                    logger.warning("Failed to apply feedback to classifier: %s", exc)

        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": feedback_id,
            "ticket_id": ticket_id,
            "admin_id": admin_id,
            "admin_name": admin_name,
            "feedback_type": feedback_type,
            "feedback_label": FEEDBACK_TYPE_LABELS.get(feedback_type, feedback_type),
            "ai_category": ai_category,
            "correct_category": correct_category or ai_category,
            "ai_priority": ai_priority,
            "correct_priority": correct_priority or ai_priority,
            "ai_rule_ids": ai_rule_ids,
            "correct_rule_ids": correct_rule_ids if correct_rule_ids is not None else ai_rule_ids,
            "note": note,
            "created_at": _now(),
            "reviewed": True if applied_to_classifier else False,
            "applied_to_model": applied_to_classifier,
        }

        with self._lock:
            self._feedback.append(entry)
            self._save()

        logger.info(
            "feedback_recorded id=%s ticket=%s type=%s admin=%s applied=%s",
            feedback_id,
            ticket_id,
            feedback_type,
            admin_name,
            applied_to_classifier,
        )
        return entry

    def get_for_ticket(self, ticket_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self._feedback if f.get("ticket_id") == ticket_id]

    def list_all(self, reviewed_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._feedback)
        if reviewed_only:
            items = [f for f in items if f.get("reviewed")]
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items[:limit]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._feedback)
            by_type: Dict[str, int] = {}
            correct_count = 0
            reviewed_count = 0
            applied_count = 0

            for fb in self._feedback:
                t = fb.get("feedback_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
                if t == "correct_classification":
                    correct_count += 1
                if fb.get("reviewed"):
                    reviewed_count += 1
                if fb.get("applied_to_model"):
                    applied_count += 1

        ai_accuracy = round(correct_count / total * 100, 1) if total > 0 else 100.0

        return {
            "total_feedback": total,
            "correct_classification": correct_count,
            "errors_reported": total - correct_count,
            "ai_accuracy_pct": ai_accuracy,
            "accuracy_pct": ai_accuracy,
            "by_type": by_type,
            "reviewed_count": reviewed_count,
            "unreviewed_count": total - reviewed_count,
            "applied_to_model_count": applied_count,
            "ready_for_training": reviewed_count,
            "category_corrections": self.get_category_corrections(),
            "notice": "Обратная связь администраторов сохраняется и используется для дообучения весов классификатора.",
        }

    def mark_reviewed(self, feedback_id: str) -> bool:
        with self._lock:
            for fb in self._feedback:
                if fb["id"] == feedback_id:
                    fb["reviewed"] = True
                    fb["reviewed_at"] = _now()
                    self._save()
                    return True
        return False

    def get_category_corrections(self) -> Dict[str, Dict[str, int]]:
        corrections: Dict[str, Dict[str, int]] = {}
        with self._lock:
            for fb in self._feedback:
                if fb["feedback_type"] == "wrong_category" and fb.get("correct_category"):
                    ai_cat = fb["ai_category"]
                    correct_cat = fb["correct_category"]
                    if ai_cat not in corrections:
                        corrections[ai_cat] = {}
                    corrections[ai_cat][correct_cat] = corrections[ai_cat].get(correct_cat, 0) + 1
        return corrections


# Singleton instance
feedback_store = FeedbackStore()
