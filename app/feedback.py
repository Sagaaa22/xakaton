"""
Feedback Loop module for Grand Mobile Support Copilot.
Collects admin corrections to improve AI classification quality.
Does NOT trigger automatic retraining on unchecked data.
"""
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    "correct_classification": "✅ ИИ правильно классифицировал обращение",
    "wrong_category": "❌ Неверная категория",
    "wrong_priority": "❌ Неверный приоритет",
    "wrong_rule": "❌ Неверно найден пункт правил",
    "wrong_evidence_analysis": "❌ Ошибка анализа доказательств",
    "wrong_confidence": "⚠️ Неверная оценка уверенности",
    "intent_mismatch": "❌ Неверное намерение игрока",
}


class FeedbackStore:
    """
    Thread-safe storage for admin feedback on AI classifications.
    Feedback is stored for quality review before any use in training.
    """

    def __init__(self) -> None:
        self._feedback: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

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
    ) -> Dict[str, Any]:
        """Record admin feedback on an AI decision."""
        if feedback_type not in FEEDBACK_TYPES:
            feedback_type = "correct_classification"
        
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": feedback_id,
            "ticket_id": ticket_id,
            "admin_id": admin_id,
            "admin_name": admin_name,
            "feedback_type": feedback_type,
            "feedback_label": FEEDBACK_TYPE_LABELS.get(feedback_type, feedback_type),
            "ai_category": ai_category,
            "correct_category": correct_category,
            "ai_priority": ai_priority,
            "correct_priority": correct_priority,
            "ai_rule_ids": ai_rule_ids,
            "correct_rule_ids": correct_rule_ids,
            "note": note,
            "created_at": _now(),
            "reviewed": False,  # Must be reviewed before using for training
        }
        
        with self._lock:
            self._feedback.append(entry)
        
        logger.info(
            "feedback_recorded id=%s ticket=%s type=%s admin=%s",
            feedback_id, ticket_id, feedback_type, admin_name
        )
        return entry

    def list_all(self, reviewed_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """List all feedback records."""
        with self._lock:
            items = list(self._feedback)
        
        if reviewed_only:
            items = [f for f in items if f.get("reviewed")]
        
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about collected feedback."""
        with self._lock:
            total = len(self._feedback)
            by_type: Dict[str, int] = {}
            correct_count = 0
            reviewed_count = 0
            
            for fb in self._feedback:
                t = fb.get("feedback_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
                if t == "correct_classification":
                    correct_count += 1
                if fb.get("reviewed"):
                    reviewed_count += 1
        
        ai_accuracy = round(correct_count / total * 100, 1) if total > 0 else 0.0
        
        return {
            "total_feedback": total,
            "correct_classification": correct_count,
            "ai_accuracy_pct": ai_accuracy,
            "by_type": by_type,
            "reviewed_count": reviewed_count,
            "unreviewed_count": total - reviewed_count,
            "ready_for_training": reviewed_count,
            "notice": "Данные обратной связи проходят проверку качества перед использованием для обучения модели",
        }

    def mark_reviewed(self, feedback_id: str) -> bool:
        """Mark a feedback entry as reviewed and ready for training."""
        with self._lock:
            for fb in self._feedback:
                if fb["id"] == feedback_id:
                    fb["reviewed"] = True
                    fb["reviewed_at"] = _now()
                    return True
        return False

    def get_category_corrections(self) -> Dict[str, Dict[str, int]]:
        """Get category correction patterns to understand AI weaknesses."""
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
