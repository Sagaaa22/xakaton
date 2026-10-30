"""
Anti-fake report detection and Fraud Risk scoring for Grand Mobile Support Copilot.
Detects suspicious patterns: duplicates, mass complaints, contradictions, manipulations.

IMPORTANT: Fraud Risk Score does NOT mean the reporter is guilty.
It is a signal for human review only. No automated punishments are issued.
"""
import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.fraud")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _trigrams(text: str) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


def _similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class FraudDetector:
    """
    Detects suspicious complaint patterns and computes Fraud Risk Score.
    
    IMPORTANT:
    - Score does NOT prove guilt.  
    - No automatic actions are taken based solely on fraud risk.
    - Every flagged complaint must be reviewed by a human admin.
    """

    def __init__(self) -> None:
        # In-memory stores: player_id/nick -> list of complaint records
        self._complaints_by_reporter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._complaints_by_offender: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._seen_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> [ticket_ids]
        self._all_complaints: List[Dict[str, Any]] = []

    def record_complaint(
        self,
        ticket_id: str,
        reporter: Optional[str],
        offender: Optional[str],
        text: str,
        category: str,
        server: Any = None,
    ) -> None:
        """Record a complaint for future fraud analysis."""
        record = {
            "ticket_id": ticket_id,
            "reporter": reporter,
            "offender": offender,
            "text": text,
            "category": category,
            "server": server,
            "created_at": _now(),
            "hash": _text_hash(text),
        }
        self._all_complaints.append(record)
        
        if reporter:
            self._complaints_by_reporter[reporter.lower()].append(record)
            
        if offender:
            self._complaints_by_offender[offender.lower()].append(record)
        
        h = _text_hash(text)
        self._seen_hashes[h].append(ticket_id)

    def analyze(
        self,
        ticket_id: str,
        reporter: Optional[str],
        offender: Optional[str],
        text: str,
        category: str,
        confidence: float,
        has_evidence: bool,
        attachments_count: int,
        server: Any = None,
        prior_complaints: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a complaint for fraud risk.
        Returns a dict with score (0-100), reasons, duplicates, related tickets.
        Higher score = more suspicious = warrants human review.
        """
        score = 0
        reasons: List[str] = []
        related_tickets: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # 1. Exact duplicate check (same text hash)
        text_hash = _text_hash(text)
        existing_with_hash = [tid for tid in self._seen_hashes.get(text_hash, []) if tid != ticket_id]
        if existing_with_hash:
            score += 40
            reasons.append(f"Обнаружена точная копия жалобы (ID: {', '.join(existing_with_hash[:3])})")
            related_tickets.extend({"ticket_id": tid, "relation": "exact_duplicate", "similarity": 1.0} for tid in existing_with_hash[:3])

        # 2. Near-duplicate check (high similarity)
        similar_count = 0
        for prev in self._all_complaints:
            if prev["ticket_id"] == ticket_id:
                continue
            sim = _similarity(text, prev["text"])
            if sim >= 0.80:
                similar_count += 1
                if sim < 1.0 and not any(r["ticket_id"] == prev["ticket_id"] for r in related_tickets):
                    related_tickets.append({
                        "ticket_id": prev["ticket_id"],
                        "relation": "near_duplicate",
                        "similarity": round(sim, 2),
                    })
        
        if similar_count >= 3:
            score += 20
            reasons.append(f"Обнаружено {similar_count} высокосхожих жалоб (возможная массовая рассылка)")
        elif similar_count >= 1:
            score += 10
            reasons.append(f"Обнаружена {similar_count} схожая жалоба")

        # 3. Reporter spam check (same reporter filed many complaints recently)
        if reporter:
            reporter_history = self._complaints_by_reporter.get(reporter.lower(), [])
            recent_from_reporter = [c for c in reporter_history if c["ticket_id"] != ticket_id]
            if len(recent_from_reporter) >= 5:
                score += 20
                reasons.append(f"Репортер {reporter!r} подал {len(recent_from_reporter)} жалоб ранее (возможный спам)")
            elif len(recent_from_reporter) >= 3:
                score += 10
                reasons.append(f"Репортер {reporter!r} уже подавал {len(recent_from_reporter)} жалоб")

        # 4. Mass complaints against one offender
        if offender:
            offender_complaints = self._complaints_by_offender.get(offender.lower(), [])
            different_reporters = set(c.get("reporter", "") for c in offender_complaints if c["ticket_id"] != ticket_id)
            if len(different_reporters) >= 5:
                score += 15
                reasons.append(f"На игрока {offender!r} уже подано {len(offender_complaints)} жалоб от {len(different_reporters)} разных репортеров")
            elif len(different_reporters) >= 3:
                score += 5
                reasons.append(f"На игрока {offender!r} подано {len(offender_complaints)} предыдущих жалоб")

        # 5. High-value claim without evidence
        high_value_categories = {"scam_fraud", "cheating_exploits", "billing_refund", "ban_appeal"}
        if category in high_value_categories and not has_evidence:
            score += 15
            reasons.append(f"Категория '{category}' требует доказательств, но доказательства не приложены")

        # 6. Contradiction: claims evidence but attachments count is 0
        evidence_keywords = ["прикрепил", "скриншот", "видео", "приложил", "есть доказательства", "attached", "screenshot", "video"]
        claims_evidence = any(kw in text.lower() for kw in evidence_keywords)
        if claims_evidence and attachments_count == 0:
            score += 10
            reasons.append("Игрок упоминает доказательства в тексте, но файлы не прикреплены")

        # 7. Suspicious patterns in text
        manipulation_patterns = [
            (r"\bигнор(ирую|ируют|ируете)\b", "Возможная попытка давления через угрозу игнора"),
            (r"\bудалю\s+аккаунт\b", "Угроза удаления аккаунта как инструмент давления"),
            (r"\bнапишу\s+(жалобу|репорт).*(снова|опять|еще)\b", "Угроза повторной жалобы"),
        ]
        for pattern, msg in manipulation_patterns:
            if re.search(pattern, text.lower()):
                score += 8
                reasons.append(msg)

        # 8. Very low AI confidence (uncertain classification)
        if confidence < 0.50:
            score += 5
            reasons.append(f"Очень низкая уверенность классификации ИИ ({int(confidence*100)}%) — возможно нетипичное обращение")

        # Cap at 100
        score = min(100, score)

        # Risk level label
        if score >= 70:
            risk_level = "high"
            risk_label = "Высокий риск"
        elif score >= 40:
            risk_level = "medium"
            risk_label = "Средний риск"
        elif score >= 20:
            risk_level = "low"
            risk_label = "Низкий риск"
        else:
            risk_level = "minimal"
            risk_label = "Минимальный риск"

        return {
            "fraud_risk_score": score,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "reasons": reasons,
            "related_tickets": related_tickets[:5],
            "similar_count": similar_count,
            "reporter_complaint_count": len(self._complaints_by_reporter.get((reporter or "").lower(), [])),
            "offender_complaint_count": len(self._complaints_by_offender.get((offender or "").lower(), [])),
            "is_suspicious": score >= 40,
            "requires_human_review": score >= 40,
            "disclaimer": "ВАЖНО: Fraud Risk Score — сигнал для ревью, не доказательство вины. Решение принимает человек.",
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated fraud detection statistics."""
        return {
            "total_analyzed": len(self._all_complaints),
            "unique_reporters": len(self._complaints_by_reporter),
            "unique_offenders": len(self._complaints_by_offender),
            "duplicate_hashes": sum(1 for v in self._seen_hashes.values() if len(v) > 1),
        }

    def get_top_reporters(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return reporters with most complaints (for admin awareness)."""
        reporters = [
            {"reporter": reporter, "count": len(complaints)}
            for reporter, complaints in self._complaints_by_reporter.items()
        ]
        reporters.sort(key=lambda x: x["count"], reverse=True)
        return reporters[:limit]

    def get_top_offenders(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return most-complained-about players (for admin awareness)."""
        offenders = [
            {"offender": offender, "count": len(complaints)}
            for offender, complaints in self._complaints_by_offender.items()
        ]
        offenders.sort(key=lambda x: x["count"], reverse=True)
        return offenders[:limit]


# Singleton instance
fraud_detector = FraudDetector()
