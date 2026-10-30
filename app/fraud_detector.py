"""
Anti-fake report detection and Fraud Risk scoring for Grand Mobile Support Copilot.
Detects suspicious patterns: duplicates, mass complaints, contradictions between text and evidence,
and manipulation attempts.

IMPORTANT: Fraud Risk Score does NOT mean the reporter is guilty.
It is a signal for human review only. No automated punishments are issued.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("grand_mobile_copilot.fraud")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _trigrams(text: str) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i : i + 3] for i in range(max(0, len(t) - 2))}


def _similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class FraudDetector:
    """
    Detects suspicious complaint patterns, text-to-evidence contradictions,
    and computes Fraud Risk Score (0-100).

    IMPORTANT:
    - Score does NOT prove guilt.
    - No automatic punishments are ever taken based on fraud risk.
    - Every flagged complaint is routed for human admin review.
    """

    def __init__(self) -> None:
        self._complaints_by_reporter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._complaints_by_offender: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._seen_hashes: Dict[str, List[str]] = defaultdict(list)
        self._all_complaints: List[Dict[str, Any]] = []
        self._flagged_reports: Dict[str, Dict[str, Any]] = {}

    def record_complaint(
        self,
        ticket_id: str,
        reporter: Optional[str],
        offender: Optional[str],
        text: str,
        category: str,
        server: Any = None,
        fraud_result: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record a complaint for future cross-ticket fraud analysis."""
        if fraud_result is None and "fraud_score" in kwargs:
            fraud_result = {
                "fraud_risk_score": kwargs.get("fraud_score", 0),
                "reasons": kwargs.get("reasons", []),
                "contradictions": kwargs.get("contradictions", []),
            }
        # Avoid duplicate recording of the exact same ticket_id on retriage
        if any(c["ticket_id"] == ticket_id for c in self._all_complaints):
            if fraud_result and fraud_result.get("fraud_risk_score", 0) >= 25:
                self._flagged_reports[ticket_id] = {
                    "ticket_id": ticket_id,
                    "reporter": reporter,
                    "offender": offender,
                    "category": category,
                    "server": server,
                    "snippet": text[:160],
                    "created_at": _now(),
                    **fraud_result,
                }
            return

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

        if fraud_result and fraud_result.get("fraud_risk_score", 0) >= 25:
            self._flagged_reports[ticket_id] = {
                "ticket_id": ticket_id,
                "reporter": reporter,
                "offender": offender,
                "category": category,
                "server": server,
                "snippet": text[:160],
                "created_at": _now(),
                **fraud_result,
            }

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
        evidence_report: Optional[Dict[str, Any]] = None,
        prior_complaints: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a complaint for fake/fraud indicators and contradictions with evidence.
        Returns score (0-100), risk level, reasons, contradictions, and related tickets.
        """
        score = 0
        reasons: List[str] = []
        contradictions: List[str] = []
        related_tickets: List[Dict[str, Any]] = []

        # 1. Exact duplicate check (same text hash)
        text_hash = _text_hash(text)
        existing_with_hash = [tid for tid in self._seen_hashes.get(text_hash, []) if tid != ticket_id]
        if existing_with_hash:
            score += 40
            reasons.append(f"Обнаружена точная копия жалобы (ID: {', '.join(existing_with_hash[:3])})")
            related_tickets.extend(
                {"ticket_id": tid, "relation": "exact_duplicate", "similarity": 1.0}
                for tid in existing_with_hash[:3]
            )

        # 2. Near-duplicate check (high text similarity across tickets)
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
            score += 25
            reasons.append(f"Массовое обращение: обнаружено {similar_count} высокосхожих жалоб")
        elif similar_count >= 1:
            score += 12
            reasons.append(f"Повторное обращение: обнаружена схожая жалоба ({similar_count} шт.)")

        # 3. Reporter flood check (same reporter filed many complaints)
        if reporter:
            reporter_history = self._complaints_by_reporter.get(reporter.lower(), [])
            recent_from_reporter = [c for c in reporter_history if c["ticket_id"] != ticket_id]
            if len(recent_from_reporter) >= 5:
                score += 20
                reasons.append(f"Репортер {reporter!r} подал уже {len(recent_from_reporter)} жалоб (возможный флуд жалобами)")
            elif len(recent_from_reporter) >= 3:
                score += 10
                reasons.append(f"Повышенная активность заявителя {reporter!r}: {len(recent_from_reporter)} жалоб ранее")

        # 4. Coordinated mass complaints against one offender
        if offender:
            offender_complaints = self._complaints_by_offender.get(offender.lower(), [])
            different_reporters = set(
                c.get("reporter", "") for c in offender_complaints if c["ticket_id"] != ticket_id and c.get("reporter")
            )
            if len(different_reporters) >= 5:
                score += 15
                reasons.append(
                    f"Координированная волна жалоб: на {offender!r} подано {len(offender_complaints)} обращений от {len(different_reporters)} разных игроков"
                )

        # 5. Contradictions between complaint text and attached evidence (OCR / Media check)
        lower = text.lower()
        evidence_claim_keywords = [
            "прикрепил", "скриншот есть", "видео приложил", "приложил",
            "есть доказательства", "attached", "screenshot attached", "фрапс приложил",
        ]
        claims_attached = any(kw in lower for kw in evidence_claim_keywords)
        if claims_attached and attachments_count == 0 and not has_evidence:
            score += 25
            msg = "Противоречие: игрок утверждает в тексте, что приложил доказательства, но вложений 0"
            contradictions.append(msg)
            reasons.append(msg)

        # Check contradicts from EvidenceAnalyzer (OCR nicknames, IDs, servers, media type)
        if evidence_report and isinstance(evidence_report, dict):
            ev_contradictions = evidence_report.get("contradictions") or []
            for c_msg in ev_contradictions:
                if c_msg not in contradictions:
                    contradictions.append(c_msg)
                    reasons.append(f"Противоречие доказательств: {c_msg}")
                    score += 25

        # 6. Contradiction inside text itself (e.g. claims both "меня убили без причины" and "я сам первый начал стрелять")
        internal_contradictions = [
            (
                r"без\s+причин\w*.*я\s+(?:сам\s+)?(?:первый\s+)?(?:стрелял|ударил|начал)",
                "Внутреннее противоречие в тексте: заявляет об убийстве 'без причины', но признаёт, что сам начал конфликт",
            ),
            (
                r"(?:не\s+использовал\s+чит\w*|банили\s+просто\s+так).*(?:чуть-чуть\s+спидхак|только\s+телепорт|скачал\s+аим)",
                "Внутреннее противоречие в апелляции: отрицает нарушение, но упоминает использование стороннего ПО",
            ),
        ]
        for pat, msg in internal_contradictions:
            if re.search(pat, lower):
                score += 30
                contradictions.append(msg)
                reasons.append(msg)

        # 7. High-severity accusation without any evidence
        high_value_categories = {"scam_fraud", "cheating_exploits", "billing_refund"}
        if category in high_value_categories and not has_evidence:
            score += 15
            reasons.append(
                f"Серьёзное обвинение в категории '{category}' подано без каких-либо доказательств (скриншотов/видео/чеков)"
            )

        # 8. Manipulation & emotional pressure patterns
        manipulation_patterns = [
            (r"\bигнор(ирую|ируют|ируете)\b", "Попытка эмоционального давления (обвинение поддержки в игноре)"),
            (r"\bудалю\s+аккаунт\b", "Манипуляция угрозой удаления аккаунта"),
            (r"\bнапишу\s+(жалобу|репорт).*(снова|опять|еще|миллион)\b", "Угроза спамом повторными жалобами"),
            (r"\bмой\s+друг\s+гл(авный)?\s*админ\b", "Попытка давления связями с администрацией"),
        ]
        for pattern, msg in manipulation_patterns:
            if re.search(pattern, lower):
                score += 10
                reasons.append(msg)

        # 9. Very low AI classification confidence
        if confidence < 0.50:
            score += 5
            reasons.append(f"Низкая уверенность классификации ИИ ({int(confidence*100)}%) — нестандартное обращение")

        score = min(100, score)

        if score >= 70:
            risk_level = "high"
            risk_label = "Высокий риск фейка"
        elif score >= 40:
            risk_level = "medium"
            risk_label = "Средний риск (требует проверки)"
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
            "contradictions": contradictions,
            "related_tickets": related_tickets[:5],
            "similar_count": similar_count,
            "reporter_complaint_count": len(self._complaints_by_reporter.get((reporter or "").lower(), [])),
            "offender_complaint_count": len(self._complaints_by_offender.get((offender or "").lower(), [])),
            "is_suspicious": score >= 40,
            "requires_human_review": score >= 40,
            "automated_punishment_blocked": True,
            "disclaimer": "ВАЖНО: Никакие автоматические наказания на основании подозрения не применяются. Решение всегда принимает администратор.",
        }

    def list_suspicious_complaints(self, min_score: int = 25, limit: int = 50) -> List[Dict[str, Any]]:
        """Return flagged/suspicious complaints sorted by fraud risk score descending."""
        items = [
            v for v in self._flagged_reports.values()
            if int(v.get("fraud_risk_score", 0)) >= min_score
        ]
        items.sort(key=lambda x: int(x.get("fraud_risk_score", 0)), reverse=True)
        return items[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        suspicious_cnt = sum(1 for v in self._flagged_reports.values() if v.get("fraud_risk_score", 0) >= 40)
        contradiction_cnt = sum(1 for v in self._flagged_reports.values() if v.get("contradictions"))
        return {
            "total_analyzed": len(self._all_complaints),
            "suspicious_count": suspicious_cnt,
            "suspicious_flagged": suspicious_cnt,
            "contradictions_detected": contradiction_cnt,
            "unique_reporters": len(self._complaints_by_reporter),
            "unique_offenders": len(self._complaints_by_offender),
            "duplicate_hashes": sum(1 for v in self._seen_hashes.values() if len(v) > 1),
            "automated_punishments_issued": 0,
        }

    def get_top_reporters(self, limit: int = 5) -> List[Dict[str, Any]]:
        reporters = [
            {"reporter": reporter, "count": len(complaints)}
            for reporter, complaints in self._complaints_by_reporter.items()
        ]
        reporters.sort(key=lambda x: x["count"], reverse=True)
        return reporters[:limit]

    def get_top_offenders(self, limit: int = 5) -> List[Dict[str, Any]]:
        offenders = [
            {"offender": offender, "count": len(complaints)}
            for offender, complaints in self._complaints_by_offender.items()
        ]
        offenders.sort(key=lambda x: x["count"], reverse=True)
        return offenders[:limit]


# Singleton instance
fraud_detector = FraudDetector()
