"""
End-to-end Triage Service orchestrating extraction, classification, RAG over rules,
intent recognition, evidence analysis (OCR & video frames), fraud detection,
direct admin call handling, Telegram notifications, player history lookup (ID & nickname),
AI admin copilot summaries, safety verification, feedback loop, and privacy-preserving analytics.
"""
import logging
import math
import time
from typing import Any, Dict, List, Optional
import uuid

from app.classifier import TaxonomyManager, TicketClassifier
from app.config import settings
from app.extractor import TicketFactExtractor
from app.generator import DraftGenerator, LLMClient
from app.rag import RulesRAGEngine
from app.safety import SafetyGuard
from app.schemas import CitedRule, TriageRequest, TriageResponse
from app.intent_recognizer import detect_intent, INTENT_LABELS, INTENT_REQUIRES_ADMIN
from app.fraud_detector import fraud_detector
from app.admin_requests import admin_request_store
from app.player_history import player_history_store
from app.admin_copilot import AdminCopilot
from app.evidence_analyzer import evidence_analyzer
from app.telegram_notifier import telegram_notifier
from app.feedback import feedback_store

logger = logging.getLogger("grand_mobile_copilot")


class TriageService:
    """Core service for analyzing player support tickets."""

    def __init__(
        self,
        rules_path=None,
        taxonomy_path=None,
        train_path=None,
        weights_path=None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.rules_path = rules_path or settings.rules_path
        self.taxonomy_path = taxonomy_path or settings.taxonomy_path
        self.train_path = train_path or settings.train_data_path
        self.weights_path = weights_path or settings.model_weights_path

        self.rag = RulesRAGEngine(self.rules_path)
        self.taxonomy = TaxonomyManager(self.taxonomy_path)
        self.extractor = TicketFactExtractor()
        self.classifier = TicketClassifier(
            taxonomy=self.taxonomy,
            rag=self.rag,
            train_path=self.train_path,
            weights_path=self.weights_path,
        )
        self.generator = DraftGenerator(llm_client=llm_client)

        # Register Feedback Loop -> Classifier weight updater
        feedback_store.register_model_updater(self.classifier.apply_feedback)

        self._recorded_ticket_ids: set = set()

        # Privacy-safe runtime analytics (no raw player PII stored in telemetry counters)
        self.analytics: Dict[str, Any] = {
            "total_processed": 0,
            "escalated_count": 0,
            "auto_handled_count": 0,
            "admin_calls_count": 0,
            "suspicious_count": 0,
            "senior_escalated_ids": set(),
            "category_counts": {cid: 0 for cid in self.taxonomy.valid_categories},
            "subcategory_counts": {},
            "priority_counts": {pid: 0 for pid in self.taxonomy.valid_priorities},
            "language_counts": {"ru": 0, "kk": 0, "uz": 0, "en": 0, "und": 0},
            "latencies_ms": [],
            "total_cost_usd": 0.0,
        }

    def analyze(
        self,
        request: TriageRequest,
        record_cache: bool = True,
    ) -> TriageResponse:
        start_ts = time.perf_counter()
        if not hasattr(self, "_copilot"):
            self._copilot = AdminCopilot(
                rules_by_id=self.rag.rules_by_id,
                taxonomy_categories=self.taxonomy.categories,
            )
        reasons: List[str] = []

        ticket_id = str(request.id or request.ticket_id or f"tkt_{uuid.uuid4().hex[:8]}")
        player_msgs = [
            m for m in request.messages if m.role in ("player", "user", "client", "reporter")
        ]
        if not player_msgs:
            player_msgs = request.messages

        combined_player_text = "\n".join(m.text.strip() for m in player_msgs if m.text and m.text.strip())

        # 1. Spam & Duplicate Check
        if request.regenerate:
            is_spam, is_duplicate, spam_reasons = self.extractor.detect_spam_and_duplicate(
                player_msgs, record_in_cache=False
            )
            is_duplicate = False
        else:
            is_spam, is_duplicate, spam_reasons = self.extractor.detect_spam_and_duplicate(
                player_msgs, record_in_cache=record_cache
            )
        reasons.extend(spam_reasons)

        # 2. Language & Structured Facts Extraction
        language, facts = self.extractor.extract_facts(
            request=request,
            valid_rule_ids=self.rag.valid_rule_ids,
        )
        if facts.is_mixed_language:
            reasons.append(
                f"Обнаружено смешение языков ({', '.join(facts.detected_languages)}); основной язык ответа: '{language}'"
            )
        else:
            reasons.append(f"Определен язык обращения: '{language}'")

        if facts.slang_or_typos_normalized:
            reasons.append(
                f"Распознан игровой сленг/опечатки: {', '.join(facts.slang_or_typos_normalized[:4])}"
            )

        if facts.unknown_mentioned_rules:
            reasons.append(
                f"Игрок сослался на несуществующий в rules.json пункт правил: {', '.join(facts.unknown_mentioned_rules)}"
            )

        # 3. Category, Subcategory & Priority Classification
        category, priority, confidence, cls_reasons = self.classifier.classify(
            text=combined_player_text,
            kind=request.kind or "complaint",
            facts=facts,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
        )
        reasons.extend(cls_reasons)

        subcategory, subcategory_name = self.classifier.detect_subcategory(
            text=combined_player_text,
            category=category,
        )
        confidence_pct = int(round(confidence * 100))
        reasons.append(
            f"Подкатегория классификации: «{subcategory_name}» ({subcategory}), Confidence Score: {confidence_pct}%"
        )

        # 4. RAG Retrieval over rules.json
        cited_rules, _, unknown_rules, has_reliable_rule = self.rag.retrieve(
            query_text=combined_player_text,
            category=category,
            mentioned_rule_ids=facts.mentioned_rules,
            top_k=3,
        )

        if (
            not facts.has_evidence
            and category in ("insult_toxicity", "rp_violation", "scam_fraud", "cheating_exploits")
            and self.rag.is_valid_rule_id("7.2")
        ):
            already_has_72 = any(r.id == "7.2" for r in cited_rules)
            if not already_has_72:
                r72 = self.rag.rules_by_id["7.2"]
                cited_rules.append(
                    CitedRule(
                        id="7.2",
                        section=r72.section,
                        title=r72.title,
                        quote=r72.text,
                        punishment=r72.punishment,
                        relevance_score=0.65,
                    )
                )

        if (
            (is_spam or is_duplicate or category == "spam_offtopic")
            and self.rag.is_valid_rule_id("7.3")
            and not any(r.id == "7.3" for r in cited_rules)
        ):
            r73 = self.rag.rules_by_id["7.3"]
            cited_rules.insert(
                0,
                CitedRule(
                    id="7.3",
                    section=r73.section,
                    title=r73.title,
                    quote=r73.text,
                    punishment=r73.punishment,
                    relevance_score=0.85,
                ),
            )

        cited_rules = [r for r in cited_rules if self.rag.is_valid_rule_id(r.id)]
        rule_ids = self.rag.verify_rule_ids([r.id for r in cited_rules])

        if rule_ids:
            reasons.append(f"Найдены подтверждающие пункты rules.json: {', '.join(rule_ids)}")
        else:
            reasons.append("Надежного подтверждения в rules.json не найдено")

        # 5. Evidence Analysis (JPG, PNG, MP4, WEBM, OCR & cross-check with complaint text)
        ev_files: List[Dict[str, Any]] = []
        if request.evidence_files:
            for f in request.evidence_files:
                ev_files.append(f.dict() if hasattr(f, "dict") else dict(f))
        for m in request.messages:
            if isinstance(m.attachments, list):
                for item in m.attachments:
                    if isinstance(item, dict):
                        ev_files.append(dict(item))
                    elif isinstance(item, str) and item.strip():
                        ev_files.append({"filename": item.strip()})

        # If attachments count > 0 but no explicit file objects passed, synthesize descriptors from text/evidence_types
        total_atts = sum(m.attachments_count for m in request.messages)
        if not ev_files and total_atts > 0:
            has_vid_type = any(t in facts.evidence_types for t in ("video_file", "video_mention"))
            default_ext = "mp4" if has_vid_type else "png"
            for idx in range(min(total_atts, 3)):
                ev_files.append({"filename": f"attachment_{idx + 1}.{default_ext}"})

        evidence_report = None
        if ev_files or facts.has_evidence or category in ("insult_toxicity", "rp_violation", "scam_fraud", "cheating_exploits"):
            evidence_report = evidence_analyzer.analyze_and_compare(
                media_items=ev_files,
                complaint_text=combined_player_text,
                complaint_facts=facts.dict(),
                category=category,
            )
            if evidence_report.get("contradictions"):
                reasons.append(
                    f"🔍 Проверка доказательств выявила расхождения: {'; '.join(evidence_report['contradictions'][:2])}"
                )

        # 6. Determine `needs_human` and `needs_human_reason`
        needs_human, needs_human_reason = self._evaluate_human_escalation(
            category=category,
            priority=priority,
            language=language,
            facts=facts,
            cited_rules=cited_rules,
            has_reliable_rule=has_reliable_rule,
            confidence=confidence,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
            player_text=combined_player_text,
        )
        if evidence_report and not evidence_report.get("is_consistent", True):
            needs_human = True
            contradiction_str = "; ".join(evidence_report.get("contradictions", [])[:2])
            needs_human_reason = (
                f"{needs_human_reason}; Расхождение текста с доказательствами: {contradiction_str}"
                if needs_human_reason
                else f"Расхождение текста с доказательствами: {contradiction_str}"
            )

        if needs_human and needs_human_reason:
            reasons.append(f"Требуется оператор: {needs_human_reason}")
        elif not needs_human:
            reasons.append(
                f"Автоматическая обработка ИИ разрешена: высокая уверенность ({confidence_pct}%) и безопасная категория"
            )

        # 7. Intent Recognition
        intent, intent_conf, intent_signals = detect_intent(
            text=combined_player_text,
            category_hint=category,
        )
        intent_label = INTENT_LABELS.get(intent, intent)
        requires_admin = bool(INTENT_REQUIRES_ADMIN.get(intent, False) or needs_human)
        if intent_signals:
            reasons.append(f"Намерение игрока: {intent_label} (уверенность {int(intent_conf * 100)}%)")

        # 8. Fraud / Anti-fake Detection (with evidence cross-check)
        reporter_nick = facts.reporter_nickname or (request.nickname if request.nickname else None)
        offender_nick = facts.offender_nickname or (request.offender_nickname if request.offender_nickname else None)
        fraud_result = fraud_detector.analyze(
            ticket_id=ticket_id,
            reporter=reporter_nick,
            offender=offender_nick,
            text=combined_player_text,
            category=category,
            confidence=confidence,
            has_evidence=facts.has_evidence,
            attachments_count=total_atts,
            server=facts.server or request.server,
            evidence_report=evidence_report,
        )
        if ticket_id not in self._recorded_ticket_ids and not request.regenerate:
            fraud_detector.record_complaint(
                ticket_id=ticket_id,
                reporter=reporter_nick,
                offender=offender_nick,
                text=combined_player_text,
                category=category,
                server=facts.server or request.server,
                fraud_score=fraud_result["fraud_risk_score"],
                reasons=fraud_result["reasons"],
                contradictions=fraud_result.get("contradictions", []),
            )

        if fraud_result["is_suspicious"] and not is_spam:
            reasons.append(
                f"⚠️ Fraud Risk Score: {fraud_result['fraud_risk_score']}/100 ({fraud_result['risk_label']}). "
                f"Требует ручной проверки. Автоматические наказания строго заблокированы."
            )
            needs_human = True
            if not needs_human_reason:
                needs_human_reason = "Обнаружены подозрительные паттерны или противоречия — требуется ревью администратором"

        # 9. Player (Offender) History by Nickname AND In-Game ID
        offender_history = None
        offender_id = facts.offender_id
        if offender_nick or offender_id:
            lookup_key = offender_nick or str(offender_id)
            offender_history = player_history_store.get_player_history(lookup_key, player_id=offender_id)
            if ticket_id not in self._recorded_ticket_ids and not request.regenerate:
                player_history_store.record_complaint_against(
                    offender_nickname=offender_nick or f"ID_{offender_id}",
                    ticket_id=ticket_id,
                    category=category,
                    confirmed=False,
                    offender_id=offender_id,
                    server=facts.server if isinstance(facts.server, int) else None,
                )
            if offender_history and offender_history.get("found") and offender_history.get("risk_level") in ("high", "critical"):
                reasons.append(
                    f"🔁 Повторный нарушитель: {offender_history.get('nickname')} "
                    f"[ID: {offender_history.get('player_id') or '—'}] "
                    f"(подтверждено нарушений: {offender_history.get('confirmed_violations', 0)}, "
                    f"всего жалоб: {offender_history.get('total_complaints', 0)})"
                )

        if not request.regenerate:
            self._recorded_ticket_ids.add(ticket_id)

        # 10. Direct Admin Call Handling (when player explicitly asks for a live human admin)
        admin_request_id = None
        admin_request_status = None
        if intent == "admin_request":
            existing_msgs = [m.dict() for m in request.messages]
            ar = admin_request_store.create(
                player_nickname=reporter_nick or f"Player_{facts.reporter_id or 'Guest'}",
                server=facts.server or request.server,
                context_messages=existing_msgs,
                ticket_id=ticket_id,
                language=language,
            )
            admin_request_id = ar["id"]
            admin_request_status = ar["status"]
            reasons.append(
                f"👤 Игрок вызвал живого администратора → создано обращение {admin_request_id} (статус: Ожидание)"
            )
            needs_human = True
            requires_admin = True
            needs_human_reason = "Прямой вызов живого администратора игроком"
            if record_cache:
                telegram_notifier.notify_direct_admin_call(ar)

        # 11. Confidence Score & Auto-handling Routing
        auto_handled = bool(not needs_human and confidence >= 0.70)
        if auto_handled:
            routing_decision = "auto_handled"
        elif intent == "admin_request":
            routing_decision = "live_admin_call"
        elif (
            priority == "critical"
            or category in ("ban_appeal", "scam_fraud", "cheating_exploits", "account_recovery", "billing_refund")
            or fraud_result["fraud_risk_score"] >= 60
        ):
            routing_decision = "senior_admin"
        else:
            routing_decision = "junior_admin"

        # 12. Generate Safe Grounded Draft Response
        draft_response, cost_usd = self.generator.generate_draft(
            language=language,
            category=category,
            player_text=combined_player_text,
            facts=facts,
            cited_rules=cited_rules,
            valid_rule_ids=self.rag.valid_rule_ids,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
            needs_human=needs_human,
            reasons=reasons,
            intent=intent,
            admin_request_id=admin_request_id,
        )

        # Final guardrail verification on the outgoing draft
        is_safe, safety_violations = SafetyGuard.validate_draft(
            draft_response,
            valid_rule_ids=self.rag.valid_rule_ids,
            unknown_user_rules=facts.unknown_mentioned_rules,
        )
        if not is_safe:
            reasons.append(f"Сработал защитный фильтр черновика: {', '.join(safety_violations)}")
            needs_human = True
            auto_handled = False
            if not needs_human_reason:
                needs_human_reason = "Черновик требует ручной проверки безопасности оператором"

        # 13. AI Admin Copilot Summary
        ai_summary = self._copilot.generate_summary(
            ticket_id=ticket_id,
            original_text=combined_player_text,
            language=language,
            category=category,
            priority=priority,
            confidence=confidence,
            facts=facts.dict(),
            cited_rules=[r.dict() for r in cited_rules],
            needs_human_reason=needs_human_reason,
            fraud_risk_score=fraud_result["fraud_risk_score"],
            intent=intent,
            subcategory=subcategory,
            subcategory_name=subcategory_name,
            evidence_report=evidence_report,
            offender_history=offender_history,
        )

        # 14. Telegram Notification for Complaints Requiring Admin Intervention
        telegram_notification = None
        if needs_human and intent != "admin_request" and record_cache and not request.regenerate:
            cat_name_for_tg = (
                self.taxonomy.categories[category].name
                if category in self.taxonomy.categories
                else category
            )
            telegram_notification = telegram_notifier.notify_ticket_escalation(
                ticket_id=ticket_id,
                player_nickname=reporter_nick,
                player_id=facts.reporter_id,
                offender_nickname=offender_nick,
                offender_id=facts.offender_id,
                server=facts.server or request.server,
                category=category,
                category_name=cat_name_for_tg,
                subcategory=subcategory_name,
                priority=priority,
                confidence=confidence,
                ai_summary_text=ai_summary.get("summary", ""),
                needs_human_reason=needs_human_reason,
                rule_ids=rule_ids,
                fraud_risk_score=fraud_result["fraud_risk_score"],
            )

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)

        # Update privacy-safe metrics
        self._record_metrics(
            category=category,
            subcategory=subcategory,
            priority=priority,
            language=language,
            needs_human=needs_human,
            auto_handled=auto_handled,
            is_admin_call=(intent == "admin_request"),
            is_suspicious=fraud_result["is_suspicious"],
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
        )

        logger.info(
            "triage_completed ticket_id=%s category=%s subcategory=%s priority=%s lang=%s needs_human=%s rules=%s latency_ms=%.2f",
            ticket_id,
            category,
            subcategory,
            priority,
            language,
            needs_human,
            rule_ids,
            elapsed_ms,
        )

        cat_name = (
            self.taxonomy.categories[category].name
            if category in self.taxonomy.categories
            else category
        )

        return TriageResponse(
            ticket_id=ticket_id,
            language=language,
            category=category,
            category_name=cat_name,
            subcategory=subcategory,
            subcategory_name=subcategory_name,
            priority=priority,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
            facts=facts,
            draft_response=draft_response,
            rule_ids=rule_ids,
            cited_rules=cited_rules,
            needs_human=needs_human,
            needs_human_reason=needs_human_reason,
            confidence=confidence,
            confidence_pct=confidence_pct,
            auto_handled=auto_handled,
            routing_decision=routing_decision,
            reasons=reasons,
            processing_time_ms=elapsed_ms,
            cost_usd=cost_usd,
            intent=intent,
            intent_label=intent_label,
            intent_confidence=round(intent_conf, 3),
            requires_admin=requires_admin,
            fraud_risk_score=fraud_result["fraud_risk_score"],
            fraud_risk_level=fraud_result["risk_level"],
            fraud_reasons=fraud_result["reasons"],
            fraud_related_tickets=fraud_result.get("related_tickets", []),
            admin_request_id=admin_request_id,
            admin_request_status=admin_request_status,
            player_history=offender_history,
            ai_summary=ai_summary,
            evidence_report=evidence_report,
            telegram_notification=telegram_notification,
        )

    def _evaluate_human_escalation(
        self,
        category: str,
        priority: str,
        language: str,
        facts,
        cited_rules: List[CitedRule],
        has_reliable_rule: bool,
        confidence: float,
        is_spam: bool,
        is_duplicate: bool,
        player_text: str,
    ) -> tuple[bool, Optional[str]]:
        escalation_reasons: List[str] = []

        if facts.unknown_mentioned_rules:
            escalation_reasons.append(
                f"Игрок указал неизвестный пункт правил ({', '.join(facts.unknown_mentioned_rules)}), отсутствующий в rules.json"
            )

        if category == "ban_appeal":
            escalation_reasons.append(
                "Обжалование наказания и запрос на разбан рассматриваются только старшей администрацией (п. 7.1)"
            )
        elif category == "billing_refund":
            escalation_reasons.append(
                "Финансовые вопросы, донат и возврат средств требуют ручной проверки чека специалистом (п. 6.1, 6.2)"
            )
        elif category == "account_recovery":
            escalation_reasons.append(
                "Восстановление доступа или взлом аккаунта требует верификации регистрационных данных владельца (п. 1.2, 1.3)"
            )
        elif category == "cheating_exploits":
            escalation_reasons.append(
                "Подозрение на читы или багоюз требует проверки античит-логов и видеозаписи администратором (п. 5.1, 5.2)"
            )
        elif category == "scam_fraud":
            escalation_reasons.append(
                "Жалоба на NonRP обман/мошенничество требует сверки условий сделки и логов передач старшим администратором (п. 4.1)"
            )
        elif category in ("insult_toxicity", "rp_violation"):
            if not facts.has_evidence:
                escalation_reasons.append(
                    "Жалоба на нарушение игрока без приложенных доказательств: требуется запрос материалов или проверка логов человеком (п. 7.2)"
                )
            else:
                escalation_reasons.append(
                    "Окончательное решение о выдаче наказания игроку по приложенным доказательствам принимает модератор"
                )
        elif category == "technical_bug":
            lower = player_text.lower()
            if any(w in lower for w in ("пропал", "исчез", "слетел", "потерял", "багажник", "тюнинг", "жоғалды", "yo'qoldi")):
                escalation_reasons.append(
                    "Заявлена пропажа игрового имущества из-за тех. сбоя — требуется проверка серверных логов (п. 8.1)"
                )

        if not has_reliable_rule and not is_spam and category != "spam_offtopic":
            escalation_reasons.append(
                "Не найдено надежного подтверждения в базе правил (rules.json) — передача человеку без утверждения нарушения"
            )

        if confidence < 0.68 and not is_spam:
            escalation_reasons.append(
                f"Низкая уверенность классификации ({confidence:.2f}) требует ручной проверки"
            )

        if escalation_reasons:
            return True, "; ".join(escalation_reasons)

        if category in self.taxonomy.categories and self.taxonomy.categories[category].needs_human_default:
            return True, f"Категория '{category}' по регламенту требует подтверждения оператором"

        return False, None

    def _record_metrics(
        self,
        category: str,
        subcategory: Optional[str],
        priority: str,
        language: str,
        needs_human: bool,
        auto_handled: bool,
        is_admin_call: bool,
        is_suspicious: bool,
        elapsed_ms: float,
        cost_usd: float,
    ) -> None:
        self.analytics["total_processed"] += 1
        if needs_human:
            self.analytics["escalated_count"] += 1
        if auto_handled:
            self.analytics["auto_handled_count"] += 1
        if is_admin_call:
            self.analytics["admin_calls_count"] += 1
        if is_suspicious:
            self.analytics["suspicious_count"] += 1

        self.analytics["category_counts"][category] = (
            self.analytics["category_counts"].get(category, 0) + 1
        )
        if subcategory:
            self.analytics["subcategory_counts"][subcategory] = (
                self.analytics["subcategory_counts"].get(subcategory, 0) + 1
            )
        self.analytics["priority_counts"][priority] = (
            self.analytics["priority_counts"].get(priority, 0) + 1
        )
        self.analytics["language_counts"][language] = (
            self.analytics["language_counts"].get(language, 0) + 1
        )
        self.analytics["latencies_ms"].append(elapsed_ms)
        self.analytics["total_cost_usd"] = round(
            self.analytics["total_cost_usd"] + cost_usd, 6
        )

    def get_analytics_summary(self) -> Dict[str, Any]:
        total = self.analytics["total_processed"]
        latencies = sorted(self.analytics["latencies_ms"])
        if latencies:
            avg_ms = round(sum(latencies) / len(latencies), 2)
            p95_idx = min(len(latencies) - 1, int(math.ceil(0.95 * len(latencies)) - 1)) if latencies else 0
            p95_ms = round(latencies[max(0, p95_idx)], 2)
        else:
            avg_ms = 0.0
            p95_ms = 0.0

        escalation_rate = round((self.analytics["escalated_count"] / total) * 100.0, 1) if total > 0 else 0.0
        auto_rate = round((self.analytics["auto_handled_count"] / total) * 100.0, 1) if total > 0 else 0.0
        avg_cost = round(self.analytics["total_cost_usd"] / total, 6) if total > 0 else 0.0

        cat_list = [
            {
                "id": cid,
                "name": self.taxonomy.categories[cid].name if cid in self.taxonomy.categories else cid,
                "count": cnt,
                "share_pct": round((cnt / total) * 100.0, 1) if total > 0 else 0.0,
            }
            for cid, cnt in self.analytics["category_counts"].items()
        ]
        cat_list.sort(key=lambda x: x["count"], reverse=True)

        fb_stats = feedback_store.get_stats()

        return {
            "total_processed": total,
            "escalated_count": self.analytics["escalated_count"],
            "auto_handled_count": self.analytics["auto_handled_count"],
            "auto_handled_rate_pct": auto_rate,
            "admin_calls_count": admin_request_store.get_stats()["total"],
            "admin_calls_waiting": admin_request_store.get_waiting_count(),
            "suspicious_count": fraud_detector.get_statistics()["suspicious_flagged"],
            "repeat_offenders_count": len(player_history_store.get_repeat_offenders(min_complaints=2)),
            "senior_escalated_count": len(self.analytics["senior_escalated_ids"]),
            "escalation_rate_pct": escalation_rate,
            "avg_processing_time_ms": avg_ms,
            "p95_processing_time_ms": p95_ms,
            "avg_cost_usd": avg_cost,
            "ai_accuracy_pct": fb_stats.get("accuracy_pct"),
            "feedback_total": fb_stats.get("total_feedback", 0),
            "categories": cat_list,
            "subcategories": self.analytics["subcategory_counts"],
            "priorities": self.analytics["priority_counts"],
            "languages": self.analytics["language_counts"],
            "using_synthetic_data": settings.is_using_synthetic_train,
        }
