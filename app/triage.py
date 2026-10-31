"""
End-to-end Triage Service orchestrating extraction, classification, RAG over rules,
safety verification, and privacy-preserving analytics.
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

        # Privacy-safe runtime analytics (no raw player PII stored in telemetry counters)
        self.analytics: Dict[str, Any] = {
            "total_processed": 0,
            "escalated_count": 0,
            "senior_escalated_ids": set(),
            "category_counts": {cid: 0 for cid in self.taxonomy.valid_categories},
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
            normalize_rule_fn=self.rag.normalize_rule_ref,
        )
        if facts.is_mixed_language:
            reasons.append(
                f"Обнаружено смешение языков ({', '.join(facts.detected_languages)}); основной язык ответа: '{language}'"
            )
        else:
            reasons.append(f"Определен язык обращения: '{language}'")

        if facts.unknown_mentioned_rules:
            reasons.append(
                f"Игрок сослался на несуществующий в rules.json пункт правил: {', '.join(facts.unknown_mentioned_rules)}"
            )

        # 3. Category & Priority Classification (strictly within taxonomy.json)
        category, priority, confidence, cls_reasons = self.classifier.classify(
            text=combined_player_text,
            kind=request.kind or "complaint",
            facts=facts,
            is_spam=is_spam,
            is_duplicate=is_duplicate,
        )
        reasons.extend(cls_reasons)

        # 4. RAG Retrieval over rules.json
        cat_rule_ids = (
            self.taxonomy.categories[category].related_rule_prefixes
            if category in self.taxonomy.categories
            else []
        )
        rag_query = f"{combined_player_text} {facts.rule_refs_raw or ''}".strip()
        cited_rules, _, unknown_rules, has_reliable_rule = self.rag.retrieve(
            query_text=rag_query,
            category=category,
            mentioned_rule_ids=facts.mentioned_rules,
            category_rule_ids=cat_rule_ids,
            top_k=3,
        )

        # If complaint has insufficient evidence, include complaint evidence rule ("ЖБ 1.2.2" or "ЖБ 1.1.2" or legacy "7.2")
        complaint_cats = {
            "complaint.insult", "complaint.dm", "complaint.scam", "complaint.cheat",
            "complaint.rmt_ads", "complaint.invalid",
            "insult_toxicity", "rp_violation", "scam_fraud", "cheating_exploits",
        }
        if not facts.has_evidence and category in complaint_cats:
            for ev_rule_id in ("ЖБ 1.2.2", "ЖБ 1.1.2", "7.2"):
                if self.rag.is_valid_rule_id(ev_rule_id) and not any(r.id == ev_rule_id for r in cited_rules):
                    r_ev = self.rag.rules_by_id[ev_rule_id]
                    cited_rules.append(
                        CitedRule(
                            id=ev_rule_id,
                            section=r_ev.section,
                            title=r_ev.title,
                            quote=r_ev.text,
                            punishment=r_ev.punishment,
                            relevance_score=0.68,
                        )
                    )
                    break

        # If spam, duplicate, or offtopic, include rule "3.1" (or legacy "7.3") if present in rules.json
        if is_spam or is_duplicate or category in ("report.offtopic", "spam_offtopic"):
            for sp_rule_id in ("3.1", "7.3"):
                if self.rag.is_valid_rule_id(sp_rule_id) and not any(r.id == sp_rule_id for r in cited_rules):
                    r_sp = self.rag.rules_by_id[sp_rule_id]
                    cited_rules.insert(
                        0,
                        CitedRule(
                            id=sp_rule_id,
                            section=r_sp.section,
                            title=r_sp.title,
                            quote=r_sp.text,
                            punishment=r_sp.punishment,
                            relevance_score=0.85,
                        ),
                    )
                    break

        # Strict verification that every cited rule ID exists in rules.json
        cited_rules = [r for r in cited_rules if self.rag.is_valid_rule_id(r.id)]
        rule_ids = self.rag.verify_rule_ids([r.id for r in cited_rules])

        if rule_ids:
            reasons.append(f"Найдены подтверждающие пункты rules.json: {', '.join(rule_ids)}")
        else:
            reasons.append("Надежного подтверждения в rules.json не найдено")

        # 5. Determine `needs_human` and `needs_human_reason`
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
        if needs_human and needs_human_reason:
            reasons.append(f"Требуется оператор: {needs_human_reason}")

        # 6. Generate Safe Grounded Draft Response
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
            if not needs_human_reason:
                needs_human_reason = "Черновик требует ручной проверки безопасности оператором"

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)

        # Update privacy-safe metrics
        self._record_metrics(
            category=category,
            priority=priority,
            language=language,
            needs_human=needs_human,
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
        )

        # Log processing time without personal data (no nicknames or raw message text)
        logger.info(
            "triage_completed ticket_id=%s category=%s priority=%s lang=%s needs_human=%s rules=%s latency_ms=%.2f",
            ticket_id,
            category,
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
            reasons=reasons,
            processing_time_ms=elapsed_ms,
            cost_usd=cost_usd,
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

        # Check explicit `needs_human_always` from taxonomy.json
        if category in self.taxonomy.needs_human_always:
            if category in ("complaint.admin", "ban_appeal"):
                escalation_reasons.append(
                    "Жалоба на администратора / обжалование наказания рассматривается только старшей администрацией (п. 3.4)"
                )
            elif category in ("report.payment", "billing_refund"):
                escalation_reasons.append(
                    "Платёж и донат требуют ручной сверки транзакции и чека профильным специалистом"
                )
            elif category in ("report.account", "account_recovery"):
                escalation_reasons.append(
                    "Вопросы доступа, взлома или пропажи имущества на аккаунте требуют верификации владельца (п. 1.4)"
                )
            else:
                escalation_reasons.append(
                    f"Категория '{category}' входит в список обязательной проверки человеком (needs_human_always)"
                )
        elif category in ("ban_appeal", "complaint.admin"):
            escalation_reasons.append(
                "Обжалование наказания и жалоба на администратора рассматриваются старшей администрацией (п. 3.4)"
            )
        elif category in ("billing_refund", "report.payment"):
            escalation_reasons.append(
                "Финансовые вопросы, донат и возврат средств требуют ручной проверки чека специалистом"
            )
        elif category in ("account_recovery", "report.account"):
            escalation_reasons.append(
                "Восстановление доступа или взлом аккаунта требует верификации регистрационных данных владельца (п. 1.4)"
            )
        elif category in ("cheating_exploits", "complaint.cheat"):
            escalation_reasons.append(
                "Подозрение на читы или багоюз требует проверки доказательств и логов администратором (п. 5.1, 5.2)"
            )
        elif category in ("scam_fraud", "complaint.scam"):
            escalation_reasons.append(
                "Жалоба на обман/мошенничество с имуществом требует сверки условий сделки и логов передач администратором (п. 4.1, 4.2)"
            )
        elif category == "complaint.rmt_ads":
            escalation_reasons.append(
                "Подозрение на рекламу или RMT (продажу за реальные деньги) требует проверки логов старшим администратором (п. 2.3, 4.4)"
            )
        elif category in ("insult_toxicity", "rp_violation", "complaint.insult", "complaint.dm", "complaint.other", "report.complaint"):
            if not facts.has_evidence:
                escalation_reasons.append(
                    "Жалоба на нарушение игрока без приложенных доказательств: требуется запрос материалов или слежка в игре (п. ЖБ 1.2.2)"
                )
            else:
                escalation_reasons.append(
                    "Окончательное решение о выдаче наказания игроку по приложенным доказательствам принимает человек-модератор"
                )
        elif category == "complaint.invalid":
            escalation_reasons.append(
                "Жалоба без доказательств или не по форме требует проверки модератором перед закрытием или запросом материалов (п. ЖБ 1.1.2, ЖБ 1.2.2)"
            )
        elif category in ("technical_bug", "report.help"):
            lower = player_text.lower()
            if any(w in lower for w in ("пропал", "исчез", "слетел", "потерял", "багажник", "тюнинг", "застрял", "телепорт", "спавн", "флип")):
                escalation_reasons.append(
                    "Требуется действие администратора в игре (телепорт/спавн) или проверка технических логов"
                )

        # For gameplay questions (`report.question`), if no rule was needed or found, only escalate if confidence is low
        is_offtopic_or_question = is_spam or category in ("spam_offtopic", "report.offtopic", "report.question", "gameplay_question")
        if not has_reliable_rule and not is_offtopic_or_question:
            escalation_reasons.append(
                "Не найдено надежного подтверждения в базе правил (rules.json) — передача человеку без утверждения нарушения"
            )

        if confidence < 0.65 and not is_spam and category not in ("report.offtopic", "spam_offtopic"):
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
        priority: str,
        language: str,
        needs_human: bool,
        elapsed_ms: float,
        cost_usd: float,
    ) -> None:
        self.analytics["total_processed"] += 1
        if needs_human:
            self.analytics["escalated_count"] += 1
        self.analytics["category_counts"][category] = (
            self.analytics["category_counts"].get(category, 0) + 1
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

        return {
            "total_processed": total,
            "escalated_count": self.analytics["escalated_count"],
            "senior_escalated_count": len(self.analytics["senior_escalated_ids"]),
            "escalation_rate_pct": escalation_rate,
            "avg_processing_time_ms": avg_ms,
            "p95_processing_time_ms": p95_ms,
            "avg_cost_usd": avg_cost,
            "categories": cat_list,
            "priorities": self.analytics["priority_counts"],
            "languages": self.analytics["language_counts"],
            "using_synthetic_data": settings.is_using_synthetic_train,
        }
