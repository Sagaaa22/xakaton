"""
AI Admin Copilot — helper module for admin decision-making.
Provides: case summaries, rule lookup, response drafting,
similar case lookup, contradiction detection, and missing evidence hints.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.admin_copilot")


RESPONSE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "insult_toxicity": {
        "ru": "Здравствуйте, {player}! Ваша жалоба на оскорбления и токсичное поведение игрока {offender} принята к рассмотрению. По результатам проверки доказательств будет принято решение согласно п. 2.1 правил проекта. Ответ получите в течение {sla}.",
        "kk": "Сәлем, {player}! {offender} ойыншысына арналған шағымыңыз қабылданды. Дәлелдер тексерілгеннен кейін шешім қабылданады.",
        "en": "Hello, {player}! Your complaint about player {offender}'s toxic behavior has been received. We'll review the evidence and take action per rule 2.1.",
        "uz": "Salom, {player}! {offender} o'yinchisi haqidagi shikoyatingiz qabul qilindi.",
    },
    "scam_fraud": {
        "ru": "Здравствуйте, {player}! Ваша жалоба на мошенничество принята в приоритетном порядке. Для рассмотрения необходимы логи сделки и скриншоты/видео переговоров. Дело передано старшему администратору согласно п. 4.1.",
        "kk": "Сәлем! Алаяқтық туралы шағымыңыз басым тәртіпте қабылданды. Мәмілені растайтын дәлелдер қажет.",
        "en": "Hello! Your fraud complaint has been received with high priority. We need trade logs and screenshots as evidence. Escalated to senior admin per rule 4.1.",
        "uz": "Salom! Firibgarlik haqidagi shikoyatingiz ustuvor tartibda qabul qilindi.",
    },
    "cheating_exploits": {
        "ru": "Здравствуйте, {player}! Жалоба на использование сторонних программ или багоюза передана в отдел античита для проверки серверных логов. Решение будет принято после технической проверки (п. 5.1).",
        "kk": "Сәлем! Жалоба античит бөліміне жіберілді. Техникалық тексеруден кейін шешім қабылданады.",
        "en": "Hello! Your cheat report has been forwarded to the anti-cheat department for log verification (rule 5.1).",
        "uz": "Salom! Shikoyatiz antichit bo'limiga yuborildi.",
    },
    "ban_appeal": {
        "ru": "Здравствуйте, {player}! Ваша апелляция на блокировку аккаунта принята. Рассмотрение апелляций проводится только старшей администрацией согласно п. 7.1. Решение будет принято в течение {sla}.",
        "kk": "Сәлем! Апелляцияңыз қабылданды. Тек жоғарғы администрация қарайды (п. 7.1).",
        "en": "Hello, {player}! Your ban appeal has been received and will be reviewed by senior administration only (rule 7.1). Decision within {sla}.",
        "uz": "Salom! Apellyasiyangiz qabul qilindi.",
    },
    "billing_refund": {
        "ru": "Здравствуйте, {player}! Ваш запрос по вопросам доната/возврата средств принят. Для рассмотрения необходимы чеки оплаты. Пожалуйста, прикрепите подтверждение транзакции (п. 6.1).",
        "kk": "Сәлем! Донат/қайтару сұрауыңыз қабылданды. Төлем чегін тіркеңіз (п. 6.1).",
        "en": "Hello! Your refund/donation request received. Please provide payment receipts (rule 6.1).",
        "uz": "Salom! To'lov chekini ilova qiling.",
    },
    "account_recovery": {
        "ru": "Здравствуйте, {player}! Запрос на восстановление аккаунта принят. Для верификации потребуется предоставить регистрационные данные: никнейм, сервер регистрации, город и провайдер (п. 1.3).",
        "kk": "Сәлем! Аккаунты қалпына келтіру сұрауы қабылданды. Тіркеу деректерін ұсыныңыз (п. 1.3).",
        "en": "Hello! Account recovery request received. Please provide registration details: server, city, provider (rule 1.3).",
        "uz": "Salom! Akkauntni tiklash so'rovi qabul qilindi.",
    },
    "rp_violation": {
        "ru": "Здравствуйте, {player}! Жалоба на нарушение правил RolePlay принята к рассмотрению. Для вынесения решения необходимы доказательства: скриншот или видео с видимым /time и ID нарушителя (п. 7.2).",
        "kk": "Сәлем! РП ережелерін бұзу туралы шағымыңыз қабылданды. Дәлелдер қажет (п. 7.2).",
        "en": "Hello! Your RP violation report received. Please provide screenshots or video with /time visible (rule 7.2).",
        "uz": "Salom! RP qoidalarini buzish haqidagi shikoyat qabul qilindi.",
    },
    "technical_bug": {
        "ru": "Здравствуйте, {player}! Ваше сообщение о технической проблеме принято. Наша техническая команда изучит логи сервера. При пропаже имущества необходимо зафиксировать время события (п. 8.1).",
        "kk": "Сәлем! Техникалық мәселе туралы хабарлама қабылданды.",
        "en": "Hello! Your technical issue report received. Our team will check server logs.",
        "uz": "Salom! Texnik muammo haqidagi murojaat qabul qilindi.",
    },
    "gameplay_question": {
        "ru": "Здравствуйте, {player}! Спасибо за ваш вопрос. {answer} Если у вас остались вопросы, вы также можете воспользоваться командой /report → Вопрос внутри игры (п. 1.1).",
        "kk": "Сәлем! Сұрағыңыз үшін рахмет.",
        "en": "Hello, {player}! Thank you for your question. {answer} You can also use /report → Question in-game (rule 1.1).",
        "uz": "Salom! Savolingiz uchun rahmat.",
    },
}

SLA_BY_PRIORITY = {
    "critical": "1–2 часов",
    "high": "4–6 часов",
    "medium": "24 часов",
    "low": "48–72 часов",
}

MISSING_EVIDENCE_HINTS: Dict[str, List[str]] = {
    "insult_toxicity": [
        "Скриншот или видео с оскорблением в чате",
        "Видимое игровое время (/time) на скриншоте",
        "ID или никнейм нарушителя на скриншоте",
        "Для голосовых оскорблений — видеозапись с аудио",
    ],
    "rp_violation": [
        "Видеозапись инцидента (предпочтительно)",
        "Скриншот с ID нарушителя и временем /time",
        "Описание ситуации: где, когда, при каких обстоятельствах",
        "Показания других свидетелей (если есть)",
    ],
    "scam_fraud": [
        "Скриншот условий сделки (переписка/диалог)",
        "Скриншот или видео момента передачи имущества",
        "ID или никнейм предполагаемого мошенника",
        "Стоимость/описание утраченного имущества",
        "Время и сервер, на котором произошла сделка",
    ],
    "cheating_exploits": [
        "Видеозапись (обязательно для читов!)",
        "Временные метки подозрительных действий",
        "ID и никнейм предполагаемого читера",
        "Сервер и время инцидента",
        "Описание подозрительного поведения",
    ],
    "billing_refund": [
        "Чек/скриншот оплаты",
        "ID транзакции",
        "Дата и время платежа",
        "Сумма платежа и что должно было начислиться",
    ],
    "account_recovery": [
        "Никнейм аккаунта",
        "Сервер регистрации",
        "Город и провайдер регистрации",
        "Чеки пополнений (если есть)",
        "E-mail регистрации (если применимо)",
    ],
    "ban_appeal": [
        "Дата и срок блокировки",
        "Администратор, выдавший бан (если известен)",
        "Пункт правил, по которому выдан бан",
        "Объяснение причины оспаривания",
        "Скриншот уведомления о бане",
    ],
}

POSSIBLE_CONTRADICTIONS: List[Tuple[str, str, str]] = [
    ("scam_fraud", "нет доказательств", "Заявление о мошенничестве без доказательств трудно проверить"),
    ("cheating_exploits", "нет видео", "Жалобы на читы без видеозаписи редко могут быть рассмотрены"),
    ("rp_violation", "нет скриншота", "Нарушения РП без доказательств: слово против слова"),
]


class AdminCopilot:
    """AI helper for administrators reviewing tickets."""

    def __init__(self, rules_by_id: Dict[str, Any], taxonomy_categories: Dict[str, Any]) -> None:
        self.rules_by_id = rules_by_id
        self.taxonomy_categories = taxonomy_categories

    def generate_summary(
        self,
        ticket_id: str,
        original_text: str,
        language: str,
        category: str,
        priority: str,
        confidence: float,
        facts: Dict[str, Any],
        cited_rules: List[Dict[str, Any]],
        needs_human_reason: Optional[str],
        fraud_risk_score: int = 0,
        intent: str = "complaint",
    ) -> Dict[str, Any]:
        """Generate a complete AI summary for the admin."""
        
        # Category name
        cat_obj = self.taxonomy_categories.get(category)
        category_name = cat_obj.name if cat_obj and hasattr(cat_obj, "name") else category

        # Find missing evidence
        missing_evidence = MISSING_EVIDENCE_HINTS.get(category, [])
        has_evidence = facts.get("has_evidence", False)
        attachments = facts.get("attachments_count", 0)
        
        missing_specific = []
        if not has_evidence:
            missing_specific = missing_evidence[:3]

        # Find potential contradictions
        contradictions = []
        for cat_check, signal, description in POSSIBLE_CONTRADICTIONS:
            if cat_check == category:
                if not has_evidence and attachments == 0:
                    contradictions.append(description)

        # Build rule suggestions
        rule_suggestions = []
        for rule_id, rule_obj in self.rules_by_id.items():
            if hasattr(rule_obj, "category") and rule_obj.category == category:
                rule_suggestions.append({
                    "id": rule_id,
                    "title": getattr(rule_obj, "title", ""),
                    "punishment": getattr(rule_obj, "punishment", ""),
                    "section": getattr(rule_obj, "section", ""),
                })
            if len(rule_suggestions) >= 3:
                break

        # Build recommended actions
        recommendations = []
        if not has_evidence and category in MISSING_EVIDENCE_HINTS:
            recommendations.append("Запросить доказательства у репортера")
        if confidence < 0.70:
            recommendations.append("Перепроверить категорию вручную из-за низкой уверенности ИИ")
        if fraud_risk_score >= 60:
            recommendations.append(f"Проверить подозрительность жалобы (Fraud Risk: {fraud_risk_score}%)")
        if category in ("cheating_exploits", "scam_fraud", "ban_appeal", "billing_refund", "account_recovery"):
            recommendations.append("Требуется проверка логов / чеков / истории транзакций")
        if category == "cheating_exploits":
            recommendations.append("Запросить видеодоказательства для передачи в античит")

        return {
            "ticket_id": ticket_id,
            "summary": self._build_plain_summary(
                original_text, language, category_name, priority, confidence,
                facts, needs_human_reason
            ),
            "category": category,
            "category_name": category_name,
            "priority": priority,
            "confidence_pct": int(confidence * 100),
            "intent": intent,
            "language": language,
            "needs_human_reason": needs_human_reason,
            "fraud_risk_score": fraud_risk_score,
            "has_evidence": has_evidence,
            "attachments_count": attachments,
            "missing_evidence": missing_specific,
            "all_evidence_hints": missing_evidence,
            "contradictions": contradictions,
            "rule_suggestions": rule_suggestions,
            "cited_rule_ids": [r.get("id", "") for r in cited_rules] if cited_rules else [],
            "recommendations": recommendations,
            "sla": SLA_BY_PRIORITY.get(priority, "24 часов"),
            "facts_extracted": {
                "nicknames": facts.get("nicknames", []),
                "offender": facts.get("offender_nickname"),
                "server": facts.get("server"),
                "amounts": facts.get("amounts", []),
                "is_mixed_language": facts.get("is_mixed_language", False),
                "detected_languages": facts.get("detected_languages", []),
            },
        }

    def _build_plain_summary(
        self,
        text: str,
        language: str,
        category_name: str,
        priority: str,
        confidence: float,
        facts: Dict[str, Any],
        needs_human_reason: Optional[str],
    ) -> str:
        """Build a plain-text case summary for admin."""
        snippets = []
        
        if text:
            snippet = text[:200].strip()
            if len(text) > 200:
                snippet += "..."
            snippets.append(f"Содержание обращения: «{snippet}»")
        
        snippets.append(f"Категория: {category_name}")
        snippets.append(f"Приоритет: {priority.upper()}")
        snippets.append(f"Уверенность ИИ: {int(confidence*100)}%")
        
        offender = facts.get("offender_nickname")
        if offender:
            snippets.append(f"Нарушитель: {offender}")
        
        server = facts.get("server")
        if server:
            snippets.append(f"Сервер: {server}")
        
        if facts.get("has_evidence"):
            snippets.append(f"Доказательства: прикреплены ({facts.get('attachments_count', 0)} файл(а))")
        else:
            snippets.append("Доказательства: не приложены")
        
        if needs_human_reason:
            snippets.append(f"Причина передачи администрации: {needs_human_reason[:150]}")
        
        return "\n".join(snippets)

    def build_draft_response(
        self,
        category: str,
        language: str,
        player_nickname: Optional[str] = None,
        offender_nickname: Optional[str] = None,
        priority: str = "medium",
    ) -> str:
        """Generate a suggested admin response based on category and language."""
        lang = language if language in ("ru", "kk", "en", "uz") else "ru"
        templates = RESPONSE_TEMPLATES.get(category, {})
        template = templates.get(lang, templates.get("ru", ""))
        
        if not template:
            if lang == "ru":
                return f"Здравствуйте! Ваше обращение принято и передано на рассмотрение администрации. Ответ последует в течение {SLA_BY_PRIORITY.get(priority, '24 часов')}."
            elif lang == "kk":
                return "Сәлем! Өтінішіңіз қабылданды және администрацияға берілді."
            elif lang == "en":
                return f"Hello! Your request has been received and forwarded to the administration. Response within {SLA_BY_PRIORITY.get(priority, '24 hours')}."
            else:
                return "Salom! Murojaatingiz qabul qilindi."
        
        return template.format(
            player=player_nickname or "Уважаемый игрок",
            offender=offender_nickname or "указанный игрок",
            sla=SLA_BY_PRIORITY.get(priority, "24 часов"),
            answer="Пожалуйста, уточните ваш вопрос.",
        )

    def find_applicable_rules(
        self,
        category: str,
        mentioned_rule_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:  
        """Find rules applicable to this category."""
        results = []
        mentioned = set(mentioned_rule_ids or [])
        
        for rule_id, rule_obj in self.rules_by_id.items():
            rule_cat = getattr(rule_obj, "category", "")
            is_mentioned = rule_id in mentioned
            category_match = rule_cat == category
            
            if category_match or is_mentioned:
                results.append({
                    "id": rule_id,
                    "section": getattr(rule_obj, "section", ""),
                    "title": getattr(rule_obj, "title", ""),
                    "text": getattr(rule_obj, "text", "")[:200],
                    "punishment": getattr(rule_obj, "punishment", ""),
                    "is_mentioned_by_player": is_mentioned,
                    "category_match": category_match,
                })
        
        results.sort(key=lambda x: (x["is_mentioned_by_player"], x["category_match"]), reverse=True)
        return results[:top_k]
