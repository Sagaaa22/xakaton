"""
AI Admin Copilot — helper module for admin decision-making.
Provides:
- Case summaries (краткое содержание жалобы)
- Grand Mobile rule lookup (поиск соответствующих пунктов правил)
- Evidence analysis integration (анализ доказательств)
- Similar resolved cases lookup (поиск похожих рассмотренных обращений)
- Decision recommendations (рекомендации по возможным решениям)
- Player response generation in RU/KK/UZ/EN (генерация ответов игрокам)
- Strict policy: final decision always remains with the human administrator.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.admin_copilot")


RESPONSE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "insult_toxicity": {
        "ru": "Здравствуйте, {player}! Ваша жалоба на оскорбления и токсичное поведение игрока {offender} принята к рассмотрению. По результатам проверки доказательств будет принято решение согласно п. 2.1 правил проекта. Ответ получите в течение {sla}.",
        "kk": "Сәлем, {player}! {offender} ойыншысына арналған шағымыңыз қабылданды. Дәлелдер тексерілгеннен кейін ереженің п. 2.1 сәйкес шешім қабылданады.",
        "en": "Hello, {player}! Your complaint about player {offender}'s toxic behavior has been received. We'll review the evidence and take action per rule 2.1.",
        "uz": "Salom, {player}! {offender} o'yinchisi haqidagi shikoyatingiz qabul qilindi. Dalillar tekshirilgach 2.1-band bo'yicha qaror qabul qilinadi.",
    },
    "scam_fraud": {
        "ru": "Здравствуйте, {player}! Ваша жалоба на мошенничество принята в приоритетном порядке. Для рассмотрения необходимы логи сделки и скриншоты/видео переговоров. Дело передано старшему администратору согласно п. 4.1.",
        "kk": "Сәлем, {player}! Алаяқтық туралы шағымыңыз басым тәртіпте қабылданды. Мәмілені растайтын дәлелдер мен логтар п. 4.1 бойынша тексеріледі.",
        "en": "Hello, {player}! Your fraud complaint has been received with high priority. We need trade logs and screenshots as evidence. Escalated to senior admin per rule 4.1.",
        "uz": "Salom, {player}! Firibgarlik haqidagi shikoyatingiz ustuvor tartibda qabul qilindi (4.1-band).",
    },
    "cheating_exploits": {
        "ru": "Здравствуйте, {player}! Жалоба на использование сторонних программ или багоюза передана в отдел античита для проверки серверных логов. Решение будет принято после технической проверки (п. 5.1).",
        "kk": "Сәлем, {player}! Шағым античит бөліміне жіберілді. Техникалық тексеруден кейін шешім қабылданады (п. 5.1).",
        "en": "Hello, {player}! Your cheat report has been forwarded to the anti-cheat department for log verification (rule 5.1).",
        "uz": "Salom, {player}! Shikoyatingiz antichit bo'limiga yuborildi (5.1-band).",
    },
    "ban_appeal": {
        "ru": "Здравствуйте, {player}! Ваша апелляция на блокировку аккаунта принята. Рассмотрение апелляций проводится только старшей администрацией согласно п. 7.1. Решение будет принято в течение {sla}.",
        "kk": "Сәлем, {player}! Апелляцияңыз қабылданды. Тек жоғарғы әкімшілік қарайды (п. 7.1).",
        "en": "Hello, {player}! Your ban appeal has been received and will be reviewed by senior administration only (rule 7.1). Decision within {sla}.",
        "uz": "Salom, {player}! Apellyasiyangiz qabul qilindi va Bosh ma'muriyat tomonidan ko'rib chiqiladi (7.1-band).",
    },
    "billing_refund": {
        "ru": "Здравствуйте, {player}! Ваш запрос по вопросам доната/возврата средств принят. Для рассмотрения необходимы чеки оплаты. Пожалуйста, прикрепите подтверждение транзакции (п. 6.1).",
        "kk": "Сәлем, {player}! Донат/қайтару сұрауыңыз қабылданды. Төлем чегін тіркеңіз (п. 6.1).",
        "en": "Hello, {player}! Your refund/donation request received. Please provide payment receipts (rule 6.1).",
        "uz": "Salom, {player}! To'lov chekini ilova qiling (6.1-band).",
    },
    "account_recovery": {
        "ru": "Здравствуйте, {player}! Запрос на восстановление аккаунта принят. Для верификации потребуется предоставить регистрационные данные: никнейм, сервер регистрации, город и провайдер (п. 1.3).",
        "kk": "Сәлем, {player}! Аккаунты қалпына келтіру сұрауы қабылданды. Тіркеу деректерін ұсыныңыз (п. 1.3).",
        "en": "Hello, {player}! Account recovery request received. Please provide registration details: server, city, provider (rule 1.3).",
        "uz": "Salom, {player}! Akkauntni tiklash so'rovi qabul qilindi (1.3-band).",
    },
    "rp_violation": {
        "ru": "Здравствуйте, {player}! Жалоба на нарушение правил RolePlay принята к рассмотрению. Для вынесения решения необходимы доказательства: скриншот или видео с видимым /time и ID нарушителя (п. 7.2).",
        "kk": "Сәлем, {player}! РП ережелерін бұзу туралы шағымыңыз қабылданды. Дәлелдер қажет (п. 7.2).",
        "en": "Hello, {player}! Your RP violation report received. Please provide screenshots or video with /time visible (rule 7.2).",
        "uz": "Salom, {player}! RP qoidalarini buzish haqidagi shikoyat qabul qilindi (7.2-band).",
    },
    "technical_bug": {
        "ru": "Здравствуйте, {player}! Ваше сообщение о технической проблеме принято. Наша техническая команда изучит логи сервера. При пропаже имущества необходимо зафиксировать время события (п. 8.1).",
        "kk": "Сәлем, {player}! Техникалық мәселе туралы хабарлама қабылданды (п. 8.1).",
        "en": "Hello, {player}! Your technical issue report received. Our team will check server logs (rule 8.1).",
        "uz": "Salom, {player}! Texnik muammo haqidagi murojaat qabul qilindi (8.1-band).",
    },
    "gameplay_question": {
        "ru": "Здравствуйте, {player}! Спасибо за ваш вопрос. {answer} Если у вас остались вопросы, вы также можете воспользоваться командой /report → Вопрос внутри игры (п. 1.1).",
        "kk": "Сәлем, {player}! Сұрағыңыз үшін рахмет. {answer}",
        "en": "Hello, {player}! Thank you for your question. {answer} You can also use /report → Question in-game (rule 1.1).",
        "uz": "Salom, {player}! Savolingiz uchun rahmat. {answer}",
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

# Reference archive of resolved cases (похожие рассмотренные обращения)
RESOLVED_PRECEDENTS: List[Dict[str, Any]] = [
    {
        "case_id": "arch_0101",
        "category": "rp_violation",
        "subcategory": "dm_rdm",
        "title": "Убийство без RP-причины (DM) у больницы со скриншотом /time",
        "keywords": ["дм", "dm", "убил", "без причины", "стрелял", "больница", "спавн"],
        "rule_id": "3.1",
        "admin_decision": "punishment_confirmed",
        "resolution": "Проверены килл-лог и видео с /time. Выдан деморган (Jail) 60 минут по п. 3.1.",
    },
    {
        "case_id": "arch_0102",
        "category": "rp_violation",
        "subcategory": "db",
        "title": "Намеренный наезд автомобилем (DB) на авторынке",
        "keywords": ["дб", "db", "задавил", "машиной", "тачкой", "сбил", "авторынок"],
        "rule_id": "3.2",
        "admin_decision": "punishment_confirmed",
        "resolution": "Подтвержден намеренный таран игрока транспортом на видео. Выдан Jail 45 минут по п. 3.2.",
    },
    {
        "case_id": "arch_0103",
        "category": "rp_violation",
        "subcategory": "pg",
        "title": "PowerGaming (PG) — безоружный против троих вооруженных",
        "keywords": ["пг", "pg", "пауэргейминг", "кулаками", "против оружия", "один против"],
        "rule_id": "3.4",
        "admin_decision": "punishment_confirmed",
        "resolution": "На видео зафиксировано нападение с кулаками на вооруженный патруль. Выдан Jail 60 минут (п. 3.4).",
    },
    {
        "case_id": "arch_0104",
        "category": "rp_violation",
        "subcategory": "mg",
        "title": "MetaGaming (MG) — использование OOC-информации в IC-чате",
        "keywords": ["мг", "mg", "метагейминг", "дискорд", "форум", "админ в рп чат", "ooc"],
        "rule_id": "3.5",
        "admin_decision": "punishment_confirmed",
        "resolution": "На скриншоте с /time подтвержден MG в РП-чате. Выдан мут 30 минут по п. 3.5.",
    },
    {
        "case_id": "arch_0105",
        "category": "insult_toxicity",
        "subcategory": "family_insult",
        "title": "Упоминание и оскорбление родных в чате (п. 2.2)",
        "keywords": ["оск род", "мать", "мамку", "родных", "семью", "родителей", "анаңды"],
        "rule_id": "2.2",
        "admin_decision": "punishment_confirmed",
        "resolution": "Проверен скриншот с /time и лог чата сервера. Выдан бан на 15 дней по п. 2.2.",
    },
    {
        "case_id": "arch_0106",
        "category": "insult_toxicity",
        "subcategory": "insult_chat",
        "title": "Оскорбление игрока и мат в OOC-чате (п. 2.1)",
        "keywords": ["оскорбил", "оск", "мат", "токсик", "чат", "обозвал", "балағаттады", "so'kdi"],
        "rule_id": "2.1",
        "admin_decision": "punishment_confirmed",
        "resolution": "Нарушение п. 2.1 подтверждено скриншотом с /time. Выдан мут чата на 60 минут.",
    },
    {
        "case_id": "arch_0107",
        "category": "scam_fraud",
        "subcategory": "trade_scam",
        "title": "NonRP обман при обмене авто с доплатой (п. 4.1)",
        "keywords": ["обманул", "кинул", "сделка", "обмен", "машину", "тачку", "доплата", "скам", "алдап"],
        "rule_id": "4.1",
        "admin_decision": "punishment_confirmed",
        "resolution": "Старший администратор сверил видео договора и логи трейда. Аккаунт мошенника заблокирован на 30 дней с обнулением сделки (п. 4.1).",
    },
    {
        "case_id": "arch_0108",
        "category": "cheating_exploits",
        "subcategory": "cheats",
        "title": "Использование SpeedHack / Aimbot / Телепорта (п. 5.1)",
        "keywords": ["чит", "спидхак", "аим", "телепорт", "флай", "гм", "бессмертие", "софт"],
        "rule_id": "5.1",
        "admin_decision": "punishment_confirmed",
        "resolution": "Видеозапись и телеметрия античита подтвердили стороннее ПО. Выдан перманентный бан (п. 5.1).",
    },
    {
        "case_id": "arch_0109",
        "category": "cheating_exploits",
        "subcategory": "bug_abuse",
        "title": "Багоюз / дюп предметов и проход сквозь текстуры (п. 5.2)",
        "keywords": ["багоюз", "дюп", "текстур", "уязвимость", "абуз"],
        "rule_id": "5.2",
        "admin_decision": "punishment_confirmed",
        "resolution": "По логам инвентаря подтвержден намеренный багоюз. Выдан бан 15 дней и изъятие полученных средств (п. 5.2).",
    },
    {
        "case_id": "arch_0110",
        "category": "ban_appeal",
        "subcategory": "ban_appeal",
        "title": "Апелляция на блокировку аккаунта (п. 7.1)",
        "keywords": ["разбан", "забанили", "апелляция", "по ошибке", "несправедливо", "баннан"],
        "rule_id": "7.1",
        "admin_decision": "forwarded",
        "resolution": "Передано Главной Администрации сервера. Запрошена видеозапись со стороны администратора, выдавшего наказание (п. 7.1).",
    },
    {
        "case_id": "arch_0111",
        "category": "billing_refund",
        "subcategory": "billing_refund",
        "title": "Не поступил донат после списания средств (п. 6.2)",
        "keywords": ["донат", "деньги", "чек", "оплата", "не пришли", "задонатил", "возврат"],
        "rule_id": "6.2",
        "admin_decision": "forwarded",
        "resolution": "Проверен банковский чек и ID транзакции в платежном шлюзе. Донат-валюта зачислена вручную финансовым отделом (п. 6.2).",
    },
    {
        "case_id": "arch_0112",
        "category": "technical_bug",
        "subcategory": "property_loss",
        "title": "Пропажа автомобиля / тюнинга после рестарта сервера (п. 8.1)",
        "keywords": ["пропала", "слетел", "рестарт", "тюнинг", "багажник", "инвентарь", "исчезла"],
        "rule_id": "8.1",
        "admin_decision": "forwarded",
        "resolution": "Сверены логи базы данных до и после рестарта. Имущество восстановлено техническим отделом (п. 8.1).",
    },
]


class AdminCopilot:
    """AI helper for administrators reviewing tickets."""

    def __init__(self, rules_by_id: Dict[str, Any], taxonomy_categories: Dict[str, Any]) -> None:
        self.rules_by_id = rules_by_id
        self.taxonomy_categories = taxonomy_categories

    def find_similar_resolved_cases(
        self,
        text: str,
        category: str,
        subcategory: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find similar historical resolved complaints to guide admin decision-making."""
        lower = (text or "").lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in RESOLVED_PRECEDENTS:
            score = 0.0
            if item["category"] == category:
                score += 0.45
            if subcategory and item.get("subcategory") == subcategory:
                score += 0.35
            kw_hits = sum(1 for kw in item.get("keywords", []) if kw in lower)
            if kw_hits:
                score += min(0.35, kw_hits * 0.12)
            if score >= 0.35:
                copy_item = dict(item)
                copy_item["similarity_pct"] = min(98, int(round(score * 100)))
                scored.append((score, copy_item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

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
        subcategory: Optional[str] = None,
        subcategory_name: Optional[str] = None,
        evidence_report: Optional[Dict[str, Any]] = None,
        offender_history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete AI Admin Assistant report for the administrator."""
        cat_obj = self.taxonomy_categories.get(category)
        category_name = cat_obj.name if cat_obj and hasattr(cat_obj, "name") else category

        missing_evidence = MISSING_EVIDENCE_HINTS.get(category, [])
        has_evidence = facts.get("has_evidence", False)
        attachments = facts.get("attachments_count", 0)

        missing_specific = []
        if not has_evidence:
            missing_specific = missing_evidence[:3]

        contradictions = []
        for cat_check, signal, description in POSSIBLE_CONTRADICTIONS:
            if cat_check == category and not has_evidence and attachments == 0:
                contradictions.append(description)
        if evidence_report and evidence_report.get("contradictions"):
            for c in evidence_report["contradictions"]:
                if c not in contradictions:
                    contradictions.append(c)

        # Build rule suggestions
        rule_suggestions = []
        for r in cited_rules:
            rid = r.get("id", "")
            if rid:
                rule_suggestions.append({
                    "id": rid,
                    "title": r.get("title", ""),
                    "punishment": r.get("punishment", ""),
                    "section": r.get("section", ""),
                })
        for rule_id, rule_obj in self.rules_by_id.items():
            if any(rs["id"] == rule_id for rs in rule_suggestions):
                continue
            if hasattr(rule_obj, "category") and rule_obj.category == category:
                rule_suggestions.append({
                    "id": rule_id,
                    "title": getattr(rule_obj, "title", ""),
                    "punishment": getattr(rule_obj, "punishment", ""),
                    "section": getattr(rule_obj, "section", ""),
                })
            if len(rule_suggestions) >= 4:
                break

        # Similar resolved cases
        similar_resolved = self.find_similar_resolved_cases(
            text=original_text,
            category=category,
            subcategory=subcategory,
            top_k=3,
        )

        # Structured decision recommendations
        recommendations: List[str] = []
        decision_options: List[Dict[str, str]] = []

        if intent == "admin_request":
            recommendations.append("Игрок вызвал живого администратора: принять вызов в очереди и подключиться к диалогу")
            decision_options.append({
                "decision": "answered",
                "label": "Подключиться к игроку и решить вопрос в живом диалоге",
                "reason": "Прямой запрос живого администратора",
            })

        if not has_evidence and category in MISSING_EVIDENCE_HINTS:
            recommendations.append("Запросить доказательства у репортера (скриншот/видео с /time)")
            decision_options.append({
                "decision": "info_requested",
                "label": "Запросить дополнительные доказательства (п. 7.2)",
                "reason": "В обращении отсутствуют скриншоты или видео с /time",
            })

        if has_evidence and rule_suggestions:
            top_r = rule_suggestions[0]
            recommendations.append(
                f"После личной верификации доказательств рассмотреть меру по п. {top_r['id']}: {top_r['punishment']}"
            )
            decision_options.append({
                "decision": "punishment_confirmed",
                "label": f"Подтвердить нарушение п. {top_r['id']} ({top_r['title']})",
                "reason": f"Регламентная санкция: {top_r['punishment']} (только после ручной проверки доказательств)",
            })

        if confidence < 0.70:
            recommendations.append("Перепроверить категорию вручную из-за низкой уверенности ИИ")
        if fraud_risk_score >= 45:
            recommendations.append(
                f"Проверить подозрительность жалобы (Fraud Risk: {fraud_risk_score}/100) и сверить доказательства с текстом"
            )
        if offender_history and offender_history.get("confirmed_violations", 0) >= 2:
            recommendations.append(
                f"Учесть рецидив нарушителя: подтвержденных нарушений в истории — {offender_history['confirmed_violations']}"
            )
        if category in ("cheating_exploits", "scam_fraud", "ban_appeal", "billing_refund", "account_recovery"):
            recommendations.append("Требуется проверка серверных логов / чеков / истории транзакций")
            decision_options.append({
                "decision": "forwarded",
                "label": "Передать в профильный отдел (логи / античит / финансы / ГА)",
                "reason": f"Категория «{category_name}» требует проверки серверных логов или платежного шлюза",
            })
        has_video_ev = any(t in (facts.get("evidence_types") or []) for t in ("video_mention", "video_file"))
        if category == "cheating_exploits" and not has_video_ev:
            recommendations.append("Запросить видеодоказательства для передачи в античит")

        decision_options.append({
            "decision": "no_violation",
            "label": "Отклонить жалобу (нарушение не подтверждено)",
            "reason": "При отсутствии подтверждения в логах или противоречивых доказательствах",
        })

        # Multi-scenario response generation in player's language
        reporter_nick = facts.get("reporter_nickname")
        offender_nick = facts.get("offender_nickname")
        primary_draft = self.build_draft_response(
            category=category,
            language=language,
            player_nickname=reporter_nick,
            offender_nickname=offender_nick,
            priority=priority,
        )
        draft_variants = self._build_response_variants(
            language=language,
            category=category,
            player_nickname=reporter_nick,
            offender_nickname=offender_nick,
            top_rule=rule_suggestions[0] if rule_suggestions else None,
        )

        return {
            "ticket_id": ticket_id,
            "summary": self._build_plain_summary(
                original_text,
                language,
                category_name,
                subcategory_name,
                priority,
                confidence,
                facts,
                needs_human_reason,
                evidence_report,
            ),
            "category": category,
            "category_name": category_name,
            "subcategory": subcategory,
            "subcategory_name": subcategory_name,
            "priority": priority,
            "confidence_pct": int(round(confidence * 100)),
            "intent": intent,
            "language": language,
            "needs_human_reason": needs_human_reason,
            "fraud_risk_score": fraud_risk_score,
            "has_evidence": has_evidence,
            "attachments_count": attachments,
            "missing_evidence": missing_specific,
            "all_evidence_hints": missing_evidence,
            "contradictions": contradictions,
            "evidence_summary": evidence_report.get("admin_report") if evidence_report else None,
            "rule_suggestions": rule_suggestions,
            "cited_rule_ids": [r.get("id", "") for r in cited_rules] if cited_rules else [],
            "similar_resolved_cases": similar_resolved,
            "recommendations": recommendations,
            "decision_recommendations": decision_options,
            "suggested_response": primary_draft,
            "response_variants": draft_variants,
            "sla": SLA_BY_PRIORITY.get(priority, "24 часов"),
            "final_decision_by_human": True,
            "disclaimer": "ИИ формирует только аналитическую сводку и рекомендации. Финальное решение всегда принимает администратор.",
            "facts_extracted": {
                "nicknames": facts.get("nicknames", []),
                "player_ids": facts.get("player_ids", []),
                "offender": facts.get("offender_nickname"),
                "offender_id": facts.get("offender_id"),
                "reporter": facts.get("reporter_nickname"),
                "reporter_id": facts.get("reporter_id"),
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
        subcategory_name: Optional[str],
        priority: str,
        confidence: float,
        facts: Dict[str, Any],
        needs_human_reason: Optional[str],
        evidence_report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a concise structured case summary for the administrator."""
        snippets = []

        if text:
            snippet = text[:200].strip()
            if len(text) > 200:
                snippet += "..."
            snippets.append(f"Содержание обращения: «{snippet}»")

        sub_part = f" ({subcategory_name})" if subcategory_name else ""
        snippets.append(f"Категория: {category_name}{sub_part}")
        snippets.append(f"Приоритет: {priority.upper()} | Уверенность ИИ: {int(round(confidence * 100))}%")

        offender = facts.get("offender_nickname")
        offender_id = facts.get("offender_id")
        if offender or offender_id:
            off_str = f"{offender or 'Не указан'}" + (f" [ID: {offender_id}]" if offender_id else "")
            snippets.append(f"Нарушитель: {off_str}")

        server = facts.get("server")
        if server:
            snippets.append(f"Сервер: #{server}")

        if facts.get("has_evidence"):
            snippets.append(f"Доказательства: прикреплены ({facts.get('attachments_count', 0)} файл(а))")
        else:
            snippets.append("Доказательства: не приложены")

        if evidence_report and evidence_report.get("admin_report"):
            snippets.append(f"Проверка медиа: {evidence_report['admin_report']}")

        if needs_human_reason:
            snippets.append(f"Причина передачи администрации: {needs_human_reason[:160]}")

        return "\n".join(snippets)

    def _build_response_variants(
        self,
        language: str,
        category: str,
        player_nickname: Optional[str],
        offender_nickname: Optional[str],
        top_rule: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Generate 3 ready-to-use operator response templates in the player's language."""
        lang = language if language in ("ru", "kk", "en", "uz") else "ru"
        rule_id = top_rule["id"] if top_rule else "7.2"
        player = player_nickname or ("Уважаемый игрок" if lang == "ru" else "Player")
        offender = offender_nickname or ("указанный игрок" if lang == "ru" else "reported player")

        if lang == "kk":
            return {
                "accept_investigation": (
                    f"Сәлеметсіз бе, {player}! {offender} бойынша шағымыңыз қабылданды (ереже п. {rule_id}). "
                    "Дәлелдер мен сервер логтары тексерілген соң әкімші түпкілікті шешім қабылдайды."
                ),
                "request_evidence": (
                    f"Сәлеметсіз бе, {player}! Шағымды объективті тексеру үшін п. 7.2 сәйкес /time көрінетін "
                    "скриншот немесе видеожазбаны және сервер нөмірін тіркеуіңізді сұраймыз."
                ),
                "resolved_notice": (
                    f"Сәлеметсіз бе, {player}! Сіздің өтінішіңіз әкімшілікпен қаралды, ереже п. {rule_id} бойынша "
                    "тиісті шаралар қабылданды. Grand Mobile-да жақсы ойын тілейміз!"
                ),
            }
        if lang == "uz":
            return {
                "accept_investigation": (
                    f"Assalomu alaykum, {player}! {offender} bo'yicha murojaatingiz qabul qilindi ({rule_id}-band). "
                    "Dalillar va server loglari tekshirilgach administrator yakuniy qarorni qabul qiladi."
                ),
                "request_evidence": (
                    f"Assalomu alaykum, {player}! Holatni tekshirish uchun 7.2-bandga muvofiq /time ko'ringan "
                    "skrinshot yoki videoni hamda server raqamini taqdim etishingizni so'raymiz."
                ),
                "resolved_notice": (
                    f"Assalomu alaykum, {player}! Murojaatingiz ma'muriyat tomonidan ko'rib chiqildi va {rule_id}-band "
                    "asosida тегишли чоралар ko'rildi. Yoqimli o'yin tilaymiz!"
                ),
            }
        if lang == "en":
            return {
                "accept_investigation": (
                    f"Hello, {player}! Your report regarding {offender} has been accepted for review (rule {rule_id}). "
                    "An administrator will make the final decision after verifying the evidence and server logs."
                ),
                "request_evidence": (
                    f"Hello, {player}! To investigate your report per rule 7.2, please attach a screenshot or video "
                    "with /time visible, along with the server number and player ID/nickname."
                ),
                "resolved_notice": (
                    f"Hello, {player}! Your ticket has been reviewed by an administrator and appropriate action "
                    f"has been taken in accordance with rule {rule_id}. Thank you for helping keep Grand Mobile fair!"
                ),
            }
        return {
            "accept_investigation": (
                f"Здравствуйте, {player}! Ваше обращение в отношении {offender} принято на проверку (п. {rule_id} правил). "
                "Администратор изучит приложенные доказательства и серверные логи, после чего вынесет финальное решение."
            ),
            "request_evidence": (
                f"Здравствуйте, {player}! Для объективного рассмотрения обращения согласно п. 7.2 правил просим "
                "предоставить скриншот или видеозапись с видимым временем (/time), а также уточнить сервер и ID/никнейм."
            ),
            "resolved_notice": (
                f"Здравствуйте, {player}! Ваше обращение рассмотрено администратором, материалы и логи проверены в соответствии "
                f"с п. {rule_id} правил проекта. Благодарим за обращение в службу поддержки Grand Mobile!"
            ),
        }

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
                return "Сәлем! Өтінішіңіз қабылданды және әкімшілікке берілді."
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
