"""
Safety guardrails for Grand Mobile Support Copilot draft responses.
Ensures drafts NEVER:
1. Promise an unban or money refund;
2. Guarantee punishment of another player;
3. Present an accusation as a proven fact;
4. Invent rule IDs or rule content;
5. Make a final punishment decision.
"""
import re
from typing import List, Set, Tuple


FORBIDDEN_PATTERNS = [
    # 1. Promising unban
    (
        "promise_unban",
        re.compile(
            r"(?:мы\s+вас\s+разбаним|вы\s+будете\s+разбанены|аккаунт\s+разбанен|блокировка\s+снята|"
            r"гарантируем\s+разбан|сейчас\s+сниму\s+бан|разбан\s+одобрен|"
            r"we\s+will\s+unban\s+you|you\s+are\s+unbanned|ban\s+has\s+been\s+removed|"
            r"сізді\s+баннан\s+шығарамыз|sizni\s+bandan\s+chiqaramiz)",
            re.IGNORECASE,
        ),
    ),
    # 2. Promising money refund
    (
        "promise_refund",
        re.compile(
            r"(?:мы\s+вернем\s+(?:вам\s+)?деньги|возврат\s+средств\s+оформлен|деньги\s+будут\s+возвращены|"
            r"гарантируем\s+возврат|сейчас\s+верну\s+донат|средства\s+возвращены|"
            r"we\s+will\s+refund\s+your\s+money|refund\s+is\s+approved|money\s+has\s+been\s+refunded|"
            r"ақшаңызды\s+қайтарамыз|pulingizni\s+qaytaramiz)",
            re.IGNORECASE,
        ),
    ),
    # 3. Guaranteeing punishment of another player
    (
        "guarantee_punishment",
        re.compile(
            r"(?:игрок\s+будет\s+(?:забанен|наказан|заблокирован)|нарушитель\s+получит\s+бан|"
            r"мы\s+(?:забаним|накажем)\s+игрока|игрок\s+уже\s+наказан|выдал\s+бан\s+игроку|"
            r"player\s+will\s+be\s+(?:banned|punished)|we\s+have\s+banned\s+the\s+player|"
            r"ойыншы\s+жазаланады|o'yinchi\s+jazolanadi)",
            re.IGNORECASE,
        ),
    ),
    # 4. Presenting accusation as proven fact
    (
        "accusation_as_fact",
        re.compile(
            r"(?:игрок\s+действительно\s+нарушил|факт\s+нарушения\s+полностью\s+доказан|"
            r"вина\s+игрока\s+доказана|подтверждаем\s+что\s+игрок\s+виновен|"
            r"it\s+is\s+proven\s+that\s+the\s+player\s+violated)",
            re.IGNORECASE,
        ),
    ),
    # 5. Making final punishment decision
    (
        "final_punishment_decision",
        re.compile(
            r"(?:окончательное\s+решение\s*:\s*(?:бан|мут|деморган|отказ)|"
            r"я\s+принимаю\s+решение\s+забанить|выношу\s+вердикт\s*:)",
            re.IGNORECASE,
        ),
    ),
]


class SafetyGuard:
    """Validates draft responses against all safety & rule-veracity constraints."""

    @staticmethod
    def validate_draft(
        draft_text: str,
        valid_rule_ids: Set[str],
        unknown_user_rules: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check if `draft_text` complies with all safety policies and only references real rules.
        Returns (is_safe, list_of_violation_codes).
        """
        violations: List[str] = []
        if not draft_text or not draft_text.strip():
            return False, ["empty_draft"]

        for code, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(draft_text):
                violations.append(code)

        # Verify every rule ID mentioned in the draft exists in `valid_rule_ids`
        # (unless the draft is explicitly stating that a user-mentioned rule ID was not found)
        cited_in_draft = re.findall(
            r"(?:п\.|пункт\w*|правил\w*|rule|band\w*)\s*№?\s*(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)|(\d{1,3}\.\d{1,3})\s*-?\s*band",
            draft_text.lower(),
        )
        for g1, g2 in cited_in_draft:
            rid = (g1 or g2).strip()
            if not rid:
                continue
            if rid not in valid_rule_ids and rid not in unknown_user_rules:
                violations.append(f"hallucinated_rule_id:{rid}")

        return len(violations) == 0, violations

