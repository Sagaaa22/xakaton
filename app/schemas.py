"""Pydantic schemas for Grand Mobile Support Copilot (compatible with Pydantic v1 & v2)."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class MessageItem(BaseModel):
    role: str = Field(default="player", description="Sender role: player, admin, support, etc.")
    text: str = Field(..., description="Message text content")
    attachments: Union[int, List[Any], bool, None] = Field(
        default=0, description="Number of attachments or list of attachments"
    )

    @validator("role", pre=True, always=True)
    def normalize_role(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return "player"
        return v.strip().lower()

    @validator("text", pre=True)
    def validate_text_type(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Message text cannot be null")
        if not isinstance(v, str):
            raise ValueError("Message text must be a string")
        return v

    @property
    def attachments_count(self) -> int:
        if isinstance(self.attachments, bool):
            return 1 if self.attachments else 0
        if isinstance(self.attachments, int):
            return max(0, self.attachments)
        if isinstance(self.attachments, list):
            return len(self.attachments)
        return 0


class TriageRequest(BaseModel):
    id: Optional[Union[str, int]] = Field(default=None, description="Optional ticket identifier")
    ticket_id: Optional[Union[str, int]] = Field(default=None, description="Optional ticket ID alias")
    kind: Optional[str] = Field(default="complaint", description="Ticket kind: complaint, question, appeal, bug")
    messages: List[MessageItem] = Field(..., description="Non-empty list of messages in the ticket")
    server: Optional[Union[int, str]] = Field(default=None, description="Optional game server ID/number")
    nickname: Optional[str] = Field(default=None, description="Optional player nickname")
    offender_nickname: Optional[str] = Field(default=None, description="Optional reported player nickname")
    player_id: Optional[int] = Field(default=None, description="Optional reporter in-game ID")
    offender_id: Optional[int] = Field(default=None, description="Optional offender in-game ID")
    evidence_files: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional list of media attachments (JPG, PNG, MP4, WEBM) or OCR metadata"
    )
    regenerate: Optional[bool] = Field(default=False, description="Force fresh regeneration bypassing duplicate cache")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Any additional metadata")

    class Config:
        extra = "allow"

    @validator("messages")
    def validate_messages_non_empty(cls, v: List[MessageItem]) -> List[MessageItem]:
        if not v:
            raise ValueError("messages list cannot be empty")
        player_texts = [m.text.strip() for m in v if m.text and m.text.strip()]
        if not player_texts:
            raise ValueError("At least one message must contain non-empty text")
        return v


class ExtractedFacts(BaseModel):
    nicknames: List[str] = Field(default_factory=list)
    offender_nickname: Optional[str] = None
    reporter_nickname: Optional[str] = None
    player_ids: List[int] = Field(default_factory=list, description="Extracted in-game player IDs")
    offender_id: Optional[int] = None
    reporter_id: Optional[int] = None
    mentioned_rules: List[str] = Field(default_factory=list)
    valid_mentioned_rules: List[str] = Field(default_factory=list)
    unknown_mentioned_rules: List[str] = Field(default_factory=list)
    has_evidence: bool = False
    attachments_count: int = 0
    evidence_types: List[str] = Field(default_factory=list)
    server: Optional[Union[int, str]] = None
    amounts: List[str] = Field(default_factory=list)
    commands_mentioned: List[str] = Field(default_factory=list)
    is_mixed_language: bool = False
    detected_languages: List[str] = Field(default_factory=list)
    slang_or_typos_normalized: List[str] = Field(default_factory=list)
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    @validator("server", always=True)
    def normalize_server_int(cls, v: Any) -> Optional[Union[int, str]]:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return v


class CitedRule(BaseModel):
    id: str
    section: str = ""
    title: str
    quote: str
    punishment: str = ""
    relevance_score: float = 0.0


class TriageResponse(BaseModel):
    ticket_id: Optional[str] = None
    language: str = Field(..., description="Detected language: ru, kk, uz, en, or und")
    category: str = Field(..., description="Category ID strictly from taxonomy.json")
    category_name: str = Field(default="", description="Human-readable category name")
    subcategory: Optional[str] = Field(default=None, description="Specific violation subtype (cheats, dm_rdm, pg, mg, trade_scam, etc.)")
    subcategory_name: Optional[str] = Field(default=None, description="Human-readable subcategory name")
    priority: str = Field(..., description="Priority ID strictly from taxonomy.json")
    is_spam: bool = False
    is_duplicate: bool = False
    facts: ExtractedFacts
    draft_response: str = Field(..., description="Operator draft response in player language")
    rule_ids: List[str] = Field(default_factory=list, description="Verified rule IDs from rules.json")
    cited_rules: List[CitedRule] = Field(default_factory=list, description="Retrieved rule fragments with quotes")
    needs_human: bool = Field(..., description="Whether human moderator review is required")
    needs_human_reason: Optional[str] = Field(default=None, description="Explanation for human review requirement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Triage confidence score [0.0, 1.0]")
    confidence_pct: int = Field(default=0, ge=0, le=100, description="Confidence percentage 0-100%")
    auto_handled: bool = Field(default=False, description="True if simple & safe ticket was handled automatically by AI")
    routing_decision: str = Field(default="human_admin", description="Routing target: auto_handled | junior_admin | senior_admin")
    reasons: List[str] = Field(default_factory=list, description="Reasoning steps and classification triggers")
    processing_time_ms: float = 0.0
    cost_usd: float = 0.0
    # --- AI & Copilot feature fields ---
    intent: Optional[str] = Field(default=None, description="Detected player intent")
    intent_label: Optional[str] = Field(default=None, description="Human-readable intent label")
    intent_confidence: Optional[float] = Field(default=None, description="Intent detection confidence")
    requires_admin: Optional[bool] = Field(default=None, description="Whether intent requires live admin")
    fraud_risk_score: int = Field(default=0, ge=0, le=100, description="Fraud/fake report risk score 0-100")
    fraud_risk_level: Optional[str] = Field(default=None, description="Risk level: minimal/low/medium/high")
    fraud_reasons: List[str] = Field(default_factory=list, description="Fraud risk reasons")
    fraud_related_tickets: List[Dict[str, Any]] = Field(default_factory=list, description="Related tickets for fraud check")
    admin_request_id: Optional[str] = Field(default=None, description="Admin call request ID if player asked for admin")
    admin_request_status: Optional[str] = Field(default=None, description="Status of live admin request (WAITING/ACCEPTED/etc.)")
    assigned_admin: Optional[Dict[str, Any]] = Field(default=None, description="Distributed admin assigned to review")
    telegram_notification: Optional[Dict[str, Any]] = Field(default=None, description="Telegram notification details if sent")
    player_history: Optional[Dict[str, Any]] = Field(default=None, description="Offender history card")
    evidence_report: Optional[Dict[str, Any]] = Field(default=None, description="Evidence & OCR analysis report")
    ai_summary: Optional[Dict[str, Any]] = Field(default=None, description="Admin Copilot summary for operator")


class EscalateRequest(BaseModel):
    ticket_id: str
    reason: Optional[str] = "Передано старшему специалисту оператором"
    operator_note: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    rules_loaded: int
    rule_chunks_indexed: int
    categories_loaded: List[str]
    priorities_loaded: List[str]
    classifier_trained: bool
    using_synthetic_data: bool
    llm_enabled: bool
    telegram_enabled: bool = True
    game_api_connected: bool = False
    grand_mobile_api_enabled: bool = False
    ocr_available: bool = True


# ---- Admin Request Schemas ----

class AdminRequestCreate(BaseModel):
    """Player explicitly requests a live admin."""
    ticket_id: Optional[str] = None
    player_nickname: Optional[str] = None
    player_id: Optional[int] = None
    server: Optional[Union[int, str]] = None
    language: str = "ru"
    context_messages: List[Dict[str, Any]] = Field(default_factory=list)


class AdminRequestAction(BaseModel):
    """Admin takes action on an admin request."""
    request_id: str
    admin_id: str = Field(..., description="Admin's user ID")
    admin_name: str = Field(..., description="Admin's display name")
    note: Optional[str] = None


# ---- Feedback Schemas ----

class FeedbackSubmit(BaseModel):
    ticket_id: str
    admin_id: str
    admin_name: str
    feedback_type: str = Field(
        default="correct_classification",
        description="One of: correct_classification, wrong_category, wrong_priority, wrong_rule, wrong_evidence_analysis, wrong_confidence, intent_mismatch"
    )
    ai_category: str
    correct_category: Optional[str] = None
    ai_priority: str
    correct_priority: Optional[str] = None
    ai_rule_ids: List[str] = Field(default_factory=list)
    correct_rule_ids: Optional[List[str]] = None
    ticket_text: Optional[str] = None
    apply_to_model: bool = Field(default=True, description="Update classifier weights when correction is submitted")
    note: Optional[str] = None


# ---- Player History Schema ----

class PlayerHistoryQuery(BaseModel):
    nickname: Optional[str] = None
    player_id: Optional[int] = None
    include_demo: bool = True


# ---- Evidence Analyze Schema ----

class EvidenceAnalyzeRequest(BaseModel):
    ticket_id: Optional[str] = None
    complaint_text: str = ""
    category: str = "insult_toxicity"
    offender_nickname: Optional[str] = None
    reporter_nickname: Optional[str] = None
    offender_id: Optional[int] = None
    server: Optional[int] = None
    files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of file items: {filename, base64_data, ocr_text, metadata}"
    )


# ---- Telegram Callback Schema ----

class TelegramCallbackRequest(BaseModel):
    callback_data: str = Field(..., description="e.g. accept_ticket:tkt_0001 or accept_ar:ar_1234")
    admin_id: Optional[str] = None
    admin_name: Optional[str] = None
