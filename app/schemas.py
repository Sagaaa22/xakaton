"""Pydantic schemas for Grand Mobile Support Copilot (compatible with Pydantic v1 & v2)."""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class MessageItem(BaseModel):
    role: str = Field(default="player", description="Sender role: player, reported_player, admin, etc.")
    text: str = Field(..., description="Message text content")
    ts: Optional[str] = Field(default=None, description="Message timestamp if available")
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
    ticket_id: Optional[Union[str, int]] = Field(default=None, description="Ticket ID from dataset")
    kind: Optional[str] = Field(default="complaint", description="Ticket kind: complaint or report")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    region: Optional[str] = Field(default=None, description="Server region: cis, kz, uz, etc.")
    server_lang: Optional[str] = Field(default=None, description="Server language: ru, kk, uz, etc.")
    lang: Optional[str] = Field(default=None, description="Heuristic language hint from dataset")
    reporter_hash: Optional[str] = Field(default=None, description="Anonymized reporter hash")
    reported_hash: Optional[str] = Field(default=None, description="Anonymized reported player hash")
    reported_nick: Optional[str] = Field(default=None, description="Reported player nickname")
    rule_refs_raw: Optional[Union[str, List[str]]] = Field(default=None, description="Raw rule reference entered by player")
    images_count: Optional[int] = Field(default=None, description="Number of attached screenshots")
    status: Optional[Union[int, str]] = Field(default=None, description="Ticket status code")
    messages: List[MessageItem] = Field(..., description="Non-empty list of messages in the ticket")
    server: Optional[Union[int, str]] = Field(default=None, description="Optional game server ID/number")
    nickname: Optional[str] = Field(default=None, description="Optional player nickname")
    offender_nickname: Optional[str] = Field(default=None, description="Optional reported player nickname")
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
    mentioned_rules: List[str] = Field(default_factory=list)
    valid_mentioned_rules: List[str] = Field(default_factory=list)
    unknown_mentioned_rules: List[str] = Field(default_factory=list)
    rule_refs_raw: Optional[str] = None
    has_evidence: bool = False
    attachments_count: int = 0
    evidence_types: List[str] = Field(default_factory=list)
    server: Optional[Union[int, str]] = None
    region: Optional[str] = None
    amounts: List[str] = Field(default_factory=list)
    commands_mentioned: List[str] = Field(default_factory=list)
    is_mixed_language: bool = False
    detected_languages: List[str] = Field(default_factory=list)
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
    category: str = Field(..., description="Category code strictly from taxonomy.json")
    category_name: str = Field(default="", description="Human-readable category title")
    priority: str = Field(..., description="Priority code strictly from taxonomy.json")
    is_spam: bool = False
    is_duplicate: bool = False
    facts: ExtractedFacts
    draft_response: str = Field(..., description="Operator draft response in player language")
    rule_ids: List[str] = Field(default_factory=list, description="Verified rule IDs from rules.json")
    cited_rules: List[CitedRule] = Field(default_factory=list, description="Retrieved rule fragments with quotes")
    needs_human: bool = Field(..., description="Whether human moderator review is required")
    needs_human_reason: Optional[str] = Field(default=None, description="Explanation for human review requirement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Triage confidence score [0.0, 1.0]")
    reasons: List[str] = Field(default_factory=list, description="Reasoning steps and triggers")
    processing_time_ms: float = 0.0
    cost_usd: float = 0.0


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
