"""
FastAPI backend for Grand Mobile Support Copilot.
Endpoints:
- POST /triage        : Analyze a single support ticket
- GET  /health        : Health & readiness status
- GET  /              : Modern dark-mode Operator Dashboard (Russian UI)
- GET  /api/tickets   : List of loaded dataset tickets with triage results
- GET  /api/analytics : Real-time analytics summary
- POST /api/escalate  : Escalate a ticket to a senior specialist
"""
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings
from app.schemas import (
    AdminRequestAction,
    AdminRequestCreate,
    EscalateRequest,
    FeedbackSubmit,
    HealthResponse,
    PlayerHistoryQuery,
    TriageRequest,
    TriageResponse,
)
from app.triage import TriageService
from app.workspace import LABEL_KEYS, WorkspaceStore, create_router
from app.admin_requests import admin_request_store, STATUS_WAITING
from app.fraud_detector import fraud_detector
from app.player_history import player_history_store
from app.feedback import feedback_store

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grand_mobile_copilot.api")

app = FastAPI(
    title="Grand Mobile Support Copilot API",
    description="AI Copilot for player support triage, rule-grounded RAG, and operator draft generation (Decentrathon 6.0).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

triage_service = TriageService()
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Operator workspace (roles, ticket lifecycle, audit). DEMO MODE: identity via X-Demo-User header.
_state_env = os.getenv("WORKSPACE_STATE_PATH", "data/workspace_state.json").strip()
_state_path = None
if _state_env:
    _state_path = Path(_state_env)
    if not _state_path.is_absolute():
        _state_path = settings.root_dir / _state_path
workspace_store = WorkspaceStore(triage_service, settings.valid_data_path, _state_path)
app.include_router(create_router(workspace_store))

# Store preloaded queue items for Operator UI
_ui_tickets_store: List[Dict[str, Any]] = []


def _ensure_ui_queue_loaded() -> None:
    global _ui_tickets_store
    if _ui_tickets_store:
        return
    valid_path = settings.valid_data_path
    if not valid_path.exists():
        return
    try:
        lines = valid_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines[:25]):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            req = TriageRequest(**{k: v for k, v in raw.items() if k not in LABEL_KEYS})
            res = triage_service.analyze(req, record_cache=False)
            _ui_tickets_store.append(
                {
                    "id": res.ticket_id or f"tkt_{idx+1:04d}",
                    "kind": req.kind or "complaint",
                    "is_synthetic_demo": bool(raw.get("is_synthetic_demo", settings.is_using_synthetic_valid)),
                    "messages": [m.dict() for m in req.messages],
                    "triage": res.dict(),
                    "escalated_to_senior": False,
                }
            )
    except Exception as exc:
        logger.warning("Could not preload UI tickets queue: %s", exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured JSON on invalid or empty input requests (HTTP 422)."""
    logger.warning("validation_error path=%s errors_count=%d", request.url.path, len(exc.errors()))
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": (
                "Некорректный или пустой запрос. Проверьте поле 'messages'."
                if request.url.path == "/triage"
                else "Некорректные данные запроса."
            ),
            "details": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Structured error handler without leaking internal stack traces or PII."""
    logger.error("internal_error path=%s error_type=%s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Внутренняя ошибка сервиса при обработке обращения.",
            "type": type(exc).__name__,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and readiness check."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        rules_loaded=len(triage_service.rag.rules_by_id),
        rule_chunks_indexed=len(triage_service.rag.fragments),
        categories_loaded=list(triage_service.taxonomy.valid_categories),
        priorities_loaded=list(triage_service.taxonomy.valid_priorities),
        classifier_trained=triage_service.classifier.is_trained,
        using_synthetic_data=settings.is_using_synthetic_train,
        llm_enabled=bool(triage_service.generator.llm_client and triage_service.generator.llm_client.enabled),
    )


@app.post("/triage", response_model=TriageResponse)
async def triage_ticket(payload: TriageRequest) -> TriageResponse:
    """
    Analyze a single player ticket:
    - Detect language ('ru', 'kk', 'uz', 'en', 'und')
    - Classify category & priority strictly from `taxonomy.json`
    - Extract structured facts
    - Retrieve & verify supporting rule fragments from `rules.json`
    - Generate safe operator draft response in the player's language
    - Determine `needs_human` and `confidence`
    """
    start = time.perf_counter()
    result = triage_service.analyze(payload)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    logger.info(
        "api_post_triage status=200 category=%s priority=%s needs_human=%s duration_ms=%.2f",
        result.category,
        result.priority,
        result.needs_human,
        elapsed_ms,
    )
    return result


@app.get("/api/tickets")
async def list_ui_tickets() -> Dict[str, Any]:
    """Return preloaded dataset tickets and their triage results for the Operator UI."""
    _ensure_ui_queue_loaded()
    return {
        "using_synthetic_data": settings.is_using_synthetic_valid,
        "count": len(_ui_tickets_store),
        "tickets": _ui_tickets_store,
    }


@app.get("/api/analytics")
async def get_analytics() -> Dict[str, Any]:
    """Return real-time processing analytics for the Operator Dashboard."""
    _ensure_ui_queue_loaded()
    return triage_service.get_analytics_summary()


@app.post("/api/escalate")
async def escalate_to_senior(req: EscalateRequest) -> Dict[str, Any]:
    """Mark a ticket as escalated to a senior specialist."""
    _ensure_ui_queue_loaded()
    triage_service.analytics["senior_escalated_ids"].add(req.ticket_id)
    for item in _ui_tickets_store:
        if item["id"] == req.ticket_id:
            item["escalated_to_senior"] = True
            item["triage"]["needs_human"] = True
            item["triage"]["needs_human_reason"] = req.reason or "Передано старшему специалисту оператором"
            break
    logger.info("ticket_escalated_to_senior ticket_id=%s", req.ticket_id)
    return {
        "status": "escalated",
        "ticket_id": req.ticket_id,
        "reason": req.reason,
        "senior_escalated_count": len(triage_service.analytics["senior_escalated_ids"]),
    }


@app.get("/", response_class=HTMLResponse)
async def operator_dashboard() -> HTMLResponse:
    """Serve the Russian dark-mode Operator Support Copilot interface."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Operator UI file not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ============================================================
# Admin Requests — live admin call management
# ============================================================

@app.post("/api/admin-requests", tags=["Admin Requests"])
async def create_admin_request(payload: AdminRequestCreate) -> Dict[str, Any]:
    """
    Player explicitly requests a live human admin.
    Creates an admin request entry with WAITING status.
    """
    ar = admin_request_store.create(
        player_nickname=payload.player_nickname,
        server=payload.server,
        context_messages=payload.context_messages,
        ticket_id=payload.ticket_id,
        language=payload.language,
    )
    logger.info("admin_request_created_api id=%s player=%s", ar["id"], payload.player_nickname)
    return {"status": "created", "admin_request": ar}


@app.get("/api/admin-requests", tags=["Admin Requests"])
async def list_admin_requests(
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List all admin requests. Optionally filter by status."""
    requests = admin_request_store.list_requests(status=status, limit=limit)
    return {
        "count": len(requests),
        "waiting_count": admin_request_store.get_waiting_count(),
        "stats": admin_request_store.get_stats(),
        "requests": requests,
    }


@app.post("/api/admin-requests/{request_id}/accept", tags=["Admin Requests"])
async def accept_admin_request(request_id: str, action: AdminRequestAction) -> Dict[str, Any]:
    """Admin accepts a waiting admin request."""
    result = admin_request_store.accept(
        request_id=request_id,
        admin_id=action.admin_id,
        admin_name=action.admin_name,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Admin request {request_id!r} not found")
    return {"status": "accepted", "admin_request": result}


@app.post("/api/admin-requests/{request_id}/complete", tags=["Admin Requests"])
async def complete_admin_request(request_id: str, action: AdminRequestAction) -> Dict[str, Any]:
    """Admin marks a request as completed."""
    result = admin_request_store.complete(
        request_id=request_id,
        admin_id=action.admin_id,
        admin_name=action.admin_name,
        note=action.note,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Admin request {request_id!r} not found")
    return {"status": "completed", "admin_request": result}


@app.post("/api/admin-requests/{request_id}/cancel", tags=["Admin Requests"])
async def cancel_admin_request(request_id: str) -> Dict[str, Any]:
    """Cancel an admin request (by player)."""
    result = admin_request_store.cancel(request_id=request_id, requester="player")
    if not result:
        raise HTTPException(status_code=404, detail=f"Admin request {request_id!r} not found")
    return {"status": "cancelled", "admin_request": result}


# ============================================================
# Fraud Detection — statistics and reporting
# ============================================================

@app.get("/api/fraud/stats", tags=["Fraud Detection"])
async def get_fraud_stats() -> Dict[str, Any]:
    """Return fraud detection statistics and top suspicious reporters/offenders."""
    return {
        "stats": fraud_detector.get_statistics(),
        "top_reporters": fraud_detector.get_top_reporters(limit=5),
        "top_offenders": fraud_detector.get_top_offenders(limit=5),
        "disclaimer": "Данные носят аналитический характер. Решения принимает администратор.",
    }


# ============================================================
# Player History — offender profile cards
# ============================================================

@app.get("/api/player-history/{nickname}", tags=["Player History"])
async def get_player_history(nickname: str) -> Dict[str, Any]:
    """Get violation history card for a player nickname."""
    history = player_history_store.get_player_history(nickname)
    if not history:
        return {"found": False, "nickname": nickname}
    return history


@app.get("/api/repeat-offenders", tags=["Player History"])
async def get_repeat_offenders(min_complaints: int = 3, limit: int = 10) -> Dict[str, Any]:
    """Get list of repeat offenders for admin dashboard."""
    offenders = player_history_store.get_repeat_offenders(min_complaints=min_complaints, limit=limit)
    return {
        "count": len(offenders),
        "offenders": offenders,
        "disclaimer": "Жалобы не являются доказательством вины. Показаны только подтверждённые нарушения.",
    }


# ============================================================
# Feedback Loop — admin corrections for AI quality
# ============================================================

@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(payload: FeedbackSubmit) -> Dict[str, Any]:
    """Submit admin feedback on an AI triage decision for quality improvement."""
    entry = feedback_store.record(
        ticket_id=payload.ticket_id,
        admin_id=payload.admin_id,
        admin_name=payload.admin_name,
        feedback_type=payload.feedback_type,
        ai_category=payload.ai_category,
        correct_category=payload.correct_category,
        ai_priority=payload.ai_priority,
        correct_priority=payload.correct_priority,
        ai_rule_ids=payload.ai_rule_ids,
        correct_rule_ids=payload.correct_rule_ids,
        note=payload.note,
    )
    logger.info("feedback_submitted ticket=%s type=%s admin=%s", payload.ticket_id, payload.feedback_type, payload.admin_name)
    return {"status": "recorded", "feedback_id": entry["id"], "feedback": entry}


@app.get("/api/feedback/stats", tags=["Feedback"])
async def get_feedback_stats() -> Dict[str, Any]:
    """Return AI feedback quality statistics including accuracy rate."""
    return feedback_store.get_stats()


@app.get("/api/feedback", tags=["Feedback"])
async def list_feedback(limit: int = 50) -> Dict[str, Any]:
    """List all recorded feedback entries."""
    items = feedback_store.list_all(limit=limit)
    return {"count": len(items), "feedback": items}

