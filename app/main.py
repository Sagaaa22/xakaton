"""
FastAPI backend for Grand Mobile Support Copilot.
Endpoints:
- POST /triage                      : Analyze a single support ticket
- GET  /health                      : Health & readiness status
- GET  /                            : Modern dark-mode Operator Dashboard (Russian UI)
- GET  /api/tickets                 : List of loaded dataset tickets with triage results
- GET  /api/analytics               : Real-time analytics summary
- POST /api/escalate                : Escalate a ticket to a senior specialist
- /api/admin-requests/*             : Direct player calls for a live human admin
- /api/telegram/*                   : Telegram admin notifications, inline buttons, reminders & distribution
- /api/evidence/*                   : Evidence analysis (JPG, PNG, MP4, WEBM, OCR, consistency check)
- /api/fraud/*                      : Anti-fraud & contradiction detection
- /api/player-history/*             : Repeat offender lookup by In-Game ID and Nickname
- /api/feedback/*                   : Admin Feedback Loop for continuous AI improvement
"""
import base64
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings
from app.schemas import (
    AdminRequestAction,
    AdminRequestCreate,
    EscalateRequest,
    EvidenceAnalyzeRequest,
    FeedbackSubmit,
    HealthResponse,
    PlayerHistoryQuery,
    TelegramCallbackRequest,
    TriageRequest,
    TriageResponse,
)
from app.triage import TriageService
from app.workspace import DEMO_USERS, LABEL_KEYS, WorkspaceStore, _player_text, create_router
from app.admin_requests import admin_request_store, STATUS_WAITING
from app.fraud_detector import fraud_detector
from app.player_history import player_history_store
from app.feedback import feedback_store
from app.telegram_notifier import telegram_notifier
from app.evidence_analyzer import evidence_analyzer

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grand_mobile_copilot.api")

app = FastAPI(
    title="Grand Mobile Support Copilot API",
    description="AI Copilot for player support triage, rule-grounded RAG, evidence analysis, Telegram admin alerts, and operator workspace (Decentrathon 6.0).",
    version="1.1.0",
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


# Wire Telegram inline button callbacks to WorkspaceStore and AdminRequestStore
def _on_tg_accept_ticket(ticket_id: str, admin_id: str, admin_name: str) -> None:
    with workspace_store.lock:
        t = workspace_store.tickets.get(ticket_id)
        if t and t["status"] in ("new", "returned", "escalated"):
            resolved_id = admin_id if admin_id in DEMO_USERS else "junior_1"
            t["assignee"] = resolved_id
            if t["status"] == "new":
                t["status"] = "in_progress"
            workspace_store.touch(t)
            actor = DEMO_USERS.get(resolved_id, {"id": admin_id, "name": admin_name, "role": "junior_admin"})
            workspace_store._log(actor, "take_via_telegram", ticket_id, {"telegram_admin": admin_name})
            workspace_store._save()


def _on_tg_accept_admin_request(request_id: str, admin_id: str, admin_name: str) -> None:
    admin_request_store.accept(request_id=request_id, admin_id=admin_id, admin_name=admin_name)


telegram_notifier.register_on_accept_ticket(_on_tg_accept_ticket)
telegram_notifier.register_on_accept_admin_request(_on_tg_accept_admin_request)

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
        version="1.1.0",
        rules_loaded=len(triage_service.rag.rules_by_id),
        rule_chunks_indexed=len(triage_service.rag.fragments),
        categories_loaded=list(triage_service.taxonomy.valid_categories),
        priorities_loaded=list(triage_service.taxonomy.valid_priorities),
        classifier_trained=triage_service.classifier.is_trained,
        using_synthetic_data=settings.is_using_synthetic_train,
        llm_enabled=bool(triage_service.generator.llm_client and triage_service.generator.llm_client.enabled),
        telegram_enabled=bool(settings.telegram_enabled),
        grand_mobile_api_enabled=bool(settings.grand_mobile_api_enabled),
        ocr_available=True,
    )


@app.post("/triage", response_model=TriageResponse)
async def triage_ticket(payload: TriageRequest) -> TriageResponse:
    """
    Analyze a single player ticket:
    - Detect language ('ru', 'kk', 'uz', 'en', 'und') & normalize gaming slang/typos
    - Classify category, subcategory & priority strictly from `taxonomy.json`
    - Extract structured facts (nicknames, in-game IDs, server, /time, media)
    - Analyze attached evidence (JPG, PNG, MP4, WEBM) & cross-check with text
    - Retrieve & verify supporting rule fragments from `rules.json`
    - Generate safe operator draft response in the player's language
    - Determine `needs_human`, `confidence`, `auto_handled`, and send Telegram alert if needed
    """
    start = time.perf_counter()
    result = triage_service.analyze(payload)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    logger.info(
        "api_post_triage status=200 category=%s subcategory=%s priority=%s needs_human=%s duration_ms=%.2f",
        result.category,
        result.subcategory,
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
    """Mark a ticket as escalated to a senior specialist and send Telegram alert."""
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
# 1. Telegram Admin Notifications, Callbacks, Reminders & Distribution
# ============================================================

@app.get("/api/telegram/status", tags=["Telegram Notifications"])
async def get_telegram_status() -> Dict[str, Any]:
    """Return Telegram Bot connection status, admin workload distribution, and notification counts."""
    return telegram_notifier.get_status()


@app.get("/api/telegram/notifications", tags=["Telegram Notifications"])
async def list_telegram_notifications(
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List Telegram notifications sent to administrators."""
    items = telegram_notifier.list_notifications(status=status, limit=limit)
    return {
        "count": len(items),
        "status": telegram_notifier.get_status(),
        "notifications": items,
    }


@app.post("/api/telegram/callback", tags=["Telegram Notifications"])
async def handle_telegram_callback(payload: TelegramCallbackRequest) -> Dict[str, Any]:
    """
    Process inline button click («✅ Принять жалобу», «🔗 Открыть жалобу»)
    from either Telegram webhook or Dashboard interactive feed.
    """
    res = telegram_notifier.handle_callback(
        callback_data=payload.callback_data,
        admin_id=payload.admin_id,
        admin_name=payload.admin_name,
    )
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("error", "Callback target not found"))
    return res


@app.post("/api/telegram/webhook", tags=["Telegram Notifications"])
async def telegram_bot_webhook(request: Request) -> Dict[str, Any]:
    """Standard Telegram Bot API webhook receiver for callback_query updates."""
    data = await request.json()
    cb = data.get("callback_query")
    if not cb:
        return {"ok": True, "ignored": True}
    cb_data = cb.get("data", "")
    user = cb.get("from", {})
    admin_id = str(user.get("id") or "senior_1")
    admin_name = user.get("first_name") or user.get("username") or "Telegram Admin"
    return telegram_notifier.handle_callback(
        callback_data=cb_data,
        admin_id=admin_id,
        admin_name=admin_name,
    )


@app.post("/api/telegram/check-reminders", tags=["Telegram Notifications"])
async def trigger_telegram_reminders(
    max_age_minutes: Optional[int] = Query(None, description="Override SLA minutes (use 0 for immediate test)"),
) -> Dict[str, Any]:
    """Check unaccepted complaints and send SLA reminder notifications to admins."""
    reminders = telegram_notifier.check_and_send_reminders(max_age_minutes=max_age_minutes)
    return {
        "ok": True,
        "reminders_sent": len(reminders),
        "reminders": reminders,
    }


@app.post("/api/telegram/notify/{ticket_id}", tags=["Telegram Notifications"])
async def send_ticket_telegram_notification(
    ticket_id: str,
    preferred_admin_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Manually send or re-send a ticket notification to Telegram with smart admin distribution."""
    with workspace_store.lock:
        t = workspace_store.tickets.get(ticket_id)
        if not t:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id!r} not found")
        tr = t.get("triage", {})
        facts = tr.get("facts", {})
        ai_sum = tr.get("ai_summary", {})
        notif = telegram_notifier.notify_ticket_escalation(
            ticket_id=ticket_id,
            player_nickname=facts.get("reporter_nickname"),
            player_id=facts.get("reporter_id"),
            offender_nickname=facts.get("offender_nickname"),
            offender_id=facts.get("offender_id"),
            server=facts.get("server"),
            category=tr.get("category", "rp_violation"),
            category_name=tr.get("category_name", "Жалоба"),
            subcategory=tr.get("subcategory_name"),
            priority=tr.get("priority", "high"),
            confidence=float(tr.get("confidence") or 0.75),
            ai_summary_text=ai_sum.get("summary", _player_text(t)[:200]),
            needs_human_reason=tr.get("needs_human_reason"),
            rule_ids=tr.get("rule_ids", []),
            fraud_risk_score=int(tr.get("fraud_risk_score") or 0),
            preferred_admin_id=preferred_admin_id,
        )
    return {"ok": True, "notification": notif}


# ============================================================
# 3. Direct Admin Requests — live admin call management
# ============================================================

@app.post("/api/admin-requests", tags=["Admin Requests"])
async def create_admin_request(payload: AdminRequestCreate) -> Dict[str, Any]:
    """
    Player explicitly requests a live human admin.
    Creates an admin request entry with WAITING status and notifies admins in Telegram.
    """
    ar = admin_request_store.create(
        player_nickname=payload.player_nickname,
        server=payload.server,
        context_messages=payload.context_messages,
        ticket_id=payload.ticket_id,
        language=payload.language,
    )
    tg_notif = telegram_notifier.notify_direct_admin_call(ar)
    logger.info("admin_request_created_api id=%s player=%s", ar["id"], payload.player_nickname)
    return {"status": "created", "admin_request": ar, "telegram_notification": tg_notif}


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
    telegram_notifier.handle_callback(
        callback_data=f"accept_ar:{request_id}",
        admin_id=action.admin_id,
        admin_name=action.admin_name,
    )
    return {"status": "accepted", "admin_request": result}


@app.post("/api/admin-requests/{request_id}/start", tags=["Admin Requests"])
async def start_admin_request(request_id: str, action: AdminRequestAction) -> Dict[str, Any]:
    """Admin connects to the player (status -> IN_PROGRESS)."""
    result = admin_request_store.start(
        request_id=request_id,
        admin_id=action.admin_id,
        admin_name=action.admin_name,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Admin request {request_id!r} not found")
    return {"status": "in_progress", "admin_request": result}


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
# 8. Evidence Analysis (JPG, PNG, MP4, WEBM, OCR, Consistency Check)
# ============================================================

@app.post("/api/evidence/analyze", tags=["Evidence Analysis"])
async def analyze_evidence_endpoint(payload: EvidenceAnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze attached evidence files (JPG, PNG, MP4, WEBM):
    - Extract OCR text from screenshots and frame checkpoints from videos
    - Extract in-game IDs, nicknames, server numbers, /time timestamps, amounts
    - Compare against complaint text / ticket to detect contradictions
    - Generate a concise report for administration
    """
    complaint_text = payload.complaint_text or ""
    category = payload.category or "rp_violation"
    facts_dict: Dict[str, Any] = {}

    if payload.ticket_id:
        with workspace_store.lock:
            t = workspace_store.tickets.get(payload.ticket_id)
            if t:
                if not complaint_text:
                    complaint_text = _player_text(t)
                tr = t.get("triage", {})
                category = payload.category or tr.get("category", "rp_violation")
                facts_dict = dict(tr.get("facts", {}))

    if not facts_dict and complaint_text:
        _, extracted = triage_service.extractor.extract_facts(
            TriageRequest(kind="complaint", messages=[{"role": "player", "text": complaint_text}]),
            valid_rule_ids=triage_service.rag.valid_rule_ids,
        )
        facts_dict = extracted.dict()

    if payload.offender_nickname:
        facts_dict["offender_nickname"] = payload.offender_nickname
    if payload.offender_id is not None:
        facts_dict["offender_id"] = payload.offender_id
    if payload.server is not None:
        facts_dict["server"] = payload.server

    files_dicts = [f.dict() if hasattr(f, "dict") else dict(f) for f in payload.files]
    report = evidence_analyzer.analyze_and_compare(
        files=files_dicts,
        complaint_text=complaint_text,
        complaint_facts=facts_dict,
        category=category,
    )

    # If linked to an existing workspace ticket, attach the evidence report to its triage
    if payload.ticket_id:
        with workspace_store.lock:
            t = workspace_store.tickets.get(payload.ticket_id)
            if t:
                t["triage"]["evidence_report"] = report
                workspace_store.touch(t)
                workspace_store._save()

    return report


@app.post("/api/evidence/upload", tags=["Evidence Analysis"])
async def upload_and_analyze_evidence(
    file: UploadFile = File(...),
    complaint_text: str = Form(""),
    ticket_id: Optional[str] = Form(None),
    category: str = Form("rp_violation"),
    ocr_hint: str = Form(""),
) -> Dict[str, Any]:
    """Upload a real JPG, PNG, MP4, or WEBM file for binary inspection, OCR, and consistency verification."""
    raw_bytes = await file.read()
    b64_data = base64.b64encode(raw_bytes).decode("ascii")
    req = EvidenceAnalyzeRequest(
        ticket_id=ticket_id or None,
        complaint_text=complaint_text,
        category=category,
        files=[
            {
                "filename": file.filename or "uploaded_file.bin",
                "content_type": file.content_type,
                "size_bytes": len(raw_bytes),
                "data_base64": b64_data,
                "ocr_text": ocr_hint or None,
            }
        ],
    )
    return await analyze_evidence_endpoint(req)


# ============================================================
# 6. Fraud Detection — statistics and suspicious complaints
# ============================================================

@app.get("/api/fraud/stats", tags=["Fraud Detection"])
async def get_fraud_stats() -> Dict[str, Any]:
    """Return fraud detection statistics, top suspicious reporters/offenders, and flagged complaints."""
    return {
        "stats": fraud_detector.get_statistics(),
        "top_reporters": fraud_detector.get_top_reporters(limit=5),
        "top_offenders": fraud_detector.get_top_offenders(limit=5),
        "suspicious_complaints": fraud_detector.list_suspicious_complaints(min_score=25, limit=25),
        "automated_punishment_blocked": True,
        "disclaimer": "Данные носят аналитический характер. Никакие автоматические наказания на основании подозрения не применяются — решение принимает администратор.",
    }


@app.get("/api/fraud/suspicious", tags=["Fraud Detection"])
async def list_suspicious_complaints(min_score: int = 25, limit: int = 30) -> Dict[str, Any]:
    """Return detailed list of suspicious or contradictory complaints for admin review."""
    items = fraud_detector.list_suspicious_complaints(min_score=min_score, limit=limit)
    return {
        "count": len(items),
        "automated_punishment_blocked": True,
        "items": items,
    }


# ============================================================
# 7. Player History — repeat offender profile cards by ID & Nickname
# ============================================================

@app.get("/api/player-history/search", tags=["Player History"])
async def search_player_history(
    q: str = Query(..., min_length=1, description="Никнейм (напр. Player_777) или игровой ID (напр. 777, 1042)"),
    player_id: Optional[int] = Query(None, description="Опциональный числовой игровой ID"),
) -> Dict[str, Any]:
    """Search violation & punishment history by player nickname OR numeric in-game ID."""
    history = player_history_store.get_player_history(q, player_id=player_id)
    if not history:
        return {"found": False, "query": q, "api_status": player_history_store.get_api_status()}
    return history


@app.get("/api/player-history/{nickname}", tags=["Player History"])
async def get_player_history(nickname: str, player_id: Optional[int] = None) -> Dict[str, Any]:
    """Get violation history card for a player nickname or in-game ID."""
    history = player_history_store.get_player_history(nickname, player_id=player_id)
    if not history:
        return {"found": False, "nickname": nickname, "api_status": player_history_store.get_api_status()}
    return history


@app.get("/api/repeat-offenders", tags=["Player History"])
async def get_repeat_offenders(min_complaints: int = 2, limit: int = 15) -> Dict[str, Any]:
    """Get list of repeat offenders for admin dashboard."""
    offenders = player_history_store.get_repeat_offenders(min_complaints=min_complaints, limit=limit)
    return {
        "count": len(offenders),
        "offenders": offenders,
        "api_status": player_history_store.get_api_status(),
        "disclaimer": "Жалобы строго разделены на подтверждённые нарушения и неподтверждённые обращения. Решения принимает администратор.",
    }


# ============================================================
# 11. Feedback Loop — admin corrections for AI quality
# ============================================================

@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(payload: FeedbackSubmit) -> Dict[str, Any]:
    """
    Submit admin feedback on an AI triage decision.
    Persists feedback and dynamically updates classifier weights for continuous learning.
    """
    ticket_text = payload.ticket_text
    ticket_kind = "complaint"
    if not ticket_text and payload.ticket_id:
        with workspace_store.lock:
            t = workspace_store.tickets.get(payload.ticket_id)
            if t:
                ticket_text = _player_text(t)
                ticket_kind = t.get("kind", "complaint")
        if not ticket_text:
            for item in _ui_tickets_store:
                if item.get("id") == payload.ticket_id:
                    ticket_text = _player_text(item)
                    ticket_kind = item.get("kind", "complaint")
                    break

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
        ticket_text=ticket_text,
        ticket_kind=ticket_kind,
    )
    logger.info("feedback_submitted ticket=%s type=%s admin=%s", payload.ticket_id, payload.feedback_type, payload.admin_name)
    return {
        "status": "recorded",
        "feedback_id": entry["id"],
        "feedback": entry,
        "stats": feedback_store.get_stats(),
    }


@app.get("/api/feedback/stats", tags=["Feedback"])
async def get_feedback_stats() -> Dict[str, Any]:
    """Return AI feedback quality statistics including accuracy rate."""
    return feedback_store.get_stats()


@app.get("/api/feedback", tags=["Feedback"])
async def list_feedback(limit: int = 50) -> Dict[str, Any]:
    """List all recorded feedback entries."""
    items = feedback_store.list_all(limit=limit)
    return {"count": len(items), "stats": feedback_store.get_stats(), "feedback": items}
