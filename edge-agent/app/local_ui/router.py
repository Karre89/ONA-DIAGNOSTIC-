import json
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.database import get_db
from app.core.config import settings
from app.models.models import Study, Result, Feedback, SyncQueue, WorkflowAction

router = APIRouter(tags=["ui"])

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Load i18n
i18n_dir = os.path.join(os.path.dirname(__file__), "i18n")
translations = {}


def load_translations():
    """Load all translation files"""
    global translations
    for lang in ["en", "sw", "so"]:
        filepath = os.path.join(i18n_dir, f"{lang}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                translations[lang] = json.load(f)
        else:
            translations[lang] = {}


def t(key: str, lang: str = "en") -> str:
    """Get translation for key"""
    if not translations:
        load_translations()
    return translations.get(lang, {}).get(key, key)


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, lang: str = "en", db: Session = Depends(get_db)):
    """Main dashboard showing today's summary and recent studies"""

    # Get today's stats
    today = datetime.utcnow().date()

    total_today = db.query(Study).filter(
        func.date(Study.received_at) == today
    ).count()

    # Risk bucket counts
    high_count = db.query(Result).join(Study).filter(
        func.date(Study.received_at) == today,
        Result.risk_bucket == "HIGH"
    ).count()

    medium_count = db.query(Result).join(Study).filter(
        func.date(Study.received_at) == today,
        Result.risk_bucket == "MEDIUM"
    ).count()

    low_count = db.query(Result).join(Study).filter(
        func.date(Study.received_at) == today,
        Result.risk_bucket == "LOW"
    ).count()

    not_confident_count = db.query(Result).join(Study).filter(
        func.date(Study.received_at) == today,
        Result.risk_bucket == "NOT_CONFIDENT"
    ).count()

    pending_count = db.query(Study).filter(
        Study.status.in_(["RECEIVED", "PROCESSING"])
    ).count()

    error_count = db.query(Study).filter(
        Study.status == "ERROR"
    ).count()

    # Sync queue count
    sync_pending = db.query(SyncQueue).count()

    # Get recent HIGH risk results
    high_risk_results = db.query(Result).filter(
        Result.risk_bucket == "HIGH"
    ).order_by(desc(Result.created_at)).limit(10).all()

    # Get recent other results
    other_results = db.query(Result).filter(
        Result.risk_bucket != "HIGH"
    ).order_by(desc(Result.created_at)).limit(10).all()

    # Load translations
    if not translations:
        load_translations()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "lang": lang,
        "t": translations.get(lang, translations.get("en", {})),
        "stats": {
            "total_today": total_today,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "not_confident": not_confident_count,
            "pending": pending_count,
            "errors": error_count,
            "sync_pending": sync_pending
        },
        "high_risk_results": high_risk_results,
        "other_results": other_results,
        "now": datetime.utcnow()
    })


@router.get("/result/{result_id}", response_class=HTMLResponse)
async def result_detail(
    request: Request,
    result_id: str,
    lang: str = "en",
    db: Session = Depends(get_db)
):
    """Detailed result view with actions and feedback"""

    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    study = db.query(Study).filter(Study.id == result.study_id).first()

    # Get existing feedback
    feedback = db.query(Feedback).filter(Feedback.study_id == result.study_id).first()

    # Get workflow actions
    workflow = db.query(WorkflowAction).filter(
        WorkflowAction.study_id == result.study_id
    ).first()

    # Parse scores
    scores = json.loads(result.scores_json) if result.scores_json else {}

    # Load translations
    if not translations:
        load_translations()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "lang": lang,
        "t": translations.get(lang, translations.get("en", {})),
        "result": result,
        "study": study,
        "scores": scores,
        "feedback": feedback,
        "workflow": workflow,
        "now": datetime.utcnow()
    })


@router.post("/feedback/{study_id}")
async def submit_feedback(
    study_id: str,
    response: str = Form(...),
    reason: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Submit clinician feedback on a result"""

    # Validate response
    if response not in ["AGREE", "DISAGREE", "UNSURE"]:
        raise HTTPException(status_code=400, detail="Invalid response")

    # Check if feedback already exists
    existing = db.query(Feedback).filter(Feedback.study_id == study_id).first()
    if existing:
        existing.response = response
        existing.reason = reason
        existing.synced_at = None  # Mark for re-sync
    else:
        feedback = Feedback(
            study_id=study_id,
            response=response,
            reason=reason
        )
        db.add(feedback)

        # Queue for sync
        sync_item = SyncQueue(
            record_type="feedback",
            record_id=feedback.id,
            payload_json=json.dumps({
                "study_id": study_id,
                "response": response,
                "reason": reason
            })
        )
        db.add(sync_item)

    db.commit()

    return {"status": "ok", "message": "Feedback recorded"}


@router.post("/workflow/{study_id}")
async def update_workflow(
    study_id: str,
    sputum_collected: bool = Form(False),
    genexpert_done: bool = Form(False),
    genexpert_result: Optional[str] = Form(None),
    patient_referred: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Update workflow actions for a study"""

    existing = db.query(WorkflowAction).filter(
        WorkflowAction.study_id == study_id
    ).first()

    if existing:
        existing.sputum_collected = sputum_collected
        existing.genexpert_done = genexpert_done
        existing.genexpert_result = genexpert_result
        existing.patient_referred = patient_referred
        existing.synced_at = None
    else:
        workflow = WorkflowAction(
            study_id=study_id,
            sputum_collected=sputum_collected,
            genexpert_done=genexpert_done,
            genexpert_result=genexpert_result,
            patient_referred=patient_referred
        )
        db.add(workflow)

    db.commit()

    return {"status": "ok", "message": "Workflow updated"}


@router.get("/studies", response_class=HTMLResponse)
async def all_studies(
    request: Request,
    lang: str = "en",
    page: int = 1,
    db: Session = Depends(get_db)
):
    """View all studies with pagination"""

    per_page = 20
    offset = (page - 1) * per_page

    total = db.query(Study).count()
    studies = db.query(Study).order_by(desc(Study.received_at)).offset(offset).limit(per_page).all()

    # Get results for each study
    study_results = {}
    for study in studies:
        result = db.query(Result).filter(Result.study_id == study.id).first()
        if result:
            study_results[study.id] = result

    # Load translations
    if not translations:
        load_translations()

    return templates.TemplateResponse("studies.html", {
        "request": request,
        "lang": lang,
        "t": translations.get(lang, translations.get("en", {})),
        "studies": studies,
        "study_results": study_results,
        "page": page,
        "total": total,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "now": datetime.utcnow()
    })


@router.get("/admin", response_class=HTMLResponse)
async def admin_panel(
    request: Request,
    lang: str = "en",
    db: Session = Depends(get_db)
):
    """Admin panel for system status"""

    from app.models.models import DeviceState

    device_state = db.query(DeviceState).filter(DeviceState.id == "default").first()

    sync_pending = db.query(SyncQueue).count()
    sync_failed = db.query(SyncQueue).filter(SyncQueue.attempts >= 3).count()

    # Load translations
    if not translations:
        load_translations()

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "lang": lang,
        "t": translations.get(lang, translations.get("en", {})),
        "device_state": device_state,
        "sync_stats": {
            "pending": sync_pending,
            "failed": sync_failed
        },
        "now": datetime.utcnow()
    })


@router.get("/images/study/{study_id}")
async def get_study_image(study_id: str, db: Session = Depends(get_db)):
    """Serve the processed study image (PNG)"""
    # Check processed directory first
    processed_path = os.path.join(settings.edge_data_dir, "processed", f"{study_id}.png")
    if os.path.exists(processed_path):
        return FileResponse(processed_path, media_type="image/png")

    # Check samples directory
    sample_path = os.path.join(settings.edge_data_dir, "samples", "sample_cxr.png")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/heatmap/{study_id}")
async def get_heatmap_image(study_id: str, db: Session = Depends(get_db)):
    """Serve the AI attention heatmap image"""
    heatmap_path = os.path.join(settings.edge_data_dir, "heatmaps", f"{study_id}_heatmap.png")

    if os.path.exists(heatmap_path):
        # Check if it's a real image or placeholder
        file_size = os.path.getsize(heatmap_path)
        if file_size > 100:  # Real PNG is larger than placeholder text
            return FileResponse(heatmap_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Heatmap not found")
