"""
ONA Cloud — Referral API
Endpoints for the SYNARA → ONA referral pipeline.

SYNARA creates referrals. Clinic nurses look them up. Scans link to them.
This is the single source of truth for all referral data.
"""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Referral, generate_referral_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/referrals", tags=["referrals"])


# ── Request/Response Models ──

class ReferralCreate(BaseModel):
    """SYNARA sends this to create a referral."""
    synara_session_id: Optional[str] = None
    patient_hash: Optional[str] = None
    suspected_condition: str  # tb, pneumonia, respiratory
    symptoms: Optional[list] = None
    triage_confidence: Optional[float] = None
    urgency: str  # LOW, MEDIUM, HIGH, CRITICAL
    patient_language: Optional[str] = None
    patient_demographics: Optional[dict] = None
    voice_note_uri: Optional[str] = None
    photo_uri: Optional[str] = None
    referred_site_id: Optional[str] = None


class ReferralResponse(BaseModel):
    referral_code: str
    id: str
    suspected_condition: str
    symptoms: Optional[list] = None
    triage_confidence: Optional[float] = None
    urgency: str
    patient_language: Optional[str] = None
    patient_demographics: Optional[dict] = None
    status: str
    referred_at: Optional[str] = None
    scan_id: Optional[str] = None
    ona_result: Optional[str] = None
    ona_confidence: Optional[float] = None
    outcome: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class ReferralStatusUpdate(BaseModel):
    """Update referral status (nurse marks arrival, scan links, etc.)"""
    status: Optional[str] = None  # arrived, scanned, completed, no_show
    scan_id: Optional[str] = None
    ona_result: Optional[str] = None
    ona_confidence: Optional[float] = None
    outcome: Optional[str] = None
    follow_up_sent: Optional[bool] = None
    follow_up_response: Optional[str] = None


# ── Helper ──

def referral_to_dict(r: Referral) -> dict:
    return {
        "referral_code": r.referral_code,
        "id": str(r.id),
        "suspected_condition": r.suspected_condition,
        "symptoms": r.symptoms,
        "triage_confidence": r.triage_confidence,
        "urgency": r.urgency,
        "patient_language": r.patient_language,
        "patient_demographics": r.patient_demographics,
        "status": r.status,
        "referred_at": r.referred_at.isoformat() if r.referred_at else None,
        "scan_id": str(r.scan_id) if r.scan_id else None,
        "ona_result": r.ona_result,
        "ona_confidence": r.ona_confidence,
        "outcome": r.outcome,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# ── Endpoints ──

@router.post("", response_model=ReferralResponse)
def create_referral(data: ReferralCreate, db: Session = Depends(get_db)):
    """Create a new referral (called by SYNARA).
    Returns the referral code for the patient."""

    # Generate unique code (retry on collision)
    for _ in range(10):
        code = generate_referral_code()
        existing = db.query(Referral).filter(Referral.referral_code == code).first()
        if not existing:
            break
    else:
        raise HTTPException(status_code=500, detail="Failed to generate unique referral code")

    referral = Referral(
        referral_code=code,
        synara_session_id=data.synara_session_id,
        patient_hash=data.patient_hash,
        suspected_condition=data.suspected_condition,
        symptoms=data.symptoms,
        triage_confidence=data.triage_confidence,
        urgency=data.urgency,
        patient_language=data.patient_language,
        patient_demographics=data.patient_demographics,
        voice_note_uri=data.voice_note_uri,
        photo_uri=data.photo_uri,
        referred_site_id=data.referred_site_id,
        status="pending",
    )

    db.add(referral)
    db.commit()
    db.refresh(referral)

    logger.info(f"Referral created: {code} | {data.suspected_condition} | {data.urgency}")
    return ReferralResponse(**referral_to_dict(referral))


@router.get("/lookup/{code}", response_model=ReferralResponse)
def lookup_referral(code: str, db: Session = Depends(get_db)):
    """Look up a referral by code (used by clinic nurse)."""
    referral = db.query(Referral).filter(Referral.referral_code == code.upper()).first()
    if not referral:
        # Try case-insensitive
        referral = db.query(Referral).filter(
            Referral.referral_code.ilike(code)
        ).first()
    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")

    return ReferralResponse(**referral_to_dict(referral))


@router.patch("/{code}", response_model=ReferralResponse)
def update_referral(code: str, data: ReferralStatusUpdate, db: Session = Depends(get_db)):
    """Update referral status (nurse marks arrival, links scan, etc.)."""
    referral = db.query(Referral).filter(Referral.referral_code == code.upper()).first()
    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")

    now = datetime.utcnow()

    if data.status:
        referral.status = data.status
        if data.status == "arrived":
            referral.arrived_at = now
        elif data.status == "scanned":
            referral.scanned_at = now
        elif data.status == "completed":
            referral.completed_at = now

    if data.scan_id:
        referral.scan_id = data.scan_id
    if data.ona_result:
        referral.ona_result = data.ona_result
    if data.ona_confidence is not None:
        referral.ona_confidence = data.ona_confidence
    if data.outcome:
        referral.outcome = data.outcome
    if data.follow_up_sent is not None:
        referral.follow_up_sent = data.follow_up_sent
    if data.follow_up_response:
        referral.follow_up_response = data.follow_up_response

    referral.updated_at = now
    db.commit()
    db.refresh(referral)

    logger.info(f"Referral updated: {code} → {referral.status}")
    return ReferralResponse(**referral_to_dict(referral))


@router.get("/{code}/status")
def get_referral_status(code: str, db: Session = Depends(get_db)):
    """Get referral status (SYNARA polls this for follow-up)."""
    referral = db.query(Referral).filter(Referral.referral_code == code.upper()).first()
    if not referral:
        raise HTTPException(status_code=404, detail="Referral not found")

    return {
        "referral_code": referral.referral_code,
        "status": referral.status,
        "ona_result": referral.ona_result,
        "ona_confidence": referral.ona_confidence,
        "outcome": referral.outcome,
        "follow_up_sent": referral.follow_up_sent,
    }


@router.get("", response_model=List[ReferralResponse])
def list_referrals(
    status: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """List referrals (for dashboard). Filter by status or site."""
    query = db.query(Referral).order_by(Referral.created_at.desc())

    if status:
        query = query.filter(Referral.status == status)
    if site_id:
        query = query.filter(Referral.referred_site_id == site_id)

    referrals = query.limit(limit).all()
    return [ReferralResponse(**referral_to_dict(r)) for r in referrals]


@router.get("/stats/summary")
def referral_stats(db: Session = Depends(get_db)):
    """Summary stats for dashboard."""
    total = db.query(Referral).count()
    pending = db.query(Referral).filter(Referral.status == "pending").count()
    arrived = db.query(Referral).filter(Referral.status == "arrived").count()
    scanned = db.query(Referral).filter(Referral.status == "scanned").count()
    completed = db.query(Referral).filter(Referral.status == "completed").count()
    no_show = db.query(Referral).filter(Referral.status == "no_show").count()

    return {
        "total": total,
        "pending": pending,
        "arrived": arrived,
        "scanned": scanned,
        "completed": completed,
        "no_show": no_show,
        "conversion_rate": round(completed / total * 100, 1) if total > 0 else 0,
    }
