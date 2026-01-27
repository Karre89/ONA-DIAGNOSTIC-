from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import FeedbackSync, Site

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    tenant_id: UUID
    site_code: str
    study_id: str
    response: str  # AGREE, DISAGREE, UNSURE
    reason: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: UUID


@router.post("", response_model=FeedbackResponse)
def upload_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    # Validate response
    if request.response not in ["AGREE", "DISAGREE", "UNSURE"]:
        raise HTTPException(
            status_code=400,
            detail="Response must be AGREE, DISAGREE, or UNSURE"
        )

    # Get site by code
    site = db.query(Site).filter(Site.site_code == request.site_code).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    # Create feedback record
    feedback = FeedbackSync(
        tenant_id=request.tenant_id,
        site_id=site.id,
        study_id=request.study_id,
        response=request.response,
        reason=request.reason
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(feedback_id=feedback.id)


@router.get("/stats")
def get_feedback_stats(
    tenant_id: UUID | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(FeedbackSync)

    if tenant_id:
        query = query.filter(FeedbackSync.tenant_id == tenant_id)

    total = query.count()
    agree = query.filter(FeedbackSync.response == "AGREE").count()
    disagree = query.filter(FeedbackSync.response == "DISAGREE").count()
    unsure = query.filter(FeedbackSync.response == "UNSURE").count()

    agreement_rate = (agree / total * 100) if total > 0 else 0

    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "unsure": unsure,
        "agreement_rate": round(agreement_rate, 1)
    }
