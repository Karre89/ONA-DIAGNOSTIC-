from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.models.models import InferenceResult, Device
from app.api.devices import hash_token

router = APIRouter(prefix="/results", tags=["results"])


class ResultUploadRequest(BaseModel):
    tenant_id: UUID
    device_id: UUID
    study_id: str
    modality: str
    model_version: str
    inference_time_ms: Optional[int] = None
    scores: dict
    risk_bucket: str
    explanation: Optional[str] = None
    input_hash: str
    has_burned_in_text: bool = False
    heatmap_uri: Optional[str] = None


class ResultUploadResponse(BaseModel):
    result_id: UUID


class ResultItem(BaseModel):
    id: UUID
    study_id: str
    modality: str
    model_version: str
    risk_bucket: str
    scores: dict
    explanation: Optional[str]
    inference_time_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class ResultsListResponse(BaseModel):
    results: list[ResultItem]
    total: int


def get_device_from_token(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> Device:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")
    token_hash = hash_token(token)

    device = db.query(Device).filter(Device.device_token_hash == token_hash).first()
    if not device:
        raise HTTPException(status_code=401, detail="Invalid device token")

    return device


@router.post("", response_model=ResultUploadResponse)
def upload_result(
    request: ResultUploadRequest,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    device = get_device_from_token(authorization, db)

    # Verify device matches request
    if device.id != request.device_id:
        raise HTTPException(status_code=403, detail="Device ID mismatch")

    # Check for duplicate (idempotent upload)
    existing = db.query(InferenceResult).filter(
        InferenceResult.device_id == request.device_id,
        InferenceResult.study_id == request.study_id,
        InferenceResult.model_version == request.model_version,
        InferenceResult.input_hash == request.input_hash
    ).first()

    if existing:
        return ResultUploadResponse(result_id=existing.id)

    # Create new result
    result = InferenceResult(
        tenant_id=request.tenant_id,
        site_id=device.site_id,
        device_id=request.device_id,
        study_id=request.study_id,
        modality=request.modality,
        model_version=request.model_version,
        inference_time_ms=request.inference_time_ms,
        scores_json=request.scores,
        risk_bucket=request.risk_bucket,
        explanation=request.explanation,
        input_hash=request.input_hash,
        has_burned_in_text=request.has_burned_in_text,
        heatmap_uri=request.heatmap_uri
    )

    db.add(result)
    db.commit()
    db.refresh(result)

    return ResultUploadResponse(result_id=result.id)


@router.get("/recent", response_model=ResultsListResponse)
def get_recent_results(
    tenant_id: Optional[UUID] = Query(None),
    site_code: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(InferenceResult)

    if tenant_id:
        query = query.filter(InferenceResult.tenant_id == tenant_id)

    # TODO: Filter by site_code if provided (requires join)

    results = query.order_by(desc(InferenceResult.created_at)).limit(limit).all()

    result_items = [
        ResultItem(
            id=r.id,
            study_id=r.study_id,
            modality=r.modality,
            model_version=r.model_version,
            risk_bucket=r.risk_bucket,
            scores=r.scores_json,
            explanation=r.explanation,
            inference_time_ms=r.inference_time_ms,
            created_at=r.created_at
        )
        for r in results
    ]

    total = query.count()

    return ResultsListResponse(results=result_items, total=total)


@router.get("/{result_id}")
def get_result(result_id: UUID, db: Session = Depends(get_db)):
    result = db.query(InferenceResult).filter(InferenceResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    return {
        "id": str(result.id),
        "study_id": result.study_id,
        "modality": result.modality,
        "model_version": result.model_version,
        "risk_bucket": result.risk_bucket,
        "scores": result.scores_json,
        "explanation": result.explanation,
        "inference_time_ms": result.inference_time_ms,
        "has_burned_in_text": result.has_burned_in_text,
        "created_at": result.created_at.isoformat()
    }
