from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import ModelVersion

router = APIRouter(prefix="/models", tags=["models"])


class ModelManifestItem(BaseModel):
    modality: str
    version: str
    download_uri: str
    checksum: str


class ModelManifestResponse(BaseModel):
    models: list[ModelManifestItem]


@router.get("/manifest", response_model=ModelManifestResponse)
def get_model_manifest(
    device_id: UUID = Query(...),
    db: Session = Depends(get_db)
):
    # Get all active models
    models = db.query(ModelVersion).filter(
        ModelVersion.is_active == True,
        ModelVersion.rollout_percent > 0
    ).all()

    # TODO: Implement canary rollout logic based on device_id hash

    manifest_items = [
        ModelManifestItem(
            modality=m.modality,
            version=m.version,
            download_uri=m.download_uri,
            checksum=m.checksum
        )
        for m in models
    ]

    return ModelManifestResponse(models=manifest_items)


@router.get("")
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelVersion).all()

    return {
        "models": [
            {
                "id": str(m.id),
                "modality": m.modality,
                "version": m.version,
                "is_active": m.is_active,
                "rollout_percent": m.rollout_percent,
                "created_at": m.created_at.isoformat()
            }
            for m in models
        ]
    }
