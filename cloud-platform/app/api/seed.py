import secrets
import hashlib
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Tenant, Site, Device, ModelVersion

router = APIRouter(prefix="/seed", tags=["seed"])


class SeedRequest(BaseModel):
    tenant_name: str = "Ona Kenya"
    site_name: str = "Kenyatta Hospital"
    site_code: str = "KNH001"
    country: str = "KE"
    device_name: str = "edge-box-001"


class SeedResponse(BaseModel):
    tenant_id: str
    site_id: str
    device_id: str
    device_token: str
    message: str


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@router.post("", response_model=SeedResponse)
def seed_data(request: SeedRequest, db: Session = Depends(get_db)):
    # Check if tenant already exists
    existing_tenant = db.query(Tenant).filter(Tenant.name == request.tenant_name).first()

    if existing_tenant:
        tenant = existing_tenant
    else:
        tenant = Tenant(
            name=request.tenant_name,
            tenant_type="NGO"
        )
        db.add(tenant)
        db.flush()

    # Check if site already exists
    existing_site = db.query(Site).filter(Site.site_code == request.site_code).first()

    if existing_site:
        site = existing_site
    else:
        site = Site(
            tenant_id=tenant.id,
            site_code=request.site_code,
            name=request.site_name,
            country=request.country,
            has_radiologist=False,
            workflow_mode="WITHOUT_RADIOLOGIST"
        )
        db.add(site)
        db.flush()

    # Check if device already exists
    existing_device = db.query(Device).filter(
        Device.device_name == request.device_name,
        Device.site_id == site.id
    ).first()

    if existing_device:
        device = existing_device
        # Generate new token for existing device
        device_token = "dev-device-token"  # Use fixed token for dev
        device.device_token_hash = hash_token(device_token)
    else:
        device_token = "dev-device-token"  # Use fixed token for dev
        device = Device(
            tenant_id=tenant.id,
            site_id=site.id,
            device_name=request.device_name,
            device_token_hash=hash_token(device_token),
            status="OFFLINE"
        )
        db.add(device)
        db.flush()

    # Seed default model version if not exists
    existing_model = db.query(ModelVersion).filter(
        ModelVersion.modality == "CXR",
        ModelVersion.version == "ona-cxr-tb-v1.0"
    ).first()

    if not existing_model:
        model = ModelVersion(
            modality="CXR",
            version="ona-cxr-tb-v1.0",
            checksum="stub-checksum-12345",
            download_uri="file:///var/imaging-edge/models/ona-cxr-tb-v1.0.pt",
            rollout_percent=100,
            is_active=True
        )
        db.add(model)

    db.commit()

    return SeedResponse(
        tenant_id=str(tenant.id),
        site_id=str(site.id),
        device_id=str(device.id),
        device_token=device_token,
        message="Seed data created successfully"
    )
