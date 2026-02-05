import secrets
import hashlib
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Tenant, Site, Device, ModelVersion, InferenceResult, FeedbackSync

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
        # Generate new unique token for existing device
        device_token = secrets.token_urlsafe(32)
        device.device_token_hash = hash_token(device_token)
    else:
        device_token = secrets.token_urlsafe(32)  # Unique token per device
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


class DeleteTenantResponse(BaseModel):
    deleted_tenant: str
    deleted_sites: int
    deleted_devices: int
    deleted_results: int
    deleted_feedback: int
    message: str


@router.delete("/tenant/{tenant_id}", response_model=DeleteTenantResponse)
def delete_tenant(tenant_id: str, db: Session = Depends(get_db)):
    """Delete a tenant and all associated data (sites, devices, results, feedback)."""
    # Find tenant
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    tenant_name = tenant.name

    # Get site IDs for this tenant
    site_ids = [s.id for s in db.query(Site).filter(Site.tenant_id == tenant_id).all()]

    # Delete feedback for this tenant's sites
    feedback_count = db.query(FeedbackSync).filter(FeedbackSync.tenant_id == tenant_id).delete()

    # Delete inference results for this tenant
    results_count = db.query(InferenceResult).filter(InferenceResult.tenant_id == tenant_id).delete()

    # Delete devices for this tenant
    devices_count = db.query(Device).filter(Device.tenant_id == tenant_id).delete()

    # Delete sites for this tenant
    sites_count = db.query(Site).filter(Site.tenant_id == tenant_id).delete()

    # Delete the tenant
    db.delete(tenant)
    db.commit()

    return DeleteTenantResponse(
        deleted_tenant=tenant_name,
        deleted_sites=sites_count,
        deleted_devices=devices_count,
        deleted_results=results_count,
        deleted_feedback=feedback_count,
        message=f"Successfully deleted tenant '{tenant_name}' and all associated data"
    )


class TenantInfo(BaseModel):
    id: str
    name: str
    tenant_type: str
    sites_count: int
    devices_count: int


@router.get("/tenants", response_model=List[TenantInfo])
def list_tenants(db: Session = Depends(get_db)):
    """List all tenants with their site and device counts."""
    tenants = db.query(Tenant).all()
    result = []
    for tenant in tenants:
        sites_count = db.query(Site).filter(Site.tenant_id == tenant.id).count()
        devices_count = db.query(Device).filter(Device.tenant_id == tenant.id).count()
        result.append(TenantInfo(
            id=str(tenant.id),
            name=tenant.name,
            tenant_type=tenant.tenant_type,
            sites_count=sites_count,
            devices_count=devices_count
        ))
    return result
