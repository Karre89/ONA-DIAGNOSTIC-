import secrets
import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Device, Site, Tenant

router = APIRouter(prefix="/devices", tags=["devices"])


class DeviceRegisterRequest(BaseModel):
    tenant_id: UUID
    site_code: str
    device_name: str
    hardware_info: Optional[dict] = None


class DeviceRegisterResponse(BaseModel):
    device_id: UUID
    device_token: str
    site_id: UUID


class HeartbeatRequest(BaseModel):
    timestamp: datetime
    edge_version: Optional[str] = None
    models_loaded: Optional[list] = None
    health: Optional[dict] = None


class HeartbeatResponse(BaseModel):
    status: str
    commands: list = []


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify_device_token(
    device_id: UUID,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> Device:
    import logging
    log = logging.getLogger("ona.device_auth")

    if not authorization.startswith("Bearer "):
        log.warning(f"Device {device_id}: missing Bearer prefix in auth header")
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")
    token_hash = hash_token(token)

    # Check if device exists
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        log.warning(f"Device {device_id}: not found in database")
        raise HTTPException(status_code=401, detail="Device not found")

    # Check token match
    if device.device_token_hash != token_hash:
        log.warning(f"Device {device_id}: token mismatch. received={token_hash[:16]}... stored={device.device_token_hash[:16]}...")
        raise HTTPException(status_code=401, detail="Invalid device token")

    return device


@router.post("/register", response_model=DeviceRegisterResponse)
def register_device(request: DeviceRegisterRequest, db: Session = Depends(get_db)):
    # Verify tenant exists
    tenant = db.query(Tenant).filter(Tenant.id == request.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Get site by code
    site = db.query(Site).filter(Site.site_code == request.site_code).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    # Generate device token
    device_token = secrets.token_urlsafe(32)
    token_hash = hash_token(device_token)

    # Create device
    device = Device(
        tenant_id=request.tenant_id,
        site_id=site.id,
        device_name=request.device_name,
        device_token_hash=token_hash,
        health=request.hardware_info,
        status="ONLINE",
        last_seen_at=datetime.utcnow()
    )

    db.add(device)
    db.commit()
    db.refresh(device)

    return DeviceRegisterResponse(
        device_id=device.id,
        device_token=device_token,
        site_id=site.id
    )


@router.post("/{device_id}/heartbeat", response_model=HeartbeatResponse)
def device_heartbeat(
    device_id: UUID,
    request: HeartbeatRequest,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    device = verify_device_token(device_id, authorization, db)

    # Update device status
    device.last_seen_at = datetime.utcnow()
    device.status = "ONLINE"
    device.edge_version = request.edge_version
    device.models_loaded = request.models_loaded
    device.health = request.health

    db.commit()

    # TODO: Check for pending commands (model updates, config changes)
    commands = []

    return HeartbeatResponse(status="ok", commands=commands)


@router.get("/{device_id}")
def get_device(device_id: UUID, db: Session = Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return {
        "id": str(device.id),
        "device_name": device.device_name,
        "site_id": str(device.site_id),
        "tenant_id": str(device.tenant_id),
        "status": device.status,
        "last_seen_at": device.last_seen_at.isoformat() if device.last_seen_at else None,
        "edge_version": device.edge_version,
        "models_loaded": device.models_loaded
    }
