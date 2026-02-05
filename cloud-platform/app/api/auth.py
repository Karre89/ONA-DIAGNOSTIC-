from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError

from app.core.database import get_db
from app.core.config import settings
from app.models.models import User, Tenant, Site, InferenceResult

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "clinic_user"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    tenant_id: Optional[UUID]
    site_id: Optional[UUID]


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=settings.jwt_expiration_hours))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        role=request.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create token
    token = create_access_token({"sub": str(user.id)})

    return TokenResponse(
        access_token=token,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    )


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create token
    token = create_access_token({"sub": str(user.id)})

    return TokenResponse(
        access_token=token,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "tenant_id": str(user.tenant_id) if user.tenant_id else None,
            "site_id": str(user.site_id) if user.site_id else None
        }
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        site_id=current_user.site_id
    )


@router.get("/dashboard-stats")
def get_dashboard_stats(
    db: Session = Depends(get_db)
):
    """Get dashboard statistics (public for now)"""

    # Get total scans
    query = db.query(InferenceResult)
    total_scans = query.count()

    # Get risk breakdown
    high_risk = query.filter(InferenceResult.risk_bucket == "HIGH").count()
    medium_risk = query.filter(InferenceResult.risk_bucket == "MEDIUM").count()
    low_risk = query.filter(InferenceResult.risk_bucket == "LOW").count()

    # Get recent results
    recent = query.order_by(InferenceResult.created_at.desc()).limit(10).all()

    return {
        "total_scans": total_scans,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "pending_review": high_risk + medium_risk,
        "recent_results": [
            {
                "id": str(r.id),
                "study_id": r.study_id,
                "risk_bucket": r.risk_bucket,
                "scores": r.scores_json,
                "created_at": r.created_at.isoformat()
            }
            for r in recent
        ]
    }


@router.get("/cloud-stats")
def get_cloud_stats(
    db: Session = Depends(get_db)
):
    """Get cloud admin statistics (public for now)"""
    from app.models.models import Device

    total_tenants = db.query(Tenant).count()
    total_sites = db.query(Site).count()
    total_devices = db.query(Device).count()
    total_scans = db.query(InferenceResult).count()

    online_devices = db.query(Device).filter(Device.status == "ONLINE").count()

    # Get organizations with their stats
    tenants = db.query(Tenant).all()
    orgs = []
    for tenant in tenants:
        site_count = db.query(Site).filter(Site.tenant_id == tenant.id).count()
        device_count = db.query(Device).filter(Device.tenant_id == tenant.id).count()
        orgs.append({
            "id": str(tenant.id),
            "name": tenant.name,
            "type": tenant.tenant_type,
            "sites": site_count,
            "devices": device_count,
            "status": "active"
        })

    return {
        "total_organizations": total_tenants,
        "total_sites": total_sites,
        "total_devices": total_devices,
        "total_scans": total_scans,
        "devices_online_percent": round((online_devices / max(total_devices, 1)) * 100, 1),
        "organizations": orgs
    }


@router.get("/sites")
def get_sites(db: Session = Depends(get_db)):
    """Get all sites (public for now)"""
    from app.models.models import Device

    sites = db.query(Site).all()
    result = []
    for site in sites:
        tenant = db.query(Tenant).filter(Tenant.id == site.tenant_id).first()
        device_count = db.query(Device).filter(Device.site_id == site.id).count()
        result.append({
            "id": str(site.id),
            "name": site.name,
            "site_code": site.site_code,
            "country": site.country,
            "organization": tenant.name if tenant else "Unknown",
            "devices": device_count
        })
    return result


@router.get("/devices")
def get_devices(db: Session = Depends(get_db)):
    """Get all devices (public for now)"""
    from app.models.models import Device

    devices = db.query(Device).all()
    result = []
    for device in devices:
        tenant = db.query(Tenant).filter(Tenant.id == device.tenant_id).first()
        site = db.query(Site).filter(Site.id == device.site_id).first()
        result.append({
            "id": str(device.id),
            "name": device.device_name,
            "site": site.name if site else "Unknown",
            "organization": tenant.name if tenant else "Unknown",
            "status": device.status,
            "last_heartbeat": device.last_seen_at.isoformat() if device.last_seen_at else None
        })
    return result
