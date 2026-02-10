from datetime import datetime, timedelta
from typing import Optional, List
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
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


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
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Check if user already exists
        existing = db.query(User).filter(User.email == request.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        logger.info(f"Creating user: {request.email}")
        user = User(
            email=request.email,
            password_hash=hash_password(request.password),
            full_name=request.full_name,
            role=request.role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"User created: {user.id}")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {type(e).__name__}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics"""

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get cloud admin statistics"""
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
def get_sites(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all sites"""
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
def get_devices(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all devices"""
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


# --- User Management Endpoints ---

class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None
    site_id: Optional[str] = None
    is_active: Optional[bool] = None


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "clinic_user"
    tenant_id: Optional[str] = None
    site_id: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


@router.get("/users")
def list_users(
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all users. Admin only."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    query = db.query(User)
    if role:
        query = query.filter(User.role == role)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)

    users = query.order_by(User.created_at.desc()).all()
    result = []
    for u in users:
        tenant = db.query(Tenant).filter(Tenant.id == u.tenant_id).first() if u.tenant_id else None
        site = db.query(Site).filter(Site.id == u.site_id).first() if u.site_id else None
        result.append({
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "role": u.role,
            "is_active": u.is_active,
            "tenant_name": tenant.name if tenant else None,
            "site_name": site.name if site else None,
            "tenant_id": str(u.tenant_id) if u.tenant_id else None,
            "site_id": str(u.site_id) if u.site_id else None,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login": u.last_login.isoformat() if u.last_login else None,
        })
    return result


@router.post("/users")
def create_user(
    request: CreateUserRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new user. Admin only."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        role=request.role,
        tenant_id=request.tenant_id if request.tenant_id else None,
        site_id=request.site_id if request.site_id else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"id": str(user.id), "email": user.email, "message": "User created"}


@router.put("/users/{user_id}")
def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a user. Admin only (or self for profile)."""
    is_self = str(current_user.id) == user_id
    if current_user.role != "admin" and not is_self:
        raise HTTPException(status_code=403, detail="Admin access required")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if request.full_name is not None:
        user.full_name = request.full_name
    if request.role is not None and current_user.role == "admin":
        user.role = request.role
    if request.tenant_id is not None and current_user.role == "admin":
        user.tenant_id = request.tenant_id if request.tenant_id else None
    if request.site_id is not None and current_user.role == "admin":
        user.site_id = request.site_id if request.site_id else None
    if request.is_active is not None and current_user.role == "admin":
        user.is_active = request.is_active

    db.commit()
    return {"id": str(user.id), "message": "User updated"}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a user. Admin only. Cannot delete yourself."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    if str(current_user.id) == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted"}


@router.post("/change-password")
def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change own password."""
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.password_hash = hash_password(request.new_password)
    db.commit()
    return {"message": "Password changed successfully"}
