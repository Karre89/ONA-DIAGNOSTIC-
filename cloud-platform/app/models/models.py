import uuid
import random
import string
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


def generate_referral_code():
    """Generate a short, verbal-friendly referral code like REF-4719."""
    digits = ''.join(random.choices(string.digits, k=4))
    return f"REF-{digits}"


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    tenant_type = Column(Text, nullable=False)  # NGO, PRIVATE, GOVERNMENT
    created_at = Column(DateTime, default=datetime.utcnow)

    sites = relationship("Site", back_populates="tenant")
    devices = relationship("Device", back_populates="tenant")


class Site(Base):
    __tablename__ = "sites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    site_code = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    country = Column(Text, nullable=False)
    has_radiologist = Column(Boolean, default=False)
    workflow_mode = Column(Text)  # WITH_RADIOLOGIST, WITHOUT_RADIOLOGIST, CALIBRATION_STUDY
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="sites")
    devices = relationship("Device", back_populates="site")


class Device(Base):
    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=False)
    device_name = Column(Text, nullable=False)
    device_token_hash = Column(Text, nullable=False)
    edge_version = Column(Text)
    models_loaded = Column(JSON)
    health = Column(JSON)
    last_seen_at = Column(DateTime)
    status = Column(Text, default="OFFLINE")  # ONLINE, OFFLINE, DEGRADED
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="devices")
    site = relationship("Site", back_populates="devices")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    modality = Column(Text, nullable=False)  # CXR, CT, MRI
    version = Column(Text, nullable=False)
    checksum = Column(Text, nullable=False)
    download_uri = Column(Text, nullable=False)
    rollout_percent = Column(Integer, default=0)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("modality", "version", name="uq_modality_version"),
    )


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    site_id = Column(UUID(as_uuid=True), nullable=False)
    device_id = Column(UUID(as_uuid=True), nullable=False)
    study_id = Column(Text, nullable=False)
    modality = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)
    inference_time_ms = Column(Integer)
    scores_json = Column(JSON, nullable=False)
    risk_bucket = Column(Text, nullable=False)  # LOW, MEDIUM, HIGH, NOT_CONFIDENT
    explanation = Column(Text)
    heatmap_uri = Column(Text)
    input_hash = Column(Text, nullable=False)
    has_burned_in_text = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "device_id", "study_id", "model_version", "input_hash",
            name="uq_device_study_model_hash"
        ),
    )


class FeedbackSync(Base):
    __tablename__ = "feedback_sync"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    site_id = Column(UUID(as_uuid=True), nullable=False)
    study_id = Column(Text, nullable=False)
    response = Column(Text, nullable=False)  # AGREE, DISAGREE, UNSURE
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True))
    site_id = Column(UUID(as_uuid=True))
    severity = Column(Text, nullable=False)  # P1, P2, P3, P4
    category = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text)
    status = Column(Text, default="OPEN")  # OPEN, INVESTIGATING, RESOLVED
    assigned_to = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    full_name = Column(Text, nullable=False)
    role = Column(Text, default="clinic_user")  # admin, clinic_user, viewer
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=True)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class Referral(Base):
    """SYNARA → ONA referral. Single source of truth for the referral pipeline.
    Created by SYNARA when triage detects imaging-worthy symptoms.
    Looked up by clinic nurse via short code. Linked to scan result after imaging."""
    __tablename__ = "referrals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    referral_code = Column(Text, unique=True, nullable=False, default=generate_referral_code)

    # Source: SYNARA
    synara_session_id = Column(Text)  # Links back to SYNARA conversation
    patient_hash = Column(Text)  # De-identified patient identifier
    suspected_condition = Column(Text, nullable=False)  # tb, pneumonia, etc.
    symptoms = Column(JSON)  # Structured symptom list from triage
    triage_confidence = Column(Float)  # SYNARA's confidence score (0-1)
    urgency = Column(Text, nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    patient_language = Column(Text)  # ISO code: en, sw, so, etc.
    patient_demographics = Column(JSON)  # age_range, sex — no PII
    voice_note_uri = Column(Text)  # Link to cough audio if provided
    photo_uri = Column(Text)  # Link to eyelid/sputum photo if provided

    # Destination: ONA clinic
    referred_site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=True)
    referred_at = Column(DateTime, default=datetime.utcnow)

    # Status tracking
    status = Column(Text, default="pending")
    # pending → arrived → scanned → completed → no_show
    arrived_at = Column(DateTime)
    scanned_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Link to ONA scan result (populated after imaging)
    scan_id = Column(UUID(as_uuid=True), nullable=True)
    ona_result = Column(Text)  # tb_positive, tb_negative, pneumonia_positive, etc.
    ona_confidence = Column(Float)  # ONA model confidence

    # Follow-up
    follow_up_sent = Column(Boolean, default=False)
    follow_up_response = Column(Text)

    # Outcome
    outcome = Column(Text)  # confirmed, ruled_out, inconclusive

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for fast lookup
    __table_args__ = (
        Index("ix_referral_code", "referral_code"),
        Index("ix_referral_status", "status"),
        Index("ix_referral_site", "referred_site_id"),
    )
