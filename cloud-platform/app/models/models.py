import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


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
