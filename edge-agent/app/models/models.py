import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, ForeignKey

from app.core.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Study(Base):
    __tablename__ = "studies"

    id = Column(String, primary_key=True, default=generate_uuid)
    received_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="RECEIVED")  # RECEIVED, PROCESSING, READY, ERROR
    modality = Column(String)  # CXR, CT, MRI
    filepath = Column(Text)
    input_hash = Column(String)
    has_burned_in_text = Column(Boolean, default=False)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"))
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED
    attempts = Column(Integer, default=0)
    last_error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Result(Base):
    __tablename__ = "results"

    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"))
    risk_bucket = Column(String)  # LOW, MEDIUM, HIGH, NOT_CONFIDENT
    score_tb = Column(Float)
    scores_json = Column(Text)  # JSON string
    explanation = Column(Text)
    heatmap_path = Column(Text)
    model_version = Column(String)
    inference_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    synced_at = Column(DateTime)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"))
    response = Column(String)  # AGREE, DISAGREE, UNSURE
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    synced_at = Column(DateTime)


class WorkflowAction(Base):
    __tablename__ = "workflow_actions"

    id = Column(String, primary_key=True, default=generate_uuid)
    study_id = Column(String, ForeignKey("studies.id"))
    sputum_collected = Column(Boolean, default=False)
    genexpert_done = Column(Boolean, default=False)
    genexpert_result = Column(String)
    patient_referred = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    synced_at = Column(DateTime)


class SyncQueue(Base):
    __tablename__ = "sync_queue"

    id = Column(String, primary_key=True, default=generate_uuid)
    record_type = Column(String)  # result, feedback, workflow_action
    record_id = Column(String)
    payload_json = Column(Text)
    attempts = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DeviceState(Base):
    """Store device registration state"""
    __tablename__ = "device_state"

    id = Column(String, primary_key=True, default="default")
    device_id = Column(String)
    tenant_id = Column(String)
    site_id = Column(String)
    device_token = Column(String)
    registered_at = Column(DateTime)
