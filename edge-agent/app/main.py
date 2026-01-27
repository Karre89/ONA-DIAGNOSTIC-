import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.core.database import engine, Base, SessionLocal, init_db
from app.core.config import settings
from app.local_ui.router import router as ui_router
from app.dicom_ingest.ingest import create_sample_study, DicomSCP, PYNETDICOM_AVAILABLE
from app.deid.deid import DeidService
from app.inference.inference import InferenceService
from app.sync_client.sync import SyncClient, run_sync_loop
from app.device_daemon.daemon import DeviceDaemon, run_heartbeat_loop
from app.models.models import DeviceState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Background tasks
background_tasks = set()

# DICOM SCP server instance
dicom_scp: DicomSCP = None


async def initialize_device(db: Session):
    """Initialize device state if not already registered"""
    state = db.query(DeviceState).filter(DeviceState.id == "default").first()

    if not state:
        # Create default state with dev token
        state = DeviceState(
            id="default",
            device_token=settings.device_token
        )
        db.add(state)
        db.commit()
        logger.info("Created default device state")

    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Ona Edge Agent...")

    # Ensure data directories exist
    dirs = [
        settings.edge_data_dir,
        os.path.join(settings.edge_data_dir, "incoming"),
        os.path.join(settings.edge_data_dir, "processed"),
        os.path.join(settings.edge_data_dir, "results"),
        os.path.join(settings.edge_data_dir, "heatmaps"),
        os.path.join(settings.edge_data_dir, "models"),
        os.path.join(settings.edge_data_dir, "samples"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Initialize database
    logger.info("Initializing database...")
    init_db()
    Base.metadata.create_all(bind=engine)

    # Initialize device state
    db = SessionLocal()
    try:
        await initialize_device(db)
    finally:
        db.close()

    # Start background sync task
    sync_task = asyncio.create_task(run_sync_loop(SessionLocal))
    background_tasks.add(sync_task)
    sync_task.add_done_callback(background_tasks.discard)

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(run_heartbeat_loop(SessionLocal))
    background_tasks.add(heartbeat_task)
    heartbeat_task.add_done_callback(background_tasks.discard)

    # Start DICOM SCP server if available and enabled
    global dicom_scp
    if PYNETDICOM_AVAILABLE and settings.dicom_scp_enabled:
        try:
            dicom_scp = DicomSCP(
                db_session_factory=SessionLocal,
                port=settings.dicom_scp_port,
                ae_title=settings.dicom_ae_title
            )
            dicom_scp.start()
            logger.info(f"DICOM SCP started on port {settings.dicom_scp_port} (AE: {settings.dicom_ae_title})")
        except Exception as e:
            logger.warning(f"Could not start DICOM SCP: {e}")
    else:
        if not PYNETDICOM_AVAILABLE:
            logger.info("DICOM SCP not available (pynetdicom not installed)")
        else:
            logger.info("DICOM SCP disabled by configuration")

    logger.info("Ona Edge Agent started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Ona Edge Agent...")

    # Stop DICOM SCP
    if dicom_scp:
        try:
            dicom_scp.stop()
            logger.info("DICOM SCP stopped")
        except Exception as e:
            logger.warning(f"Error stopping DICOM SCP: {e}")

    # Cancel background tasks
    for task in background_tasks:
        task.cancel()

    logger.info("Ona Edge Agent shutdown complete")


app = FastAPI(
    title="Ona Edge Agent",
    description="Local inference and UI for medical imaging - Works offline",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "local_ui", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include UI router
app.include_router(ui_router)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ona-edge-agent",
        "version": "0.1.0"
    }


@app.post("/api/ingest-sample")
async def ingest_sample(db: Session = Depends(get_db)):
    """
    Create and process a sample study for testing
    """
    logger.info("Ingesting sample study...")

    # Create sample study
    study = create_sample_study(db)

    # Run de-identification
    deid_service = DeidService(db)
    deid_service.process_study(study)

    # Run inference
    inference_service = InferenceService(db)
    result = inference_service.process_study(study)

    if result:
        return {
            "status": "ok",
            "study_id": study.id,
            "result_id": result.id,
            "risk_bucket": result.risk_bucket,
            "tb_score": result.score_tb,
            "message": "Sample study processed successfully"
        }
    else:
        return {
            "status": "error",
            "study_id": study.id,
            "message": "Inference failed"
        }


@app.post("/api/sync")
async def trigger_sync(db: Session = Depends(get_db)):
    """
    Manually trigger sync to cloud
    """
    sync_client = SyncClient(db)
    stats = await sync_client.sync_pending()
    return stats


@app.post("/api/register")
async def register_device(
    tenant_id: str,
    site_code: str,
    device_name: str,
    db: Session = Depends(get_db)
):
    """
    Register this device with the cloud
    """
    daemon = DeviceDaemon(db)
    success = await daemon.register_device(tenant_id, site_code, device_name)

    if success:
        state = daemon.get_device_state()
        return {
            "status": "ok",
            "device_id": state.device_id,
            "message": "Device registered successfully"
        }
    else:
        return {
            "status": "error",
            "message": "Registration failed - check cloud connectivity"
        }


@app.get("/api/device-state")
def get_device_state(db: Session = Depends(get_db)):
    """
    Get current device registration state
    """
    state = db.query(DeviceState).filter(DeviceState.id == "default").first()

    if state:
        return {
            "device_id": state.device_id,
            "tenant_id": state.tenant_id,
            "site_id": state.site_id,
            "registered_at": state.registered_at.isoformat() if state.registered_at else None
        }
    else:
        return {"status": "not_registered"}


@app.post("/api/link-device")
def link_device(
    tenant_id: str,
    site_id: str,
    device_id: str,
    device_token: str = None,
    db: Session = Depends(get_db)
):
    """
    Link this edge device to a cloud-registered device.
    Use this after running the cloud seed endpoint.
    """
    from datetime import datetime

    state = db.query(DeviceState).filter(DeviceState.id == "default").first()

    if not state:
        state = DeviceState(id="default")
        db.add(state)

    state.tenant_id = tenant_id
    state.site_id = site_id
    state.device_id = device_id
    if device_token:
        state.device_token = device_token
    state.registered_at = datetime.utcnow()

    db.commit()

    logger.info(f"Device linked: tenant={tenant_id}, device={device_id}")

    return {
        "status": "ok",
        "message": "Device linked successfully",
        "device_id": device_id,
        "tenant_id": tenant_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
