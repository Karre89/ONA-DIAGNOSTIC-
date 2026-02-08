import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.core.config import settings
from app.api.router import api_router
# Import all models to ensure they're registered with SQLAlchemy before create_all()
from app.models import models  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Ona Cloud Platform...")

    # Create database tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

    yield

    # Shutdown
    logger.info("Shutting down Ona Cloud Platform...")


app = FastAPI(
    title="Ona Health Cloud Platform",
    description="Central control plane for edge-first medical imaging",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware - origins configured via ALLOWED_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ona-cloud-platform"}


@app.get("/")
def root():
    return {
        "service": "Ona Health Cloud Platform",
        "version": "0.1.0",
        "tagline": "See Clearly. Act Quickly."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
