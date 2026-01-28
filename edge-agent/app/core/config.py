import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Cloud API
    cloud_api_url: str = os.getenv("CLOUD_API_URL", "http://localhost:8000")
    device_token: str = os.getenv("DEVICE_TOKEN", "dev-device-token")

    # Edge Storage
    edge_data_dir: str = os.getenv("EDGE_DATA_DIR", "/var/imaging-edge")
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "/var/imaging-edge/edge.db")

    # Server
    host: str = os.getenv("EDGE_HOST", "0.0.0.0")
    port: int = int(os.getenv("EDGE_PORT", "8080"))

    # Sync
    sync_interval_seconds: int = int(os.getenv("SYNC_INTERVAL_SECONDS", "30"))
    sync_batch_size: int = int(os.getenv("SYNC_BATCH_SIZE", "50"))
    sync_max_retries: int = int(os.getenv("SYNC_MAX_RETRIES", "5"))

    # Model
    model_dir: str = os.getenv("MODEL_DIR", "/var/imaging-edge/models")
    default_model_version: str = os.getenv("DEFAULT_MODEL_VERSION", "v2.0")

    # DICOM SCP
    dicom_scp_port: int = int(os.getenv("DICOM_SCP_PORT", "104"))
    dicom_ae_title: str = os.getenv("DICOM_AE_TITLE", "ONA_EDGE")
    dicom_scp_enabled: bool = os.getenv("DICOM_SCP_ENABLED", "true").lower() == "true"

    # Device info (set during registration)
    device_id: str = os.getenv("DEVICE_ID", "")
    tenant_id: str = os.getenv("TENANT_ID", "")
    site_id: str = os.getenv("SITE_ID", "")

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.sqlite_db_path}"

    class Config:
        env_file = ".env"


settings = Settings()
