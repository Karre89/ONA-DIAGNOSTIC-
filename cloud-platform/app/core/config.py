import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Database - Railway provides DATABASE_URL directly
    database_url: Optional[str] = None

    # Fallback components for local development
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "onahealth"
    postgres_user: str = "onahealth"
    postgres_password: str = "changeme"

    # JWT
    jwt_secret: str = "dev-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Server - PORT for Railway compatibility
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS - comma-separated allowed origins
    allowed_origins: str = "http://localhost:3000"

    def get_database_url(self) -> str:
        # Railway provides DATABASE_URL directly
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    def get_allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Production safety check: refuse to run with default JWT secret if a real database is connected
if settings.database_url and settings.jwt_secret == "dev-secret-key":
    raise RuntimeError(
        "FATAL: JWT_SECRET is still set to 'dev-secret-key' but DATABASE_URL is configured. "
        "Set the JWT_SECRET environment variable to a secure random string."
    )
