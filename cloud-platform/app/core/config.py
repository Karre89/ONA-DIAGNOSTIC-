import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "onahealth")
    postgres_user: str = os.getenv("POSTGRES_USER", "onahealth")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "changeme")

    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-key")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # Server
    host: str = os.getenv("CLOUD_HOST", "0.0.0.0")
    port: int = int(os.getenv("CLOUD_PORT", "8000"))

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    class Config:
        env_file = ".env"


settings = Settings()
