"""
app/config.py
─────────────
Centralized configuration using Pydantic BaseSettings.
All environment variables are validated on import, ensuring fast-fail on misconfiguration.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ValidationInfo


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    # ── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/mandi_db",
        description="Async SQLAlchemy database URL",
    )

    # ── Database connection pool ────────────────────────────────────────────
    DB_POOL_SIZE: int = Field(default=20, description="SQLAlchemy connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections beyond pool_size")
    DB_POOL_RECYCLE: int = Field(default=1800, description="Recycle connections after N seconds")

    # ── Redis ───────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (broker + cache)",
    )

    # ── JWT ──────────────────────────────────────────────────────────────────
    JWT_SECRET: str = Field(
        ..., min_length=32,
        description="Secret key for JWT signing (min 32 chars)",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=1440, description="Token expiry in minutes (default 24h)")

    @field_validator("JWT_SECRET", mode="before")
    @classmethod
    def _validate_jwt_secret(cls, v: object) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(
                "FATAL: JWT_SECRET environment variable is not set. "
                "The application CANNOT start without it.\n"
                "  → Generate one with: openssl rand -hex 32\n"
                "  → Then set it in your .env file or environment."
            )
        if not isinstance(v, str):
            raise ValueError("JWT_SECRET must be a string.")
        v = v.strip()
        if len(v) < 32:
            raise ValueError(
                f"FATAL: JWT_SECRET is too short ({len(v)} chars). "
                "A minimum of 32 characters is required.\n"
                "  → Generate a secure key with: openssl rand -hex 32"
            )
        return v

    # ── External API Keys ───────────────────────────────────────────────────
    DATAGOV_API_KEY: str = Field(default="", description="data.gov.in API key for Agmarknet")
    OPENWEATHER_API_KEY: str = Field(default="", description="OpenWeatherMap API key")
    TWILIO_ACCOUNT_SID: str = Field(default="", description="Twilio SID for WhatsApp/SMS")
    TWILIO_AUTH_TOKEN: str = Field(default="", description="Twilio auth token")

    # ── MLflow ──────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI",
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        description="Comma-separated allowed CORS origins",
    )

    # ── Slack ───────────────────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: str = Field(default="", description="Slack webhook for notifications")

    # ── ML Model Constants ──────────────────────────────────────────────────
    SEQUENCE_LENGTH: int = Field(default=60, description="LSTM input sequence length")
    NUM_FEATURES: int = Field(default=53, description="Number of engineered features")
    OUTPUT_STEPS: int = Field(default=30, description="Multi-step forecast horizon")
    MC_DROPOUT_ITERATIONS: int = Field(default=50, description="MC Dropout forward passes")
    PROMOTION_THRESHOLD: float = Field(default=0.02, description="Min improvement % to promote model")

    # ── App ──────────────────────────────────────────────────────────────────
    PORT: int = Field(default=8000, description="Uvicorn server port")
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _validate_cors(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — instantiated once and reused across the app."""
    return Settings()
