"""
app/config.py
---------------
Centralized configuration using Pydantic BaseSettings.
All environment variables are validated on import, ensuring fast-fail on misconfiguration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/mandi_db",
        description="Async SQLAlchemy database URL",
    )

    # Database connection pool
    DB_POOL_SIZE: int = Field(default=20, description="SQLAlchemy connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections beyond pool_size")
    DB_POOL_RECYCLE: int = Field(default=1800, description="Recycle connections after N seconds")

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (broker + cache)",
    )

    # JWT
    SECRET_KEY: str = Field(
        ...,
        description="Secret key for JWT signing (min 32 chars)",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=1440, description="Token expiry in minutes (default 24h)")

    @model_validator(mode="before")
    @classmethod
    def _validate_secret_key_presence(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        secret_key = data.get("SECRET_KEY")
        if secret_key is None:
            raise ValueError(
                "FATAL: SECRET_KEY environment variable is not set. "
                "The application cannot start without it. "
                "Generate one with: openssl rand -hex 32"
            )
        if not isinstance(secret_key, str):
            raise ValueError("FATAL: SECRET_KEY must be a string.")
        if not secret_key.strip():
            raise ValueError(
                "FATAL: SECRET_KEY environment variable is empty. "
                "The application cannot start without it. "
                "Generate one with: openssl rand -hex 32"
            )

        data["SECRET_KEY"] = secret_key.strip()
        return data

    @field_validator("SECRET_KEY")
    @classmethod
    def _validate_secret_key_length(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError(
                "FATAL: SECRET_KEY must be at least 32 characters long. "
                "Generate one with: openssl rand -hex 32"
            )
        return value

    # External API Keys
    DATAGOV_API_KEY: str = Field(default="", description="data.gov.in API key for Agmarknet")
    OPENWEATHER_API_KEY: str = Field(default="", description="OpenWeatherMap API key")
    TWILIO_ACCOUNT_SID: str = Field(default="", description="Twilio SID for WhatsApp/SMS")
    TWILIO_AUTH_TOKEN: str = Field(default="", description="Twilio auth token")

    # MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI",
    )

    # Mandi Dataset
    MANDI_DATASET_PATH: str = Field(
        default="data/raw/mandi_prices.csv",
        description="Path to mandi price CSV (absolute or relative to project root)",
    )

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        description="Comma-separated allowed CORS origins",
    )

    # Slack
    SLACK_WEBHOOK_URL: str = Field(default="", description="Slack webhook for notifications")

    # ML Model Constants
    SEQUENCE_LENGTH: int = Field(default=60, description="LSTM input sequence length")
    NUM_FEATURES: int = Field(default=53, description="Number of engineered features")
    OUTPUT_STEPS: int = Field(default=30, description="Multi-step forecast horizon")
    MC_DROPOUT_ITERATIONS: int = Field(default=50, description="MC Dropout forward passes")
    PROMOTION_THRESHOLD: float = Field(default=0.02, description="Min improvement % to promote model")

    # App
    PORT: int = Field(default=8000, description="Uvicorn server port")
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _validate_cors(cls, value: str) -> str:
        return value.strip()

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Cached singleton instantiated once and reused across the app."""
    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(_format_settings_error(exc)) from None


def _format_settings_error(exc: ValidationError) -> str:
    for error in exc.errors():
        message = error.get("msg", "")
        location = str(error.get("loc", ()))
        if "SECRET_KEY" in location or "SECRET_KEY" in message:
            return message

    return "FATAL: Application settings are invalid. Check required environment variables."
