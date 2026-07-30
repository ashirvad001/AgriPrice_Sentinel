"""
app/database.py
───────────────
SQLAlchemy async ORM models and engine configuration.

Changes from previous version:
  - Uses Pydantic BaseSettings (app.config) instead of raw os.getenv
  - Explicit connection pool limits (pool_size, max_overflow, pool_recycle)
  - JSON → JSONB for PostgreSQL-optimised binary storage
  - Individual indexes replaced with composite indexes where appropriate
"""

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Date, DateTime, Text, Float, Boolean, Integer,
    UniqueConstraint, Index,
)
from sqlalchemy.types import JSON  # fallback for SQLite
from app.config import get_settings

# ── Try to import JSONB; fall back to plain JSON for SQLite ──────────────────
try:
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB
    _JsonColumn = _JSONB
except ImportError:
    _JsonColumn = JSON

settings = get_settings()


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp for SQLAlchemy column defaults."""
    return datetime.now(timezone.utc)


# Detect if we're using SQLite (which doesn't support JSONB)
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_JsonType = JSON if _is_sqlite else _JsonColumn


# ── Engine with explicit connection pooling ──────────────────────────────────
_engine_kwargs = dict(echo=False)
if not _is_sqlite:
    _engine_kwargs.update(
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_pre_ping=True,
    )

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RawPrice(Base):
    __tablename__ = "raw_prices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100))
    fetch_date: Mapped[datetime.date] = mapped_column(Date)
    raw_data: Mapped[dict] = mapped_column(_JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        UniqueConstraint('crop', 'state', 'fetch_date', name='uq_rawprice_crop_state_date'),
        # Composite index replaces three individual indexes — significantly
        # faster for queries that filter on crop + state + date range.
        Index('idx_rawprice_crop_state_date', 'crop', 'state', 'fetch_date'),
    )


class ScrapeError(Base):
    __tablename__ = "scrape_errors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100))
    error_message: Mapped[str] = mapped_column(Text)
    failed_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class WeatherObservation(Base):
    __tablename__ = "weather_obs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    district: Mapped[str] = mapped_column(String(100))
    date: Mapped[datetime.date] = mapped_column(Date)

    # Weather Metrics
    rainfall_mm: Mapped[float] = mapped_column(nullable=True)
    max_temp: Mapped[float] = mapped_column(nullable=True)
    min_temp: Mapped[float] = mapped_column(nullable=True)
    humidity: Mapped[float] = mapped_column(nullable=True)
    wind_speed: Mapped[float] = mapped_column(nullable=True)

    __table_args__ = (
        UniqueConstraint('district', 'date', name='uq_weather_district_date'),
        Index('idx_weather_district_date', 'district', 'date'),
    )


class ModelDiagnostic(Base):
    __tablename__ = "model_diagnostics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    mandi: Mapped[str] = mapped_column(String(200))
    test_name: Mapped[str] = mapped_column(String(50))   # e.g. "ADF"
    stage: Mapped[str] = mapped_column(String(30))        # "original" or "differenced"
    adf_statistic: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    critical_1pct: Mapped[float] = mapped_column(Float, nullable=True)
    critical_5pct: Mapped[float] = mapped_column(Float, nullable=True)
    critical_10pct: Mapped[float] = mapped_column(Float, nullable=True)
    is_stationary: Mapped[bool] = mapped_column(Boolean)
    differencing_applied: Mapped[bool] = mapped_column(Boolean)
    tested_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        UniqueConstraint('crop', 'mandi', 'test_name', 'stage',
                         name='uq_diag_crop_mandi_test_stage'),
        Index('idx_diag_crop_mandi', 'crop', 'mandi'),
    )


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100), index=True)
    lstm_units: Mapped[int] = mapped_column(nullable=False)
    dropout: Mapped[float] = mapped_column(Float, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    sequence_length: Mapped[int] = mapped_column(nullable=False)
    batch_size: Mapped[int] = mapped_column(nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    optimized_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class ShapExplanation(Base):
    """Stores per-feature SHAP values for each prediction, enabling
    farmer-friendly explainability on the frontend dashboard."""
    __tablename__ = "shap_explanations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    prediction_date: Mapped[datetime.date] = mapped_column(Date)
    feature_name: Mapped[str] = mapped_column(String(100))
    shap_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=True)
    farmer_label: Mapped[str] = mapped_column(String(200))  # e.g. "Price 1 week ago"
    rank: Mapped[int] = mapped_column(Integer, nullable=False)  # importance rank
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        Index('idx_shap_crop_date', 'crop', 'prediction_date'),
    )


class User(Base):
    """Registered farmer / user for JWT authentication."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String(15), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)


class AlertSubscription(Base):
    """Farmer price-alert subscriptions for WhatsApp / SMS notifications."""
    __tablename__ = "alert_subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=True)  # Optional for WhatsApp-direct
    phone_number: Mapped[str] = mapped_column(String(20), index=True, nullable=True)
    language: Mapped[str] = mapped_column(String(20), nullable=False, default="English", server_default="English")
    crop: Mapped[str] = mapped_column(String(100), nullable=False)
    mandi: Mapped[str] = mapped_column(String(200), nullable=False)
    threshold_price: Mapped[float] = mapped_column(Float, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        Index('idx_alert_crop_active', 'crop', 'is_active'),
    )


class RetrainingLog(Base):
    """Audit log for weekly LSTM model retraining runs."""
    __tablename__ = "retraining_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    mandi: Mapped[str] = mapped_column(String(200), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    rmse_before: Mapped[float] = mapped_column(Float, nullable=True)
    rmse_after: Mapped[float] = mapped_column(Float, nullable=True)
    improvement_pct: Mapped[float] = mapped_column(Float, nullable=True)
    model_promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    feature_importance_delta: Mapped[dict] = mapped_column(_JsonType, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | success | failed
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)

    __table_args__ = (
        Index('idx_retrain_crop_status', 'crop', 'status'),
    )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
