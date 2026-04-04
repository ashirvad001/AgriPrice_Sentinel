import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Date, DateTime, Text, JSON, Float, Boolean, Integer, UniqueConstraint
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class Base(DeclarativeBase):
    pass

class RawPrice(Base):
    __tablename__ = "raw_prices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100), index=True)
    state: Mapped[str] = mapped_column(String(100), index=True)
    fetch_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    raw_data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ScrapeError(Base):
    __tablename__ = "scrape_errors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(100))
    error_message: Mapped[str] = mapped_column(Text)
    failed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class WeatherObservation(Base):
    __tablename__ = "weather_obs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    district: Mapped[str] = mapped_column(String(100), index=True)
    date: Mapped[datetime.date] = mapped_column(Date, index=True)
    
    # Weather Metrics
    rainfall_mm: Mapped[float] = mapped_column(nullable=True)
    max_temp: Mapped[float] = mapped_column(nullable=True)
    min_temp: Mapped[float] = mapped_column(nullable=True)
    humidity: Mapped[float] = mapped_column(nullable=True)
    wind_speed: Mapped[float] = mapped_column(nullable=True)

    # Maintain uniqueness
    __table_args__ = (
        UniqueConstraint('district', 'date', name='uq_weather_district_date'),
    )

class ModelDiagnostic(Base):
    __tablename__ = "model_diagnostics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100), index=True)
    mandi: Mapped[str] = mapped_column(String(200), index=True)
    test_name: Mapped[str] = mapped_column(String(50))  # e.g. "ADF"
    stage: Mapped[str] = mapped_column(String(30))       # "original" or "differenced"
    adf_statistic: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    critical_1pct: Mapped[float] = mapped_column(Float, nullable=True)
    critical_5pct: Mapped[float] = mapped_column(Float, nullable=True)
    critical_10pct: Mapped[float] = mapped_column(Float, nullable=True)
    is_stationary: Mapped[bool] = mapped_column(Boolean)
    differencing_applied: Mapped[bool] = mapped_column(Boolean)
    tested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('crop', 'mandi', 'test_name', 'stage',
                         name='uq_diag_crop_mandi_test_stage'),
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
    optimized_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ShapExplanation(Base):
    """Stores per-feature SHAP values for each prediction, enabling
    farmer-friendly explainability on the frontend dashboard."""
    __tablename__ = "shap_explanations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100), index=True)
    prediction_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    feature_name: Mapped[str] = mapped_column(String(100))
    shap_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=True)
    farmer_label: Mapped[str] = mapped_column(String(200))  # e.g. "Price 1 week ago"
    rank: Mapped[int] = mapped_column(Integer, nullable=False)  # importance rank
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class User(Base):
    """Registered farmer / user for JWT authentication."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(String(15), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AlertSubscription(Base):
    """Farmer price-alert subscriptions for WhatsApp / SMS notifications."""
    __tablename__ = "alert_subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=True)  # Optional for WhatsApp-direct
    phone_number: Mapped[str] = mapped_column(String(20), index=True, nullable=True)
    language: Mapped[str] = mapped_column(String(20), default="English")
    crop: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    mandi: Mapped[str] = mapped_column(String(200), nullable=False)
    threshold_price: Mapped[float] = mapped_column(Float, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RetrainingLog(Base):
    """Audit log for weekly LSTM model retraining runs."""
    __tablename__ = "retraining_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    crop: Mapped[str] = mapped_column(String(100), index=True)
    mandi: Mapped[str] = mapped_column(String(200), index=True, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    rmse_before: Mapped[float] = mapped_column(Float, nullable=True)
    rmse_after: Mapped[float] = mapped_column(Float, nullable=True)
    improvement_pct: Mapped[float] = mapped_column(Float, nullable=True)
    model_promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    feature_importance_delta: Mapped[dict] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | success | failed
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
