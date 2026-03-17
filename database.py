import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Date, DateTime, Text, JSON, UniqueConstraint
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

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
