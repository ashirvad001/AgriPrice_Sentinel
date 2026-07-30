"""
app/repositories/forecast_repository.py
───────────────────────────────────────
Data access layer for forecast data.
Abstracts all SQLAlchemy calls and Redis caching related to forecasting.
"""

from datetime import date, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
import json

from app.database import RawPrice
from app.api.deps import cache_get, cache_set
from app.logger import get_logger

logger = get_logger(__name__)


class ForecastRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_cached_forecast(self, cache_key: str) -> Optional[dict]:
        """Retrieve a forecast from Redis cache."""
        return await cache_get(cache_key)

    async def set_cached_forecast(self, cache_key: str, data: dict, ttl: int = 3600) -> None:
        """Store a forecast in Redis cache."""
        await cache_set(cache_key, data, ttl=ttl)

    async def fetch_historical_prices(self, crop: str, mandi: str, days: int = 365) -> list[dict]:
        """
        Return a chronological list of {date, modal_price, min_price, max_price, …}
        dicts from the raw_prices table.
        """
        cutoff = date.today() - timedelta(days=days)
        stmt = (
            select(RawPrice)
            .where(and_(RawPrice.crop.ilike(crop), RawPrice.fetch_date >= cutoff))
            .order_by(desc(RawPrice.fetch_date))
        )
        result = await self.db.execute(stmt)
        rows = result.scalars().all()

        records: list[dict] = []
        for row in rows:
            raw = row.raw_data or {}
            # Fallback for old schema where raw_data was a string (SQLite issues)
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
                    
            row_mandi = raw.get("market_name", raw.get("mandi", raw.get("market", "")))
            if mandi.lower() not in row_mandi.lower() and row_mandi:
                continue
                
            records.append({
                "date": row.fetch_date,
                "modal_price": raw.get("modal_price"),
                "min_price": raw.get("min_price"),
                "max_price": raw.get("max_price"),
                "arrivals_tonnes": raw.get("arrivals_tonnes", 0),
                "rainfall_mm": raw.get("rainfall_mm", 0),
                "max_temp": raw.get("max_temp"),
                "min_temp": raw.get("min_temp"),
                "freight_index": raw.get("freight_index", 100),
                "futures_price": raw.get("futures_price"),
                "msp": raw.get("msp", 0),
            })

        # Return in chronological order
        records.sort(key=lambda r: r["date"])
        return records
