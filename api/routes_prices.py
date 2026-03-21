"""
api/routes_prices.py
────────────────────
Historical price retrieval from the raw_prices table.
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from database import RawPrice
from api.schemas import PriceHistoryResponse, PriceRecord
from api.deps import get_db

router = APIRouter(prefix="/api/v1", tags=["Historical Prices"])


@router.get(
    "/prices/{crop}/{mandi}",
    response_model=PriceHistoryResponse,
    summary="Get historical mandi prices",
    description=(
        "Returns historical daily prices for a crop at a given mandi "
        "from the PostgreSQL `raw_prices` table.  Defaults to the last 365 days."
    ),
)
async def get_prices(
    crop: str,
    mandi: str,
    days: int = Query(365, ge=1, le=3650, description="Number of days of history to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    **Historical prices** for a crop–mandi pair.

    - **crop**: Crop name (e.g. `wheat`, `rice`)
    - **mandi**: Mandi market name
    - **days**: Look-back window in days (default: 365, max: 3650)

    Returns an ordered list of daily price records with `modal_price`,
    `min_price`, and `max_price` from the mandi dataset.
    """
    cutoff = date.today() - timedelta(days=days)

    stmt = (
        select(RawPrice)
        .where(
            and_(
                RawPrice.crop.ilike(crop),
                RawPrice.fetch_date >= cutoff,
            )
        )
        .order_by(desc(RawPrice.fetch_date))
    )

    result = await db.execute(stmt)
    rows = result.scalars().all()

    # Extract price fields from the raw_data JSON column
    prices: list[PriceRecord] = []
    for row in rows:
        raw = row.raw_data or {}
        # Filter by mandi if present in raw_data
        row_mandi = raw.get("market_name", raw.get("mandi", ""))
        if mandi.lower() not in row_mandi.lower() and row_mandi:
            continue
        prices.append(PriceRecord(
            date=row.fetch_date,
            modal_price=raw.get("modal_price"),
            min_price=raw.get("min_price"),
            max_price=raw.get("max_price"),
            mandi=row_mandi or mandi,
        ))

    return PriceHistoryResponse(
        crop=crop,
        mandi=mandi,
        days_requested=days,
        total_records=len(prices),
        prices=prices,
    )
