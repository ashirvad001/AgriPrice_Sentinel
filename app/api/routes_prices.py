"""
api/routes_prices.py
────────────────────
Historical price retrieval from the raw_prices table.
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from app.database import RawPrice
from app.api.schemas import CropPath, MandiPath, PriceHistoryResponse, PriceRecord
from app.api.deps import get_db
from app.logger import get_logger
from app.utils.lttb import downsample
import numpy as np

logger = get_logger(__name__)

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
    crop: CropPath,
    mandi: MandiPath,
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

    # Sort prices chronologically
    prices.sort(key=lambda x: x.date)

    original_count = len(prices)

    # Downsample using LTTB if there are too many points
    MAX_POINTS = 365
    if len(prices) > MAX_POINTS:
        logger.info(f"Downsampling {len(prices)} points to {MAX_POINTS} via LTTB for {crop}/{mandi}")
        
        # We need a numeric x-axis for LTTB (e.g. timestamp)
        # Filter out records without a modal price, as we need a y-value
        valid_prices = [p for p in prices if p.modal_price is not None]
        
        if len(valid_prices) > MAX_POINTS:
            data_points = np.array([
                [p.date.toordinal(), p.modal_price] 
                for p in valid_prices
            ])
            
            try:
                downsampled_data = downsample(data_points.tolist(), n_out=MAX_POINTS)
                
                # Reconstruct PriceRecord list from downsampled ordinals
                # Note: This loses min/max info on the dropped points, 
                # but preserves the visual shape of the time series
                downsampled_prices = []
                # Map ordinals back to original records for full data (if needed)
                ordinal_to_record = {p.date.toordinal(): p for p in valid_prices}
                
                for x, y in downsampled_data:
                    ordinal = int(x)
                    original_record = ordinal_to_record.get(ordinal)
                    if original_record:
                        downsampled_prices.append(original_record)
                    else:
                        # Fallback if interpolation happened (unlikely with LTTB on int x)
                        downsampled_prices.append(PriceRecord(
                            date=date.fromordinal(ordinal),
                            modal_price=y,
                            mandi=mandi,
                        ))
                prices = downsampled_prices
            except Exception as e:
                logger.error(f"LTTB downsampling failed: {e}")

    return PriceHistoryResponse(
        crop=crop,
        mandi=mandi,
        days_requested=days,
        total_records=original_count,
        prices=prices,
    )
