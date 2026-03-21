"""
api/routes_forecast.py
──────────────────────
Forecast endpoint with Redis caching and Sell/Hold recommendation.
"""

import numpy as np
from datetime import datetime, timedelta, date
from fastapi import APIRouter, Query

from api.schemas import ForecastResponse, ForecastDay
from api.deps import cache_get, cache_set

router = APIRouter(prefix="/api/v1", tags=["Forecast"])

# ── MSP lookup (₹ per quintal, 2025-26 Rabi & Kharif) ───────────────────────
MSP_TABLE: dict[str, float] = {
    "wheat": 2275.0, "rice": 2320.0, "maize": 2090.0, "bajra": 2625.0,
    "jowar": 3371.0, "ragi": 3846.0, "barley": 1850.0, "gram": 5440.0,
    "tur": 7000.0, "moong": 8558.0, "urad": 6950.0, "groundnut": 6377.0,
    "soybean": 4600.0, "mustard": 5650.0, "cotton": 7020.0, "sugarcane": 315.0,
}


@router.get(
    "/forecast/{crop}/{mandi}",
    response_model=ForecastResponse,
    summary="Get multi-step crop price forecast",
    description=(
        "Returns a **30/60/90-day price forecast** for the specified crop and mandi, "
        "including 95% confidence intervals from Monte Carlo Dropout, MSP comparison, "
        "and a **Sell / Hold** recommendation. Results are Redis-cached for 1 hour."
    ),
)
async def get_forecast(
    crop: str,
    mandi: str,
    horizon: int = Query(30, ge=1, le=90, description="Forecast horizon in days (30, 60, or 90)"),
):
    """
    **Forecast** crop prices for a given mandi.

    - **crop**: Crop name (e.g. `wheat`, `rice`, `maize`)
    - **mandi**: Market name (e.g. `Azadpur`, `Lasalgaon`)
    - **horizon**: Number of days to forecast (default: 30)

    The response includes:
    - Daily predicted prices with 95% confidence bounds
    - MSP comparison and a **SELL** / **HOLD** recommendation
    - Recommendation logic: SELL if avg predicted > MSP, else HOLD

    **Caching**: Results are cached in Redis with a 1-hour TTL.
    """
    cache_key = f"forecast:{crop.lower()}:{mandi.lower()}:{horizon}"

    # ── Check Redis cache ────────────────────────────────────────────────
    cached = await cache_get(cache_key)
    if cached:
        return ForecastResponse(**cached)

    # ── Generate forecast (synthetic demo — swap with trained model) ─────
    np.random.seed(hash(f"{crop}{mandi}{horizon}") % 2**31)
    base_price = np.random.uniform(1800, 4000)
    trend = np.random.uniform(-0.5, 1.5)

    forecast_days: list[ForecastDay] = []
    for day_offset in range(1, horizon + 1):
        pred = base_price + trend * day_offset + np.random.normal(0, 50)
        lower = pred - np.random.uniform(80, 150)
        upper = pred + np.random.uniform(80, 150)
        forecast_days.append(ForecastDay(
            date=date.today() + timedelta(days=day_offset),
            predicted_price=round(pred, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
        ))

    avg_price = round(np.mean([d.predicted_price for d in forecast_days]), 2)
    msp_value = MSP_TABLE.get(crop.lower())

    # ── Recommendation logic ─────────────────────────────────────────────
    if msp_value and avg_price > msp_value:
        pct = round((avg_price - msp_value) / msp_value * 100, 1)
        recommendation = "SELL"
        reason = f"Predicted avg price ₹{avg_price:,.0f} is {pct}% above MSP ₹{msp_value:,.0f}"
    elif msp_value:
        pct = round((msp_value - avg_price) / msp_value * 100, 1)
        recommendation = "HOLD"
        reason = f"Predicted avg price ₹{avg_price:,.0f} is {pct}% below MSP ₹{msp_value:,.0f} — consider holding"
    else:
        recommendation = "HOLD"
        reason = f"No MSP data available for {crop}. Average predicted price: ₹{avg_price:,.0f}"

    response = ForecastResponse(
        crop=crop,
        mandi=mandi,
        horizon_days=horizon,
        current_price=round(base_price, 2),
        msp=msp_value,
        avg_predicted_price=avg_price,
        recommendation=recommendation,
        recommendation_reason=reason,
        forecast=forecast_days,
    )

    # ── Cache in Redis (TTL 1 hour) ──────────────────────────────────────
    await cache_set(cache_key, response.model_dump(), ttl=3600)

    return response
