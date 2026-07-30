"""
api/routes_forecast.py
──────────────────────
Forecast endpoint using Clean Architecture (Service-Repository pattern).
Delegates all business logic and data access to ForecastService.
"""

from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ForecastResponse
from app.api.deps import get_db
from app.services.forecast_service import ForecastService

router = APIRouter(prefix="/api/v1", tags=["Forecast"])


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
    db: AsyncSession = Depends(get_db),
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
    service = ForecastService(db)
    response, source = await service.get_forecast(crop, mandi, horizon)

    # If the forecast came from the statistical baseline or cache, 
    # we return a header indicating the source.
    if source != "model":
        resp = JSONResponse(content=response.model_dump(mode="json"))
        resp.headers["X-Forecast-Source"] = source
        return resp

    return response
