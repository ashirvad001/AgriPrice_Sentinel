"""
api/routes_forecast.py
----------------------
Forecast endpoint using Clean Architecture (Service-Repository pattern).
Delegates all business logic and data access to ForecastService.
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import CropPath, MandiPath, ForecastResponse
from app.services.forecast_service import ForecastService

router = APIRouter(prefix="/api/v1", tags=["Forecast"])


@router.get(
    "/forecast/{crop}/{mandi}",
    response_model=ForecastResponse,
    summary="Get multi-step crop price forecast",
    description=(
        "Runs real model inference for the specified crop and mandi, including "
        "95% confidence intervals from Monte Carlo Dropout, MSP comparison, "
        "and a SELL or HOLD recommendation. The endpoint does not fall back to "
        "synthetic or statistical placeholder data."
    ),
)
async def get_forecast(
    crop: CropPath,
    mandi: MandiPath,
    horizon: int = Query(30, ge=1, le=90, description="Forecast horizon in days"),
    db: AsyncSession = Depends(get_db),
):
    """Forecast crop prices for a given mandi using fresh model inference."""
    service = ForecastService(db)
    response, source = await service.get_forecast(crop, mandi, horizon, force_refresh=True)

    resp = JSONResponse(content=response.model_dump(mode="json"))
    resp.headers["X-Forecast-Source"] = source
    return resp
