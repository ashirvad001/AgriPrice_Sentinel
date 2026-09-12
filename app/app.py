"""
app.py
------
FastAPI application entry point for AgriPrice Sentinel.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.deps import close_redis, init_redis
from app.api.routes_alerts import router as alerts_router
from app.api.routes_forecast import router as forecast_router
from app.api.routes_prices import router as prices_router
from app.api.routes_shap import router as shap_router
from app.api.routes_whatsapp import router as whatsapp_router
from app.config import get_settings
from app.database import init_db
from app.logger import get_logger
from app.services.forecast_service import CANARY_CROP, CANARY_MANDI, ForecastService

settings = get_settings()
logger = get_logger(__name__)


def _build_readiness_payload(ready: bool, reason: str | None = None) -> dict:
    payload = {
        "status": "ready" if ready else "not_ready",
        "canary_model": f"{CANARY_CROP}/{CANARY_MANDI}",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if reason:
        payload["reason"] = reason
    return payload


def _run_startup_model_probe() -> dict:
    ready, detail = ForecastService.probe_model_readiness(CANARY_CROP, CANARY_MANDI)
    if ready:
        logger.info(f"Startup model readiness probe passed for {detail}")
        return _build_readiness_payload(True)

    logger.error(f"Startup model readiness probe failed: {detail}")
    return _build_readiness_payload(False, detail)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler: init DB, Redis, and startup readiness state."""
    logger.info("Starting AgriPrice Sentinel API...")
    app.state.model_readiness = _build_readiness_payload(False, "Startup readiness probe has not run yet")

    try:
        import alembic.config

        alembic.config.main(argv=["upgrade", "head"])
        logger.info("Database migrations applied")
    except (Exception, SystemExit) as exc:
        logger.warning(f"Database migration skipped ({exc})")
        await init_db()
        logger.info("Database tables initialized via SQLAlchemy")

    await init_redis()
    app.state.model_readiness = _run_startup_model_probe()

    yield

    await close_redis()
    logger.info("AgriPrice Sentinel API shut down")


app = FastAPI(
    title="AgriPrice Sentinel",
    description=(
        "**Crop price forecasting API** for Indian mandi markets.\n\n"
        "Provides:\n"
        "- Multi-step LSTM forecasts with 95% confidence intervals\n"
        "- Historical price data from 16+ crops\n"
        "- Farmer price-alert subscriptions\n"
        "- JWT authentication\n"
        "- Prometheus metrics at `/metrics`\n\n"
        "Built with FastAPI, Async SQLAlchemy 2.0, and Redis caching."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/docs", "/redoc", "/openapi.json"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed - /metrics disabled")

from app.api.routes_ws import router as ws_router

app.include_router(auth_router)
app.include_router(forecast_router)
app.include_router(prices_router)
app.include_router(alerts_router)
app.include_router(whatsapp_router)
app.include_router(shap_router)
app.include_router(ws_router)


@app.get("/", tags=["Health"], summary="API health check")
async def root():
    """Returns a simple health-check response confirming the API is running."""
    return {"status": "ok", "service": "AgriPrice Sentinel", "version": "1.0.0"}


@app.get("/healthz/ready", tags=["Health"], summary="Model readiness check")
async def readiness_check(request: Request):
    """Report the startup canary-model readiness state."""
    readiness = getattr(
        request.app.state,
        "model_readiness",
        _build_readiness_payload(False, "Startup readiness probe has not run yet"),
    )
    status_code = 200 if readiness.get("status") == "ready" else 503
    return JSONResponse(status_code=status_code, content=readiness)


if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
