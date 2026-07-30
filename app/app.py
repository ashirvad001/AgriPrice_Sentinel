"""
app.py
──────
FastAPI application entry point for AgriPrice Sentinel.

Features:
- JWT authentication (register / login)
- Crop price forecast with Redis caching (TTL 1 hr)
- Historical prices from PostgreSQL
- Price alert subscriptions
- Prometheus metrics via /metrics
- Async SQLAlchemy 2.0 + Pydantic v2
- All routes under /api/v1 prefix
"""

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logger import get_logger
from app.database import init_db
from app.api.deps import init_redis, close_redis
from app.api.auth import router as auth_router
from app.api.routes_forecast import router as forecast_router
from app.api.routes_prices import router as prices_router
from app.api.routes_alerts import router as alerts_router
from app.api.routes_whatsapp import router as whatsapp_router
from app.api.routes_shap import router as shap_router

settings = get_settings()
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN — startup / shutdown hooks
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler: init DB tables and Redis pool on startup."""
    logger.info("Starting AgriPrice Sentinel API…")
    try:
        import alembic.config
        alembic.config.main(argv=["upgrade", "head"])
        logger.info("Database migrations applied")
    except Exception as e:
        logger.warning(f"Database migration skipped ({e})")
        # Fallback for SQLite or when alembic is not configured
        await init_db()
        logger.info("Database tables initialized via SQLAlchemy")

    await init_redis()

    yield  # ← app runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    await close_redis()
    logger.info("AgriPrice Sentinel API shut down")


# ═══════════════════════════════════════════════════════════════════════════════
#  APP INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="AgriPrice Sentinel",
    description=(
        "**Crop price forecasting API** for Indian mandi markets.\n\n"
        "Provides:\n"
        "- 🌾 Multi-step LSTM forecasts with 95% confidence intervals\n"
        "- 📉 Historical price data from 16+ crops\n"
        "- 🔔 Farmer price-alert subscriptions\n"
        "- 🔐 JWT authentication\n"
        "- 📊 Prometheus metrics at `/metrics`\n\n"
        "Built with FastAPI, Async SQLAlchemy 2.0, and Redis caching."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prometheus metrics ───────────────────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/docs", "/redoc", "/openapi.json"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed — /metrics disabled")


from app.api.routes_ws import router as ws_router

# ── Register routers (all under /api/v1 via their own prefix) ───────────────
app.include_router(auth_router)
app.include_router(forecast_router)
app.include_router(prices_router)
app.include_router(alerts_router)
app.include_router(whatsapp_router)
app.include_router(shap_router)
app.include_router(ws_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="API health check")
async def root():
    """Returns a simple health-check response confirming the API is running."""
    return {"status": "ok", "service": "AgriPrice Sentinel", "version": "1.0.0"}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
