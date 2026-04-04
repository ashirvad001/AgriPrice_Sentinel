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
"""

import os
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from api.deps import init_redis, close_redis
from api.auth import router as auth_router
from api.routes_forecast import router as forecast_router
from api.routes_prices import router as prices_router
from api.routes_alerts import router as alerts_router
from api.routes_whatsapp import router as whatsapp_router
from api.routes_shap import router as shap_router


# ═══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN — startup / shutdown hooks
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler: init DB tables and Redis pool on startup."""
    # ── Startup ──────────────────────────────────────────────────────────
    print("🚀  Starting AgriPrice Sentinel API…")
    try:
        await init_db()
        print("✅  Database tables ready")
    except Exception as e:
        print(f"⚠  Database init skipped ({e})")

    await init_redis()

    yield  # ← app runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    await close_redis()
    print("👋  AgriPrice Sentinel API shut down")


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
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
    print("📊  Prometheus metrics enabled at /metrics")
except ImportError:
    print("⚠  prometheus-fastapi-instrumentator not installed — /metrics disabled")


from api.routes_ws import router as ws_router

# ── Register routers ────────────────────────────────────────────────────────
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
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
