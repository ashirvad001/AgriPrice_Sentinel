from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import cache_get
from app.database import RawPrice
from app.services.forecast_service import FORECAST_CACHE_PREFIX

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSockets"])

# Validation patterns — same rules as CropPath / MandiPath in schemas.py
_CROP_RE = re.compile(r"^[a-zA-Z][a-zA-Z \-]{0,49}$")
_MANDI_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 \-\.]{0,99}$")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, key: str):
        await websocket.accept()
        if key not in self.active_connections:
            self.active_connections[key] = []
        self.active_connections[key].append(websocket)

    def disconnect(self, websocket: WebSocket, key: str):
        if key in self.active_connections and websocket in self.active_connections[key]:
            self.active_connections[key].remove(websocket)
            if not self.active_connections[key]:
                del self.active_connections[key]

    async def broadcast(self, message: dict, key: str):
        if key in self.active_connections:
            for connection in self.active_connections[key]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


def _is_model_forecast(payload: dict | None) -> bool:
    return bool(payload) and payload.get("forecast_source") == "model"


@router.websocket("/prices/{crop}/{mandi}")
async def websocket_endpoint(websocket: WebSocket, crop: str, mandi: str):
    # ── Validate path params before accepting ────────────────────────────
    if not _CROP_RE.match(crop) or not _MANDI_RE.match(mandi):
        await websocket.close(code=1008, reason="Invalid crop or mandi name")
        return
    key = f"{crop.lower()}:{mandi.lower()}"
    await manager.connect(websocket, key)

    cache_key = f"{FORECAST_CACHE_PREFIX}:{crop.lower()}:{mandi.lower()}:30"
    forecast_data = await cache_get(cache_key)
    if not _is_model_forecast(forecast_data):
        forecast_data = None

    if forecast_data:
        try:
            await websocket.send_json(forecast_data)
        except WebSocketDisconnect:
            manager.disconnect(websocket, key)
            return

    previous_price = forecast_data.get("current_price") if forecast_data else None

    try:
        while True:
            await asyncio.sleep(60)

            from app.database import AsyncSessionLocal

            try:
                async with AsyncSessionLocal() as db:
                    stmt = (
                        select(RawPrice)
                        .where(RawPrice.crop.ilike(crop))
                        .where(RawPrice.raw_data["market_name"].astext.ilike(mandi))
                        .order_by(desc(RawPrice.fetch_date))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    last_row = result.scalars().first()

                    if last_row and last_row.raw_data and last_row.raw_data.get("modal_price"):
                        current_price = float(last_row.raw_data["modal_price"])

                        change_pct = 0.0
                        if previous_price is not None and previous_price > 0:
                            change_pct = ((current_price - previous_price) / previous_price) * 100

                        fresh_forecast = await cache_get(cache_key)
                        if not _is_model_forecast(fresh_forecast):
                            fresh_forecast = None
                        recommendation = fresh_forecast.get("recommendation", "HOLD") if fresh_forecast else "HOLD"

                        if current_price != previous_price:
                            payload = {
                                "crop": crop,
                                "mandi": mandi,
                                "modal_price": current_price,
                                "change_pct": round(change_pct, 2),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "recommendation": recommendation,
                            }
                            await manager.broadcast(payload, key)
                            previous_price = current_price
            except Exception as exc:
                logger.error(f"WS DB query error for {key}: {exc}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, key)
