from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from api.deps import get_db, cache_get
from database import RawPrice

router = APIRouter(prefix="/ws", tags=["WebSockets"])

class ConnectionManager:
    def __init__(self):
        # Maps "crop:mandi" to a list of active websocket connections
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
                except:
                    pass

manager = ConnectionManager()

@router.websocket("/prices/{crop}/{mandi}")
async def websocket_endpoint(websocket: WebSocket, crop: str, mandi: str):
    key = f"{crop.lower()}:{mandi.lower()}"
    await manager.connect(websocket, key)
    
    # ── Initial Send: Latest cached forecast ────────────────────────────
    cache_key = f"forecast:v2:{crop.lower()}:{mandi.lower()}:30"
    forecast_data = await cache_get(cache_key)
    
    if forecast_data:
        try:
            await websocket.send_json(forecast_data)
        except WebSocketDisconnect:
            manager.disconnect(websocket, key)
            return

    previous_price = None
    if forecast_data and "current_price" in forecast_data:
        previous_price = forecast_data["current_price"]

    try:
        while True:
            # Sleep for 60 seconds. We put sleep at the beginning of loop 
            # to avoid immediate duplicate fetch right after initial send
            await asyncio.sleep(60)
            
            # Fetch latest price from database manually to ensure fresh session
            from database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(RawPrice)
                    .where(RawPrice.crop.ilike(crop))
                    .where(RawPrice.raw_data['market_name'].astext.ilike(mandi))
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
                        
                    # Re-fetch forecast to get current recommendation if exists
                    fresh_forecast = await cache_get(cache_key)
                    recommendation = fresh_forecast.get("recommendation", "HOLD") if fresh_forecast else "HOLD"
                    
                    # Prevent redundant emits if price didn't change
                    # But the prompt says "compare modal_price to previous, emit JSON"
                    if current_price != previous_price:
                        payload = {
                            "crop": crop,
                            "mandi": mandi,
                            "modal_price": current_price,
                            "change_pct": round(change_pct, 2),
                            "timestamp": datetime.utcnow().isoformat(),
                            "recommendation": recommendation
                        }
                        await manager.broadcast(payload, key)
                        previous_price = current_price

    except WebSocketDisconnect:
        manager.disconnect(websocket, key)
