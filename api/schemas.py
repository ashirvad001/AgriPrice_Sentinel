"""
api/schemas.py
──────────────
Pydantic v2 request / response models for the AgriPrice Sentinel API.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════════════════
class UserCreate(BaseModel):
    """Register a new farmer account."""
    model_config = ConfigDict(json_schema_extra={
        "example": {"phone": "9876543210", "password": "securepass123", "full_name": "Ramesh Kumar"}
    })
    phone: str = Field(..., min_length=10, max_length=15, description="Farmer's mobile number")
    password: str = Field(..., min_length=6, description="Account password")
    full_name: Optional[str] = Field(None, description="Farmer's full name")


class UserLogin(BaseModel):
    """Login with phone + password."""
    model_config = ConfigDict(json_schema_extra={
        "example": {"phone": "9876543210", "password": "securepass123"}
    })
    phone: str = Field(..., description="Registered mobile number")
    password: str = Field(..., description="Account password")


class TokenResponse(BaseModel):
    """JWT token returned on successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")


class UserOut(BaseModel):
    """Public user profile."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    phone: str
    full_name: Optional[str] = None
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
#  FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
class ForecastDay(BaseModel):
    """Single day in the forecast horizon."""
    date: date
    predicted_price: float = Field(description="Predicted modal price (₹/quintal)")
    lower_bound: float = Field(description="95% CI lower bound")
    upper_bound: float = Field(description="95% CI upper bound")


class ForecastResponse(BaseModel):
    """Complete forecast payload with MSP comparison and recommendation."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "crop": "Wheat",
            "mandi": "Azadpur",
            "horizon_days": 30,
            "msp": 2275.0,
            "recommendation": "SELL",
            "recommendation_reason": "Predicted avg price ₹2,410 is 5.9% above MSP",
            "forecast": [],
        }
    })
    crop: str
    mandi: str
    horizon_days: int
    current_price: Optional[float] = None
    msp: Optional[float] = Field(None, description="Current MSP for this crop (₹/quintal)")
    avg_predicted_price: float = Field(description="Average predicted price over the horizon")
    recommendation: str = Field(description="SELL or HOLD based on MSP comparison")
    recommendation_reason: str
    forecast: list[ForecastDay]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
#  HISTORICAL PRICES
# ═══════════════════════════════════════════════════════════════════════════════
class PriceRecord(BaseModel):
    """Single historical price entry."""
    model_config = ConfigDict(from_attributes=True)
    date: date
    modal_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    mandi: Optional[str] = None


class PriceHistoryResponse(BaseModel):
    """Historical price series response."""
    crop: str
    mandi: str
    days_requested: int
    total_records: int
    prices: list[PriceRecord]


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
class AlertCreate(BaseModel):
    """Subscribe to a price alert."""
    model_config = ConfigDict(json_schema_extra={
        "example": {"crop": "Wheat", "mandi": "Azadpur", "threshold_price": 2500.0}
    })
    crop: str = Field(..., description="Crop name to monitor")
    mandi: str = Field(..., description="Mandi / market name")
    threshold_price: float = Field(..., gt=0, description="Alert when price exceeds this (₹/quintal)")


class AlertOut(BaseModel):
    """Subscription confirmation."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    crop: str
    mandi: str
    threshold_price: float
    is_active: bool
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC
# ═══════════════════════════════════════════════════════════════════════════════
class MessageResponse(BaseModel):
    """Generic message wrapper."""
    message: str
    detail: Optional[str] = None
