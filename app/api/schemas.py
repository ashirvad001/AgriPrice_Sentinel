"""
api/schemas.py
--------------
Pydantic v2 request / response models for the AgriPrice Sentinel API.

Hardened with:
  - Constrained string types (CropName, MandiName) with length + regex limits
  - extra="forbid" on all request bodies to reject unknown fields
  - Phone regex patterns, password length caps, price upper bounds
  - Re-usable Path() annotations for route parameters
"""

from __future__ import annotations

from datetime import date as datetime_date, datetime, timezone
from typing import Annotated, Optional

from fastapi import Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


# ═══════════════════════════════════════════════════════════════════════════════
#  Constrained String Types
# ═══════════════════════════════════════════════════════════════════════════════

# Letters, spaces, and hyphens only — e.g. "Wheat", "Tur Dal", "bajra"
CropName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z][a-zA-Z \-]*$",
        strip_whitespace=True,
    ),
]

# Letters, digits, spaces, hyphens, dots — e.g. "Azadpur", "New Delhi", "S.A.S. Nagar"
MandiName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9 \-\.]*$",
        strip_whitespace=True,
    ),
]

# Digits only, 10–15 characters — Indian mobile numbers
PhoneNumber = Annotated[
    str,
    StringConstraints(
        min_length=10,
        max_length=15,
        pattern=r"^\d{10,15}$",
        strip_whitespace=True,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI Path / Query parameter annotations for route signatures
# ═══════════════════════════════════════════════════════════════════════════════

CropPath = Annotated[
    str,
    Path(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z][a-zA-Z \-]*$",
        description="Crop name (letters, spaces, hyphens only)",
        examples=["wheat", "rice", "Tur Dal"],
    ),
]

MandiPath = Annotated[
    str,
    Path(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9 \-\.]*$",
        description="Mandi / market name",
        examples=["Azadpur", "Karnal", "New Delhi"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Request Models
# ═══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """Register a new farmer account."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"phone": "9876543210", "password": "securepass123", "full_name": "Ramesh Kumar"}
        },
    )
    phone: PhoneNumber = Field(..., description="Farmer's mobile number (10–15 digits)")
    password: str = Field(..., min_length=6, max_length=128, description="Account password")
    full_name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Farmer's full name"
    )


class UserLogin(BaseModel):
    """Login with phone + password."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"phone": "9876543210", "password": "securepass123"}},
    )
    phone: PhoneNumber = Field(..., description="Registered mobile number (10–15 digits)")
    password: str = Field(..., min_length=1, max_length=128, description="Account password")


class AlertCreate(BaseModel):
    """Subscribe to a price alert."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"crop": "Wheat", "mandi": "Azadpur", "threshold_price": 2500.0}},
    )
    crop: CropName = Field(..., description="Crop name to monitor")
    mandi: MandiName = Field(..., description="Mandi / market name")
    threshold_price: float = Field(
        ..., gt=0, le=1_000_000, description="Alert when price exceeds this (Rs./quintal)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class TokenResponse(BaseModel):
    """JWT token returned on successful login."""

    access_token: str = Field(..., description="JWT access token string")
    token_type: str = Field("bearer", description="Token type (bearer)")
    expires_in: int = Field(description="Token lifetime in seconds")


class UserOut(BaseModel):
    """Public user profile."""

    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Unique user ID")
    phone: str = Field(..., description="Registered mobile number")
    full_name: Optional[str] = Field(None, description="Farmer's full name")
    created_at: datetime = Field(..., description="Account creation timestamp")


class ForecastDay(BaseModel):
    """Single day in the forecast horizon."""

    date: datetime_date
    predicted_price: float = Field(description="Predicted modal price (Rs./quintal)")
    lower_bound: float = Field(description="95% CI lower bound")
    upper_bound: float = Field(description="95% CI upper bound")


class ForecastResponse(BaseModel):
    """Complete forecast payload with MSP comparison and recommendation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "crop": "Wheat",
                "mandi": "Azadpur",
                "horizon_days": 30,
                "msp": 2275.0,
                "recommendation": "SELL",
                "recommendation_reason": "Predicted avg price Rs.2,410 is 5.9% above MSP",
                "forecast": [],
            }
        }
    )
    crop: str = Field(..., description="Name of the crop")
    mandi: str = Field(..., description="Name of the mandi (market)")
    horizon_days: int = Field(..., description="Number of days forecasted")
    current_price: Optional[float] = Field(None, description="Current modal price (Rs./quintal)")
    msp: Optional[float] = Field(None, description="Current MSP for this crop (Rs./quintal)")
    avg_predicted_price: float = Field(..., description="Average predicted price over the horizon")
    recommendation: str = Field(..., description="SELL or HOLD based on MSP comparison")
    recommendation_reason: str = Field(..., description="Explanation for the recommendation")
    forecast_source: str = Field(
        ...,
        description="Source of the forecast. Real endpoint responses must come from the trained model.",
    )
    forecast: list[ForecastDay] = Field(..., description="Daily price predictions with confidence intervals")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of forecast generation",
    )


class PriceRecord(BaseModel):
    """Single historical price entry."""

    model_config = ConfigDict(from_attributes=True)
    date: datetime_date = Field(..., description="Date of the record")
    modal_price: Optional[float] = Field(None, description="Modal price (Rs./quintal)")
    min_price: Optional[float] = Field(None, description="Minimum price (Rs./quintal)")
    max_price: Optional[float] = Field(None, description="Maximum price (Rs./quintal)")
    mandi: Optional[str] = Field(None, description="Market name")


class PriceHistoryResponse(BaseModel):
    """Historical price series response."""

    crop: str = Field(..., description="Crop name")
    mandi: str = Field(..., description="Mandi name")
    days_requested: int = Field(..., description="Number of days requested in history")
    total_records: int = Field(..., description="Number of actual records returned")
    prices: list[PriceRecord] = Field(..., description="Ordered list of daily price records")


class AlertOut(BaseModel):
    """Subscription confirmation."""

    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Alert ID")
    crop: str = Field(..., description="Monitored crop")
    mandi: str = Field(..., description="Monitored mandi")
    threshold_price: float = Field(..., description="Trigger price threshold")
    is_active: bool = Field(..., description="Whether the alert is currently active")
    created_at: datetime = Field(..., description="Subscription creation time")


class MessageResponse(BaseModel):
    """Generic message wrapper."""

    message: str = Field(..., description="Status message")
    detail: Optional[str] = Field(None, description="Optional detailed error or status info")
