"""
api/routes_alerts.py
────────────────────
Farmer price-alert subscription endpoint.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import User, AlertSubscription
from api.schemas import AlertCreate, AlertOut
from api.deps import get_db, get_current_user

router = APIRouter(prefix="/api/v1", tags=["Price Alerts"])


@router.post(
    "/alerts/subscribe",
    response_model=AlertOut,
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe to a crop price alert",
    description=(
        "Register a farmer's phone + crop + mandi + price threshold. "
        "When the predicted or live price exceeds the threshold, "
        "the farmer will receive a WhatsApp / SMS notification.  "
        "**Requires authentication** (Bearer JWT token)."
    ),
)
async def subscribe_alert(
    body: AlertCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    **Subscribe** to a price alert for a specific crop and mandi.

    - **crop**: Target crop (e.g. `wheat`)
    - **mandi**: Target market (e.g. `Azadpur`)
    - **threshold_price**: Alert threshold in ₹/quintal

    The authenticated user's ID is automatically linked to the subscription.
    Returns the created subscription record.
    """
    sub = AlertSubscription(
        user_id=current_user.id,
        crop=body.crop,
        mandi=body.mandi,
        threshold_price=body.threshold_price,
        is_active=True,
    )
    db.add(sub)
    await db.commit()
    await db.refresh(sub)
    return sub
