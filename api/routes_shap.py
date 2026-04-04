from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List

from api.deps import get_db
from database import ShapExplanation
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["SHAP"])

class ShapFeatureResponse(BaseModel):
    feature_name: str
    shap_value: float
    farmer_label: str
    rank: int

@router.get("/shap/{crop}", response_model=List[ShapFeatureResponse])
async def get_shap_features(crop: str, db: AsyncSession = Depends(get_db)):
    """
    Get top 10 SHAP feature explanations for a given crop.
    Returns the most recent available prediction date's SHAP values.
    """
    # 1. Find the most recent date with SHAP data for this crop
    latest_date_stmt = (
        select(ShapExplanation.prediction_date)
        .where(ShapExplanation.crop.ilike(crop))
        .order_by(desc(ShapExplanation.prediction_date))
        .limit(1)
    )
    result = await db.execute(latest_date_stmt)
    latest_date = result.scalar_one_or_none()
    
    if not latest_date:
        # If no SHAP data, return empty list instead of 404
        return []

    # 2. Get top 10 features for that date
    stmt = (
        select(ShapExplanation)
        .where(
            ShapExplanation.crop.ilike(crop),
            ShapExplanation.prediction_date == latest_date
        )
        .order_by(ShapExplanation.rank.asc())
        .limit(10)
    )
    
    result = await db.execute(stmt)
    features = result.scalars().all()
    
    return [
        ShapFeatureResponse(
            feature_name=f.feature_name,
            shap_value=f.shap_value,
            farmer_label=f.farmer_label,
            rank=f.rank
        ) for f in features
    ]
