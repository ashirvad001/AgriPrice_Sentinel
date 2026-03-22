"""
api/routes_whatsapp.py
──────────────────────
Twilio webhook endpoint for WhatsApp bot.
"""

from fastapi import APIRouter, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import Response

from api.deps import get_db, get_redis
from api.whatsapp_bot import process_whatsapp_message

router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp Bot"])


@router.post("/webhook")
async def twilio_webhook(
    Body: str = Form(""),
    From: str = Form(""),
    db: AsyncSession = Depends(get_db)
):
    """
    Twilio webhook endpoint for incoming WhatsApp messages.
    """
    # Get Redis instance
    redis_pool = get_redis()
    
    try:
        # Process message and get TwiML XML response
        twiml_response = await process_whatsapp_message(
            body=Body,
            from_number=From,
            redis=redis_pool,
            db=db
        )
        # Twilio expects application/xml
        return Response(content=twiml_response, media_type="application/xml")
    except Exception as e:
        import traceback
        return Response(content=traceback.format_exc(), media_type="text/plain", status_code=500)
