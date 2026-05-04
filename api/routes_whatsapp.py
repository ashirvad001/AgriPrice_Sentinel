"""
api/routes_whatsapp.py
──────────────────────
Twilio webhook endpoint for WhatsApp bot.
Includes Twilio signature validation to prevent spoofed requests.
"""

import os
import logging
from fastapi import APIRouter, Depends, Form, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import Response
from twilio.request_validator import RequestValidator

from api.deps import get_db, get_redis
from api.whatsapp_bot import process_whatsapp_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp Bot"])

# ── Twilio signature validator ───────────────────────────────────────────────
_TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")


def _validate_twilio_signature(request: Request, form_data: dict) -> bool:
    """Validate that the incoming request genuinely came from Twilio."""
    if not _TWILIO_AUTH_TOKEN:
        logger.warning("TWILIO_AUTH_TOKEN not set — skipping signature validation")
        return True  # Allow in development when no token is set

    validator = RequestValidator(_TWILIO_AUTH_TOKEN)
    signature = request.headers.get("X-Twilio-Signature", "")
    # Reconstruct the full URL that Twilio signed against
    url = str(request.url)
    return validator.validate(url, form_data, signature)


@router.post("/webhook")
async def twilio_webhook(
    request: Request,
    Body: str = Form(""),
    From: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """
    Twilio webhook endpoint for incoming WhatsApp messages.
    Validates the X-Twilio-Signature before processing.
    """
    # ── 1. Validate Twilio signature ─────────────────────────────────────
    form_data = {"Body": Body, "From": From}
    if not _validate_twilio_signature(request, form_data):
        logger.warning(f"Invalid Twilio signature from {From}")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # ── 2. Process message ───────────────────────────────────────────────
    redis_pool = get_redis()

    try:
        twiml_response = await process_whatsapp_message(
            body=Body,
            from_number=From,
            redis=redis_pool,
            db=db,
        )
        return Response(content=twiml_response, media_type="application/xml")
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}", exc_info=True)
        return Response(
            content="<Response><Message>Internal error</Message></Response>",
            media_type="application/xml",
            status_code=500,
        )
