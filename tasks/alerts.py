import os
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from twilio.rest import Client
import json

from celery_app import app
from database import AsyncSessionLocal, AlertSubscription, User
from api.routes_forecast import get_forecast

logger = logging.getLogger(__name__)

# Template mapping for the Twilio approved outbound template
# Assuming you have a standard template or using the test one provided:
# The user's requested template content_variables are: {"1":"date","2":"price"}
# For a real WhatsApp production app, templates must be pre-approved.

@app.task
def send_daily_alerts():
    """
    Celery task scheduled for 6 AM to check all active alert subscriptions
    and dispatch WhatsApp templates to users whose criteria are met.
    """
    asyncio.run(async_send_daily_alerts())


async def async_send_daily_alerts():
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not auth_token:
        logger.warning("No TWILIO_AUTH_TOKEN found. Cannot send background alerts.")
        return

    client = Client(account_sid, auth_token)
    sent_count = 0

    async with AsyncSessionLocal() as db:
        # Fetch active subscriptions and join User to get phone if user_id exists
        stmt = select(AlertSubscription, User.phone).outerjoin(
            User, AlertSubscription.user_id == User.id
        ).where(AlertSubscription.is_active == True)
        
        result = await db.execute(stmt)
        rows = result.all()
        
        for sub, user_phone in rows:
            # Determine target phone number
            target_phone = sub.phone_number if sub.phone_number else user_phone
            if not target_phone:
                continue
                
            # Ensure proper WhatsApp format
            if not target_phone.startswith("whatsapp:"):
                # if the user phone is just 9876543210, add whatsapp:+91 prefix
                if not target_phone.startswith("+"):
                    target_phone = f"whatsapp:+91{target_phone}"
                else:
                    target_phone = f"whatsapp:{target_phone}"

            try:
                # Get the latest forecast
                f_data = await get_forecast(crop=sub.crop, mandi=sub.mandi, horizon=1)
                latest_forecast = f_data.forecast[0].predicted_price
                current_price = f_data.current_price
                
                # Check trigger condition
                # If threshold is 0.0, it's a daily alert. Else, it must cross threshold.
                trigger = False
                if sub.threshold_price == 0.0:
                    trigger = True
                elif current_price > sub.threshold_price or latest_forecast > sub.threshold_price:
                    trigger = True

                if trigger:
                    # Send Twilio Template! Wait, we use the template given by the user 
                    # as a test which has 2 variables: {"1":"text", "2":"text"}
                    template_sid = 'HX07f5f3a8427e20682e1db46ae7864e7b'
                    
                    variables = {
                        "1": f"{sub.crop.title()} @ {sub.mandi.title()}",
                        "2": f"₹{latest_forecast}/q"
                    }

                    message = client.messages.create(
                        from_='whatsapp:+14155238886',
                        content_sid=template_sid,
                        content_variables=json.dumps(variables),
                        to=target_phone
                    )
                    logger.info(f"Alert sent to {target_phone} (SID: {message.sid})")
                    sent_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process alert for {target_phone}: {e}")

    logger.info(f"Completed sending alerts. Total sent: {sent_count}")
