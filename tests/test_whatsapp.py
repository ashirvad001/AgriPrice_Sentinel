import os
from twilio.rest import Client

# 1. Twilio Sandbox config
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')


try:
    client = Client(account_sid, auth_token)

    # 2. Outbound message using Twilio Sandbox template
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        content_sid='HXb5b62575e6e4ff6129ad7c8efe1f983e',
        content_variables='{"1":"12/1","2":"3pm"}',
        to='whatsapp:+916287967654'
    )

    print(f"✅ Successfully sent outbound WhatsApp message!")
    print(f"Message SID: {message.sid}")
    print(f"Status: {message.status}")

except Exception as e:
    print(f"❌ Failed to send message.\nError: {e}")
