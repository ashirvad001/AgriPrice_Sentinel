"""
api/whatsapp_bot.py
───────────────────
Core logic for the Twilio WhatsApp bot:
- Redis session management (phone -> language, crop, mandi)
- Structured command parsing (forecast, subscribe, history, help)
- Multi-language support (Hindi, Punjabi, Telugu, Marathi, English)
- Fallback to OpenAI API for free-form NLP queries
"""

from __future__ import annotations

import os
import re
import json
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from twilio.twiml.messaging_response import MessagingResponse
from openai import AsyncOpenAI

from api.routes_forecast import get_forecast
from api.routes_prices import get_prices
from database import AlertSubscription

logger = logging.getLogger(__name__)

# Basic translation map for static responses
LANG_MAP = {
    "english": {
        "help": (
            "🌾 *AgriPrice Sentinel Commands:*\n\n"
            "1️⃣ *{crop} {mandi}*\n"
            "   (e.g., 'wheat Amritsar') -> Get 7-day forecast & advice.\n\n"
            "2️⃣ *subscribe {crop} {mandi}*\n"
            "   -> Get daily 6 AM alerts.\n\n"
            "3️⃣ *history {crop} {mandi}*\n"
            "   -> Get 30-day price history.\n\n"
            "4️⃣ *Ask any question!*\n"
            "   (e.g., 'Should I sell my wheat now?')"
        ),
        "subscribed": "✅ Successfully subscribed to daily {crop} alerts at {mandi}!",
        "history": "📊 Last 30 days for {crop} at {mandi}:\nMin: ₹{min}\nMax: ₹{max}\nAvg: ₹{avg}",
        "forecast": (
            "📈 *Forecast for {crop} at {mandi}*\n"
            "Current Price: ₹{current}\n"
            "MSP: ₹{msp}\n\n"
            "7-Day Avg Expected: ₹{avg}\n"
            "Recommendation: *{rec}*\n"
            "Reason: {reason}"
        ),
        "error": "❌ Sorry, I encountered an error processing your request.",
        "lang_switch": "Language set to {lang}."
    },
    "hindi": {
        "help": (
            "🌾 *AgriPrice Sentinel कमांड्स:*\n\n"
            "1️⃣ *{crop} {mandi}*\n"
            "   (उदा. 'wheat Amritsar') -> 7-दिन का पूर्वानुमान और सलाह।\n\n"
            "2️⃣ *subscribe {crop} {mandi}*\n"
            "   -> रोज़ सुबह 6 बजे अलर्ट पाएं।\n\n"
            "3️⃣ *history {crop} {mandi}*\n"
            "   -> पिछले 30 दिनों का भाव।\n\n"
            "4️⃣ *कोई भी सवाल पूछें!*\n"
            "   (उदा. 'क्या मुझे अब अपना गेहूं बेचना चाहिए?')"
        ),
        "subscribed": "✅ {mandi} में रोज़ाना {crop} अलर्ट के लिए सफलतापूर्वक सब्सक्राइब किया गया!",
        "history": "📊 {mandi} में {crop} के पिछले 30 दिन:\nन्यूनतम: ₹{min}\nअधिकतम: ₹{max}\nऔसत: ₹{avg}",
        "forecast": (
            "📈 *{mandi} में {crop} का पूर्वानुमान*\n"
            "वर्तमान भाव: ₹{current}\n"
            "MSP: ₹{msp}\n\n"
            "7-दिन का अपेक्षित औसत: ₹{avg}\n"
            "सलाह: *{rec}*\n"
            "कारण: {reason}"
        ),
        "error": "❌ क्षमा करें, आपके अनुरोध को प्रोसेस करने में कोई त्रुटि हुई।",
        "lang_switch": "भाषा {lang} में सेट की गई।"
    },
    "punjabi": {
        "help": "🌾 *AgriPrice Sentinel:* Send 'help' or ask a question.",
        "subscribed": "✅ {mandi} ਵਿੱਚ ਰੋਜ਼ਾਨਾ {crop} ਅਲਰਟ ਲਈ ਸਬਸਕ੍ਰਾਈਬ ਕੀਤਾ ਗਿਆ!",
        "history": "📊 {mandi} ਵਿੱਚ {crop} ਦੇ ਪਿਛਲੇ 30 ਦਿਨ:\nMin: ₹{min}\nMax: ₹{max}\nAvg: ₹{avg}",
        "forecast": "📈 *{mandi} ਵਿੱਚ {crop} ਦਾ ਪੂਰਵ ਅਨੁਮਾਨ*\nभाव: ₹{current}\nMSP: ₹{msp}\nਸਲਾਹ: *{rec}*",
        "error": "❌ ਮੁਆਫ ਕਰਨਾ, ਕੋਈ ਗਲਤੀ ਆਈ ਹੈ।",
        "lang_switch": "ਭਾਸ਼ਾ {lang} ਸੈੱਟ ਕੀਤੀ ਗਈ।"
    },
    "telugu": {
        "help": "🌾 *AgriPrice Sentinel:* Send 'help' or ask a question.",
        "subscribed": "✅ {mandi} అలెర్ట్‌ల కోసం సబ్‌స్క్రైబ్ చేయబడింది!",
        "history": "📊 {mandi} లో {crop}:\nMin: ₹{min}\nMax: ₹{max}\nAvg: ₹{avg}",
        "forecast": "📈 *{mandi} లో {crop} అంచనా*\nధర: ₹{current}\nMSP: ₹{msp}\nసలహా: *{rec}*",
        "error": "❌ క్షమించండి, ఒక లోపం ఏర్పడింది.",
        "lang_switch": "భాష {lang} కి మార్చబడింది."
    },
    "marathi": {
        "help": "🌾 *AgriPrice Sentinel:* Send 'help' or ask a question.",
        "subscribed": "✅ {mandi} मधील {crop} अलर्टसाठी सबस्क्राइब केले!",
        "history": "📊 {mandi} मधील {crop}:\nMin: ₹{min}\nMax: ₹{max}\nAvg: ₹{avg}",
        "forecast": "📈 *{mandi} मधील {crop} अंदाज*\nभाव: ₹{current}\nMSP: ₹{msp}\nसल्ला: *{rec}*",
        "error": "❌ क्षमस्व, एक त्रुटी आली.",
        "lang_switch": "भाषा {lang} वर सेट केली."
    }
}

VALID_LANGS = ["english", "hindi", "punjabi", "telugu", "marathi"]


async def get_nlp_fallback(query: str, session: dict) -> str:
    """Uses OpenAI to answer agricultural pricing queries."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return f"🤖 (NLP Mock): I understand you're asking about '{query}'. Please use precise commands like 'wheat Amritsar' for now."

    client = AsyncOpenAI(api_key=openai_key)
    lang = session.get("lang", "english")
    
    prompt = (
        f"You are AgriPrice Sentinel, a helpful WhatsApp assistant for Indian farmers. "
        f"Answer the following query concisely (max 3 sentences) in {lang.title()}. "
        f"Query: {query}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return LANG_MAP[lang]["error"]


async def process_whatsapp_message(
    body: str,
    from_number: str,
    redis: Redis,
    db: AsyncSession
) -> str:
    """
    Main entry point for incoming WhatsApp messages.
    Returns a TwiML string.
    """
    # 1. Session Management
    session_key = f"wa_session:{from_number}"
    raw_session = await redis.get(session_key) if redis else None
    session: dict[str, str] = json.loads(raw_session) if raw_session else {"lang": "english"}
    lang = session["lang"]

    resp = MessagingResponse()
    body_clean = body.strip().lower()
    tokens = body_clean.split()

    if not tokens:
        resp.message(LANG_MAP[lang]["help"])
        return str(resp)

    # ── Command: Language Switch ──
    if len(tokens) == 2 and tokens[0] == "language" and tokens[1] in VALID_LANGS:
        session["lang"] = tokens[1]
        if redis:
            await redis.set(session_key, json.dumps(session), ex=86400 * 30) # 30 days
        resp.message(LANG_MAP[session["lang"]]["lang_switch"].format(lang=tokens[1].title()))
        return str(resp)

    # ── Command: Help ──
    if body_clean in ["help", "hi", "hello", "मदद"]:
        resp.message(LANG_MAP[lang]["help"])
        return str(resp)

    try:
        # ── Command: Subscribe ──
        if tokens[0] == "subscribe" and len(tokens) >= 3:
            crop = tokens[1]
            mandi = " ".join(tokens[2:])
            
            # Save to db
            sub = AlertSubscription(
                phone_number=from_number.replace("whatsapp:", ""),
                crop=crop,
                mandi=mandi,
                threshold_price=0.0, # 0 implies daily alert
                language=lang.title()
            )
            db.add(sub)
            await db.commit()
            
            resp.message(LANG_MAP[lang]["subscribed"].format(crop=crop.title(), mandi=mandi.title()))
            
            # Update session
            session["crop"] = crop
            session["mandi"] = mandi
            if redis:
                await redis.set(session_key, json.dumps(session), ex=86400 * 30)
            return str(resp)

        # ── Command: History ──
        if tokens[0] == "history" and len(tokens) >= 3:
            crop = tokens[1]
            mandi = " ".join(tokens[2:])
            history_data = await get_prices(crop=crop, mandi=mandi, days=30, db=db)
            
            if not history_data.prices:
                prices_resp = "No data found."
            else:
                p_list = [p.modal_price for p in history_data.prices if p.modal_price is not None]
                if p_list:
                    prices_resp = LANG_MAP[lang]["history"].format(
                        crop=crop.title(), mandi=mandi.title(),
                        min=min(p_list), max=max(p_list), avg=round(sum(p_list)/len(p_list))
                    )
                else:
                    prices_resp = "No modal prices recorded."
                    
            resp.message(prices_resp)
            return str(resp)

        # ── Command: Forecast (e.g. "wheat amritsar") ──
        if len(tokens) >= 2 and tokens[0] not in ["subscribe", "history", "help", "language"]:
            # Let's see if first token is a known crop (very naive check, better to have a list)
            crop = tokens[0]
            mandi = " ".join(tokens[1:])
            
            # Use out 7-day forecast logic from router
            f_data = await get_forecast(crop=crop, mandi=mandi, horizon=7)
            
            rec = "वेचें (SELL)" if f_data.recommendation == "SELL" else "रोकें (HOLD)"
            
            msg = LANG_MAP[lang]["forecast"].format(
                crop=crop.title(),
                mandi=mandi.title(),
                current=f_data.current_price,
                msp=f_data.msp or "N/A",
                avg=f_data.avg_predicted_price,
                rec=f_data.recommendation, # keep EN for punjabi/telugu/marathi for now or map it
                reason=f_data.recommendation_reason
            )
            resp.message(msg)
            
            # Update session context
            session["crop"] = crop
            session["mandi"] = mandi
            if redis:
                await redis.set(session_key, json.dumps(session), ex=86400 * 30)
            return str(resp)

        # ── Fallback: NLP ──
        # If it didn't match anything above, pass to LLM
        nlp_response = await get_nlp_fallback(body, session)
        resp.message(nlp_response)
        return str(resp)

    except Exception as e:
        logger.error(f"WhatsApp Bot Error: {e}", exc_info=True)
        resp.message(LANG_MAP[lang]["error"])
        return str(resp)
