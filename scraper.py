"""
scraper.py
──────────
Daily crop price scraper using the Agmarknet data.gov.in API.

Scheduled by APScheduler (main.py) at 6:00 AM IST daily.
Fetches commodity prices for 16 crops × 28 states, stores in raw_prices table.
"""

from __future__ import annotations
import os
import asyncio
import logging
import aiohttp
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal, RawPrice, ScrapeError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("scraper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ── 16 Crops ─────────────────────────────────────────────────────────────────
CROPS = [
    "Wheat", "Rice", "Maize", "Moong", "Urad", "Tur", "Gram", "Bajra",
    "Jowar", "Mustard", "Soybean", "Groundnut", "Cotton", "Sugarcane",
    "Onion", "Potato",
]

# ── 28 States ────────────────────────────────────────────────────────────────
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
]

# ── Agmarknet (data.gov.in) endpoint ────────────────────────────────────────
AGMARKNET_API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
DATAGOV_API_KEY = os.getenv("DATAGOV_API_KEY", "")


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_api_data(session: aiohttp.ClientSession, crop: str, state: str) -> list[dict]:
    """
    Fetch commodity price records from the Agmarknet data.gov.in API.

    Returns a list of parsed records, each containing:
    commodity, market, state, min_price, max_price, modal_price, arrival_date.
    """
    if not DATAGOV_API_KEY:
        raise ValueError("DATAGOV_API_KEY env var is not set — cannot call data.gov.in API")

    params = {
        "api-key": DATAGOV_API_KEY,
        "format": "json",
        "filters[commodity]": crop,
        "filters[state]": state,
        "limit": 100,
        "offset": 0,
    }

    async with session.get(
        AGMARKNET_API_URL,
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as response:
        response.raise_for_status()
        payload = await response.json()

    # data.gov.in wraps records under a "records" key
    records = payload.get("records", [])
    if not records:
        logger.warning(f"API returned 0 records for {crop}/{state}")
    return records


def _parse_price(value) -> float | None:
    """Safely parse a price string to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_arrival_date(date_str: str) -> datetime | None:
    """Parse arrival_date from Agmarknet (DD/MM/YYYY format)."""
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


async def process_crop_state(
    http_session: aiohttp.ClientSession,
    db_session: AsyncSession,
    crop: str,
    state: str,
):
    """Fetch Agmarknet data for a crop/state pair, parse, and save to DB."""
    try:
        records = await fetch_api_data(http_session, crop, state)

        if not records:
            # Log a skip — no data available
            logger.info(f"No records returned for {crop}/{state} — skipping")
            return

        inserted = 0
        for rec in records:
            # Parse the arrival date → use as fetch_date
            arrival_dt = _parse_arrival_date(rec.get("arrival_date", ""))
            fetch_date = arrival_dt.date() if arrival_dt else datetime.today().date()

            # Build raw_data dict matching the schema expected by feature_engineering
            raw_data = {
                "commodity": rec.get("commodity", crop),
                "market": rec.get("market", ""),
                "state": rec.get("state", state),
                "district": rec.get("district", ""),
                "min_price": _parse_price(rec.get("min_price")),
                "max_price": _parse_price(rec.get("max_price")),
                "modal_price": _parse_price(rec.get("modal_price")),
                "arrival_date": rec.get("arrival_date", ""),
                "arrivals_tonnes": _parse_price(rec.get("arrivals", None)),
            }

            raw_price = RawPrice(
                crop=crop,
                state=state,
                fetch_date=fetch_date,
                raw_data=raw_data,
            )
            db_session.add(raw_price)
            inserted += 1

        logger.info(f"Inserted {inserted} records for {crop}/{state}")

    except Exception as e:
        logger.error(f"Failed to fetch {crop} for {state}: {e}")
        scrape_error = ScrapeError(
            crop=crop,
            state=state,
            error_message=str(e),
            failed_at=datetime.utcnow(),
        )
        db_session.add(scrape_error)


async def run_scraper():
    """Main orchestrator for scraping all crops across all states."""
    logger.info(f"Starting Agmarknet data collection at {datetime.now()}")

    if not DATAGOV_API_KEY:
        logger.error("DATAGOV_API_KEY is not set! Add it to your .env file.")
        logger.error("Get a free key at: https://data.gov.in/")
        return

    async with aiohttp.ClientSession() as http_session:
        async with AsyncSessionLocal() as db_session:
            tasks = []

            # Limit to 10 concurrent API requests to avoid rate-limiting
            sem = asyncio.Semaphore(10)

            async def sem_task(crop, state):
                async with sem:
                    await process_crop_state(http_session, db_session, crop, state)

            for crop in CROPS:
                for state in STATES:
                    tasks.append(sem_task(crop, state))

            await asyncio.gather(*tasks)
            await db_session.commit()

    logger.info(f"Finished Agmarknet data collection at {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(run_scraper())
