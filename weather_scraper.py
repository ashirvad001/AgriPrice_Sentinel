"""
weather_scraper.py
──────────────────
Daily weather data scraper using the OpenWeatherMap API.

Scheduled by APScheduler (main.py) at 6:15 AM IST daily.
Fetches current weather for 50 major agricultural districts,
forward-fills missing metrics from the last 3 days, and stores
in the weather_obs table.
"""

from __future__ import annotations
import os
import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta, date
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import AsyncSessionLocal, WeatherObservation, ScrapeError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("weather_scraper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ── 50 Major Agricultural Districts ─────────────────────────────────────────
DISTRICTS = [
    "Karnal", "Kurukshetra", "Ludhiana", "Patiala", "Amritsar",
    "Bhatinda", "Sangrur", "Ganganagar", "Hanumangarh", "Bikaner",
    "Jodhpur", "Ahmedabad", "Rajkot", "Surat", "Bhavnagar",
    "Junagadh", "Nashik", "Jalgaon", "Pune", "Solapur",
    "Ahmednagar", "Indore", "Ujjain", "Dewas", "Sehore",
    "Hoshangabad", "Coimbatore", "Erode", "Madurai", "Salem",
    "Tiruppur", "Guntur", "Krishna", "Prakasam", "Kurnool",
    "West Godavari", "Nizamabad", "Karimnagar", "Warangal", "Nalgonda",
    "Khammam", "Hubli", "Belagavi", "Dharwad", "Vijayapura",
    "Raichur", "Agra", "Aligarh", "Mathura", "Hathras",
]

# ── OpenWeatherMap API ──────────────────────────────────────────────────────
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_weather_api(session: aiohttp.ClientSession, district: str) -> dict:
    """
    Fetch current weather from OpenWeatherMap for a given Indian district.

    Returns a dict with keys: rainfall_mm, max_temp, min_temp, humidity, wind_speed.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY env var is not set")

    params = {
        "q": f"{district},IN",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    async with session.get(
        OPENWEATHER_API_URL,
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as response:
        response.raise_for_status()
        data = await response.json()

    # Map OpenWeatherMap response to our schema
    main = data.get("main", {})
    wind = data.get("wind", {})
    rain = data.get("rain", {})

    return {
        "max_temp": main.get("temp_max"),
        "min_temp": main.get("temp_min"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed"),
        "rainfall_mm": rain.get("1h", 0.0),  # mm in last 1h, default 0
    }


async def forward_fill_missing(
    db_session: AsyncSession,
    district: str,
    current_date: date,
    api_data: dict,
) -> dict:
    """
    Looks backward up to 3 days to fill missing weather metrics.
    """
    metrics = ["rainfall_mm", "max_temp", "min_temp", "humidity", "wind_speed"]
    filled_data = {m: api_data.get(m) for m in metrics}

    # Check if anything is missing
    missing_metrics = [m for m, v in filled_data.items() if v is None]
    if not missing_metrics:
        return filled_data  # Nothing to fill

    # Query last 3 days
    start_date = current_date - timedelta(days=3)
    stmt = (
        select(WeatherObservation)
        .where(
            WeatherObservation.district == district,
            WeatherObservation.date >= start_date,
            WeatherObservation.date < current_date,
        )
        .order_by(WeatherObservation.date.desc())
    )

    result = await db_session.execute(stmt)
    historical_obs = result.scalars().all()

    # Fill backwards
    for obs in historical_obs:
        still_missing = []
        for m in missing_metrics:
            historical_val = getattr(obs, m)
            if historical_val is not None:
                filled_data[m] = historical_val
            else:
                still_missing.append(m)
        missing_metrics = still_missing
        if not missing_metrics:
            break

    return filled_data


async def process_district_weather(
    http_session: aiohttp.ClientSession,
    db_session: AsyncSession,
    district: str,
):
    """Fetch OpenWeatherMap data, forward fill, and save to DB."""
    current_date = datetime.today().date()
    try:
        api_data = await fetch_weather_api(http_session, district)
        filled_data = await forward_fill_missing(db_session, district, current_date, api_data)

        # Check if it already exists (upsert logic)
        stmt = select(WeatherObservation).where(
            WeatherObservation.district == district,
            WeatherObservation.date == current_date,
        )
        result = await db_session.execute(stmt)
        existing_obs = result.scalar_one_or_none()

        if existing_obs:
            for k, v in filled_data.items():
                setattr(existing_obs, k, v)
            logger.debug(f"Updated weather for {district}")
        else:
            new_obs = WeatherObservation(
                district=district,
                date=current_date,
                **filled_data,
            )
            db_session.add(new_obs)
            logger.debug(f"Inserted weather for {district}")

    except Exception as e:
        logger.error(f"Failed to fetch weather for {district}: {e}")
        scrape_error = ScrapeError(
            crop="WEATHER_API",
            state=district,
            error_message=str(e),
            failed_at=datetime.utcnow(),
        )
        db_session.add(scrape_error)


async def run_weather_scraper():
    """Main orchestrator for scraping weather for all districts."""
    logger.info(f"Starting OpenWeatherMap collection at {datetime.now()}")

    if not OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY is not set! Add it to your .env file.")
        logger.error("Get a free key at: https://openweathermap.org/api")
        return

    async with aiohttp.ClientSession() as http_session:
        async with AsyncSessionLocal() as db_session:
            tasks = []
            sem = asyncio.Semaphore(10)  # 10 concurrent requests max

            async def sem_task(district):
                async with sem:
                    await process_district_weather(http_session, db_session, district)

            for district in DISTRICTS:
                tasks.append(sem_task(district))

            await asyncio.gather(*tasks)
            await db_session.commit()

    logger.info(f"Finished OpenWeatherMap collection at {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(run_weather_scraper())
