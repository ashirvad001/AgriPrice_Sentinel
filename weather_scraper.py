from __future__ import annotations
import asyncio
import aiohttp
from datetime import datetime, timedelta, date
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import AsyncSessionLocal, WeatherObservation, ScrapeError

# 50 Major Agricultural Districts Placeholder
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
    "Raichur", "Agra", "Aligarh", "Mathura", "Hathras"
]

# API endpoint placeholder
IMD_API_URL = "https://imd.gov.in/api/v1/weather/district"

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def fetch_weather_api(session: aiohttp.ClientSession, district: str) -> dict:  # type: ignore[return]
    params = {
        "district": district,
        "date": datetime.today().strftime("%Y-%m-%d")
    }
    async with session.get(IMD_API_URL, params=params, timeout=10) as response:
        response.raise_for_status()
        return await response.json()

async def forward_fill_missing(db_session: AsyncSession, district: str, current_date: date, api_data: dict) -> dict:
    """
    Looks backward up to 3 days to fill missing weather metrics.
    """
    metrics = ["rainfall_mm", "max_temp", "min_temp", "humidity", "wind_speed"]
    filled_data = {m: api_data.get(m) for m in metrics}
    
    # Check if anything is missing
    missing_metrics = [m for m, v in filled_data.items() if v is None]
    if not missing_metrics:
        return filled_data # Nothing to fill
    
    # Query last 3 days
    start_date = current_date - timedelta(days=3)
    stmt = select(WeatherObservation).where(
        WeatherObservation.district == district,
        WeatherObservation.date >= start_date,
        WeatherObservation.date < current_date
    ).order_by(WeatherObservation.date.desc())
    
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

async def process_district_weather(http_session: aiohttp.ClientSession, db_session: AsyncSession, district: str):
    """Fetch IMD api, forward fill, and save to DB."""
    current_date = datetime.today().date()
    try:
        api_data = await fetch_weather_api(http_session, district)
        filled_data = await forward_fill_missing(db_session, district, current_date, api_data)
        
        # Check if it already exists (to update or create)
        stmt = select(WeatherObservation).where(
            WeatherObservation.district == district,
            WeatherObservation.date == current_date
        )
        result = await db_session.execute(stmt)
        existing_obs = result.scalar_one_or_none()
        
        if existing_obs:
            for k, v in filled_data.items():
                setattr(existing_obs, k, v)
        else:
            new_obs = WeatherObservation(
                district=district,
                date=current_date,
                **filled_data
            )
            db_session.add(new_obs)
    
    except Exception as e:
        print(f"Failed to fetch weather for {district}: {e}")
        scrape_error = ScrapeError(
            crop="WEATHER_API", # Repurpose crop field to identify the source
            state=district,
            error_message=str(e),
            failed_at=datetime.utcnow()
        )
        db_session.add(scrape_error)

async def run_weather_scraper():
    """Main orchestrator for scraping weather all districts."""
    print(f"Starting IMD Weather collection at {datetime.now()}")
    
    async with aiohttp.ClientSession() as http_session:
        async with AsyncSessionLocal() as db_session:
            tasks = []
            sem = asyncio.Semaphore(10) # 10 Concurrent Requests maximum
            
            async def sem_task(district):
                async with sem:
                    await process_district_weather(http_session, db_session, district)

            for district in DISTRICTS:
                tasks.append(sem_task(district))
            
            await asyncio.gather(*tasks)
            await db_session.commit()
            
    print(f"Finished IMD Weather collection at {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(run_weather_scraper())
