from __future__ import annotations
import asyncio
import aiohttp
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal, RawPrice, ScrapeError

# 16 Crops
CROPS = [
    "wheat", "rice", "maixze", "moong", "urad", "tur", "chana", "bajra",
    "jowar", "mustard", "soybean", "groundnut", "cotton", "sugarcane",
    "onion", "potato"
]

# 28 States
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

# API endpoint placeholder (adjust based on actual e-NAM API documentation)
ENAM_API_URL = "https://enam.gov.in/api/v1/prices"

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def fetch_api_data(session: aiohttp.ClientSession, crop: str, state: str) -> dict:  # type: ignore[return]
    """Fetch data with exponential backoff for a specific state and crop."""
    # Assuming params are sent as query params. Change as per actual API specs.
    params = {
        "crop": crop,
        "state": state,
        "date": datetime.today().strftime("%Y-%m-%d")
    }
    async with session.get(ENAM_API_URL, params=params, timeout=10) as response:
        response.raise_for_status()
        return await response.json()

async def process_crop_state(http_session: aiohttp.ClientSession, db_session: AsyncSession, crop: str, state: str):
    """Fetch API data and save to DB. Log errors on failure."""
    try:
        data = await fetch_api_data(http_session, crop, state)
        
        # Success: Insert raw JSON into RawPrice
        raw_price = RawPrice(
            crop=crop,
            state=state,
            fetch_date=datetime.today().date(),
            raw_data=data
        )
        db_session.add(raw_price)
    
    except Exception as e:
        # Failure: Insert error message into ScrapeError
        print(f"Failed to fetch {crop} for {state}: {e}")
        scrape_error = ScrapeError(
            crop=crop,
            state=state,
            error_message=str(e),
            failed_at=datetime.utcnow()
        )
        db_session.add(scrape_error)

async def run_scraper():
    """Main orchestrator for scraping all crops across all states."""
    print(f"Starting e-NAM data collection at {datetime.now()}")
    
    async with aiohttp.ClientSession() as http_session:
        async with AsyncSessionLocal() as db_session:
            tasks = []
            
            # Use Semaphore to limit concurrent connections (e.g., 10 concurrent requests)
            sem = asyncio.Semaphore(10)
            
            async def sem_task(crop, state):
                async with sem:
                    await process_crop_state(http_session, db_session, crop, state)

            for crop in CROPS:
                for state in STATES:
                    tasks.append(sem_task(crop, state))
            
            await asyncio.gather(*tasks)
            
            # Commit all changes (inserts to RawPrice and ScrapeError)
            await db_session.commit()
            
    print(f"Finished e-NAM data collection at {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(run_scraper())
