import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from scraper import run_scraper
from weather_scraper import run_weather_scraper
from database import init_db
import traceback

async def start_scheduler():
    # Initialize DB (creates tables if they don't exist)
    print("Initializing Database...")
    try:
        await init_db()
        print("Database Initialized Successfully.")
    except Exception as e:
        print(f"Database Initialization Failed: {e}")
        # Even if DB fails to init early, attempt to run the scheduler.
        # But logging error is essential
        traceback.print_exc()

    # Create the scheduler
    scheduler = AsyncIOScheduler(timezone=pytz.timezone('Asia/Kolkata'))
    
    # Run daily at 6:00 AM IST
    scheduler.add_job(
        run_scraper,
        trigger=CronTrigger(hour=6, minute=0),
        id="enam_scraper_job",
        replace_existing=True,
        misfire_grace_time=3600 # 1 hour grace time
    )
    
    # Run daily at 6:15 AM IST
    scheduler.add_job(
        run_weather_scraper,
        trigger=CronTrigger(hour=6, minute=15),
        id="imd_weather_scraper_job",
        replace_existing=True,
        misfire_grace_time=3600 # 1 hour grace time
    )
    
    scheduler.start()
    print("APScheduler started. E-NAM scheduled for 6:00 AM, IMD Weather for 6:15 AM IST daily.")
    
    # Keep the event loop running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(start_scheduler())
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
