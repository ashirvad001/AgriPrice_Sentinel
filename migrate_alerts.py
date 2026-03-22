import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv

load_dotenv()

async def migrate():
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
    engine = create_async_engine(DATABASE_URL)
    
    async with engine.begin() as conn:
        print("Migrating alert_subscriptions...")
        try:
            await conn.execute(text("ALTER TABLE alert_subscriptions ALTER COLUMN user_id DROP NOT NULL;"))
            print("- user_id made nullable")
        except Exception as e:
            print("- user_id already nullable or error:", e)
            
        try:
            await conn.execute(text("ALTER TABLE alert_subscriptions ADD COLUMN phone_number VARCHAR(20);"))
            print("- Added phone_number column")
        except Exception as e:
            print("- phone_number error (maybe exists):", e)
            
        try:
            await conn.execute(text("ALTER TABLE alert_subscriptions ADD COLUMN language VARCHAR(20) DEFAULT 'English';"))
            print("- Added language column")
        except Exception as e:
            print("- language error (maybe exists):", e)

    await engine.dispose()
    print("Migration finished.")

if __name__ == "__main__":
    asyncio.run(migrate())
