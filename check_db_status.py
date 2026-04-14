import asyncio
from sqlalchemy import text
from database import engine

async def check_db():
    try:
        async with engine.connect() as conn:
            # Check connection
            await conn.execute(text("SELECT 1"))
            print("✅  DB connection successful")
            
            # Check users table
            result = await conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"))
            exists = result.scalar()
            print(f"✅  Users table exists: {exists}")
            
    except Exception as e:
        print(f"❌  DB error: {e}")

if __name__ == "__main__":
    asyncio.run(check_db())
