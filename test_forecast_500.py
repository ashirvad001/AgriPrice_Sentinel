import asyncio
import pytest
from api.routes_forecast import get_forecast
from database import AsyncSessionLocal

@pytest.mark.asyncio
async def test():
    async with AsyncSessionLocal() as db:
        try:
            res = await get_forecast("wheat", "Amritsar", 30, db)
            print("SUCCESS")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
