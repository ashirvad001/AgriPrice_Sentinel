import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/test_alembic')
    await conn.execute('DROP SCHEMA public CASCADE; CREATE SCHEMA public;')
    print("Test alembic schema reset.")
    await conn.close()

asyncio.run(main())
