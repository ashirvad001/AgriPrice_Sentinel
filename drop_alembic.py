import asyncio
import asyncpg

async def drop():
    conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/test_alembic')
    await conn.execute('DROP TABLE IF EXISTS alembic_version CASCADE; DROP SCHEMA public CASCADE; CREATE SCHEMA public;')
    await conn.close()

asyncio.run(drop())
