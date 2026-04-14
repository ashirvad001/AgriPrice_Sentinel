import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/postgres')
    try:
        await conn.execute('DROP DATABASE IF EXISTS test_alembic')
    except Exception as e:
        print(e)
    try:
        await conn.execute('CREATE DATABASE test_alembic')
        print("Database test_alembic created successfully.")
    except Exception as e:
        print(e)
    await conn.close()

asyncio.run(main())
