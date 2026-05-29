"""
seed_data.py
────────────
Seed the raw_prices table with ~180 days of realistic historical
price data for key crop–mandi combinations.

This enables the forecast endpoint to produce meaningful predictions
using the statistical baseline (linear trend + seasonal decomposition)
even when the live Agmarknet scraper hasn't run yet.

Usage:
    python seed_data.py
"""

import asyncio
import random
import math
from datetime import datetime, timedelta, timezone, date

from database import AsyncSessionLocal, RawPrice, init_db
from sqlalchemy import select, and_


# ── Crop–Mandi configurations with realistic price ranges ────────────────────
SEED_CONFIGS = [
    # (crop, state, mandi_name, base_price, volatility, trend_per_day, season_amplitude)
    ("Wheat",      "Madhya Pradesh", "Indore Mandi",    2250,  35, +0.8,  50),
    ("Wheat",      "Uttar Pradesh",  "Lucknow Mandi",   2200,  30, +0.6,  45),
    ("Rice",       "Uttar Pradesh",  "Lucknow Mandi",   2350,  40, +0.5,  60),
    ("Rice",       "West Bengal",    "Kolkata Mandi",    2400,  45, +0.4,  55),
    ("Maize",      "Madhya Pradesh", "Indore Mandi",    2050,  30, +0.3,  40),
    ("Soybean",    "Madhya Pradesh", "Indore Mandi",    4500,  80, +1.2,  120),
    ("Gram",       "Madhya Pradesh", "Indore Mandi",    5300,  60, +0.9,  80),
    ("Mustard",    "Rajasthan",      "Jaipur Mandi",    5500,  70, +1.0,  100),
    ("Onion",      "Maharashtra",    "Lasalgaon Mandi", 1800,  100, -0.5, 200),
    ("Potato",     "Uttar Pradesh",  "Agra Mandi",      1200,  50, +0.2,  80),
    ("Tur",        "Maharashtra",    "Latur Mandi",     6800,  90, +1.5,  150),
    ("Moong",      "Rajasthan",      "Jaipur Mandi",    8200,  100, +1.0, 180),
    ("Cotton",     "Gujarat",        "Rajkot Mandi",    6900,  80, +0.7,  100),
    ("Bajra",      "Rajasthan",      "Jaipur Mandi",    2500,  40, +0.5,  60),
    ("Jowar",      "Maharashtra",    "Solapur Mandi",   3200,  50, +0.6,  70),
    ("Sugarcane",  "Uttar Pradesh",  "Lucknow Mandi",   310,   10, +0.1,  15),
]

DAYS_TO_SEED = 180  # ~6 months of history


def generate_price_series(
    base_price: float,
    volatility: float,
    trend_per_day: float,
    season_amplitude: float,
    days: int,
    seed: int = 42,
) -> list[dict]:
    """Generate realistic daily price data with trend, seasonality, and noise."""
    rng = random.Random(seed)
    prices = []
    today = date.today()
    price = base_price

    for i in range(days, 0, -1):
        d = today - timedelta(days=i)

        # Linear trend
        trend = trend_per_day * (days - i)

        # Weekly seasonality (mandis are busier mid-week)
        day_of_week = d.weekday()
        weekly_factor = math.sin(2 * math.pi * day_of_week / 7) * (season_amplitude * 0.3)

        # Monthly seasonality (harvest cycles)
        day_of_year = d.timetuple().tm_yday
        monthly_factor = math.sin(2 * math.pi * day_of_year / 365) * season_amplitude

        # Random walk component
        price += rng.gauss(0, volatility * 0.3)

        # Combine
        final_price = base_price + trend + weekly_factor + monthly_factor + (price - base_price) * 0.5
        final_price = max(final_price * 0.7, final_price)  # Floor at 70% of calculated

        # Add min/max spread
        spread = abs(rng.gauss(0, volatility * 0.5))
        min_price = final_price - spread
        max_price = final_price + spread

        # Arrivals (tonnes) — realistic variation
        base_arrivals = 500 + rng.random() * 2000
        arrivals = max(50, base_arrivals + rng.gauss(0, 300))

        prices.append({
            "date": d,
            "modal_price": round(final_price, 2),
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2),
            "arrivals_tonnes": round(arrivals, 1),
        })

    return prices


async def seed_database():
    """Seed the database with historical price data."""
    print("[SEED] Seeding historical price data...")

    # Ensure tables exist
    await init_db()
    print("[OK] Database tables verified")

    async with AsyncSessionLocal() as session:
        total_inserted = 0
        total_skipped = 0

        for idx, (crop, state, mandi, base, vol, trend, amp) in enumerate(SEED_CONFIGS):
            print(f"  [+] Seeding {crop} / {mandi}...", end=" ")

            prices = generate_price_series(
                base_price=base,
                volatility=vol,
                trend_per_day=trend,
                season_amplitude=amp,
                days=DAYS_TO_SEED,
                seed=42 + idx,  # Different seed per crop for variety
            )

            inserted = 0
            skipped = 0
            for p in prices:
                # Check if record already exists
                existing = await session.execute(
                    select(RawPrice).where(
                        and_(
                            RawPrice.crop == crop,
                            RawPrice.state == state,
                            RawPrice.fetch_date == p["date"],
                        )
                    )
                )
                if existing.scalar_one_or_none():
                    skipped += 1
                    continue

                # Build raw_data matching the schema expected by routes_forecast.py
                raw_data = {
                    "commodity": crop,
                    "market": mandi,
                    "market_name": mandi,
                    "mandi": mandi,
                    "state": state,
                    "modal_price": p["modal_price"],
                    "min_price": p["min_price"],
                    "max_price": p["max_price"],
                    "arrivals_tonnes": p["arrivals_tonnes"],
                    "rainfall_mm": round(random.uniform(0, 15), 1),
                    "max_temp": round(random.uniform(28, 42), 1),
                    "min_temp": round(random.uniform(12, 26), 1),
                    "freight_index": round(100 + random.uniform(-10, 15), 1),
                    "futures_price": round(p["modal_price"] * random.uniform(0.98, 1.04), 2),
                    "msp": base,
                }

                session.add(RawPrice(
                    crop=crop,
                    state=state,
                    fetch_date=p["date"],
                    raw_data=raw_data,
                    created_at=datetime.now(timezone.utc),
                ))
                inserted += 1

            await session.commit()
            total_inserted += inserted
            total_skipped += skipped
            print(f"OK {inserted} inserted, {skipped} skipped")

        print()
        print(f"[DONE] Seeding complete! {total_inserted} records inserted, {total_skipped} skipped.")
        print(f"    Crops seeded: {len(SEED_CONFIGS)}")
        print(f"    Date range: {date.today() - timedelta(days=DAYS_TO_SEED)} -> {date.today()}")


if __name__ == "__main__":
    asyncio.run(seed_database())
