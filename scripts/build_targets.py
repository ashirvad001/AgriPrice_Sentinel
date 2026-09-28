"""
Create 30/60/90-day forecasting targets from mandi_features.parquet.

Outputs: data/processed/forecast_dataset.parquet

For each (Commodity, Market) series, target_Hd = the mean Modal_Price
from real (non-interpolated) observations exactly H calendar days ahead.

Uses DuckDB for out-of-core SQL execution to prevent Out of Memory errors.
"""

import time
import duckdb

INPUT  = "data/processed/mandi_features.parquet"
OUTPUT = "data/processed/forecast_dataset.parquet"
HORIZONS = [30, 60, 90]

def main() -> None:
    t0 = time.perf_counter()

    # ── 1. Connect to DuckDB ─────────────────────────────────────────────
    print("[1/3] Connecting to DuckDB...")
    con = duckdb.connect(database=':memory:')

    # Get input row count
    input_rows = con.execute(f"SELECT count(*) FROM '{INPUT}'").fetchone()[0]
    print(f"       Loaded {input_rows:,} rows")

    # ── 2. Execute target generation query ───────────────────────────────
    print(f"[2/3] Computing targets via SQL and writing to {OUTPUT}...")
    
    # We add an interval of H days to the base Arrival_Date to find the future date
    # i.e., target date = base date + H days.
    # Therefore, we join where base Arrival_Date + H = target Arrival_Date
    query = f"""
    COPY (
        WITH real_obs AS (
            SELECT Commodity, Market, Arrival_Date, AVG(Modal_Price) AS target_price
            FROM '{INPUT}'
            WHERE is_interpolated = 0
            GROUP BY Commodity, Market, Arrival_Date
        )
        SELECT 
            m.*,
            t30.target_price AS target_30d,
            t60.target_price AS target_60d,
            t90.target_price AS target_90d
        FROM '{INPUT}' m
        LEFT JOIN real_obs t30 
            ON m.Commodity = t30.Commodity 
            AND m.Market = t30.Market 
            AND (m.Arrival_Date + INTERVAL 30 DAY) = t30.Arrival_Date
        LEFT JOIN real_obs t60 
            ON m.Commodity = t60.Commodity 
            AND m.Market = t60.Market 
            AND (m.Arrival_Date + INTERVAL 60 DAY) = t60.Arrival_Date
        LEFT JOIN real_obs t90 
            ON m.Commodity = t90.Commodity 
            AND m.Market = t90.Market 
            AND (m.Arrival_Date + INTERVAL 90 DAY) = t90.Arrival_Date
    ) TO '{OUTPUT}' (FORMAT PARQUET)
    """
    
    con.execute(query)

    # ── 3. Generate Report Statistics ────────────────────────────────────
    print("[3/3] Generating report statistics...")
    
    output_rows = con.execute(f"SELECT count(*) FROM '{OUTPUT}'").fetchone()[0]
    
    # Date range
    min_date, max_date = con.execute(f"SELECT min(Arrival_Date), max(Arrival_Date) FROM '{OUTPUT}'").fetchone()
    
    # Valid counts
    target_stats = {}
    missing = {}
    for h in HORIZONS:
        tc = f"target_{h}d"
        valid = con.execute(f"SELECT count({tc}) FROM '{OUTPUT}'").fetchone()[0]
        miss = output_rows - valid
        pct = (valid / output_rows) * 100
        target_stats[tc] = (valid, pct)
        missing[tc] = miss

    # Samples by commodity (for those with at least one target)
    by_commodity_query = f"""
        SELECT Commodity, count(*) as samples
        FROM '{OUTPUT}'
        WHERE target_30d IS NOT NULL OR target_60d IS NOT NULL OR target_90d IS NOT NULL
        GROUP BY Commodity
        ORDER BY samples DESC
    """
    by_commodity = con.execute(by_commodity_query).fetchall()
    
    # Samples by pair
    by_pair_query = f"""
        SELECT count(*) as samples
        FROM '{OUTPUT}'
        WHERE target_30d IS NOT NULL OR target_60d IS NOT NULL OR target_90d IS NOT NULL
        GROUP BY Commodity, Market
    """
    pair_stats = con.execute(f"""
        WITH pair_counts AS ({by_pair_query})
        SELECT count(*), median(samples), avg(samples), min(samples), max(samples)
        FROM pair_counts
    """).fetchone()

    elapsed = time.perf_counter() - t0

    # ── Print ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORECAST TARGET REPORT")
    print("=" * 60)
    print(f"  Input rows          : {input_rows:>14,}")
    print(f"  Output rows         : {output_rows:>14,}")
    print()
    for h in HORIZONS:
        tc = f"target_{h}d"
        v, p = target_stats[tc]
        print(f"  Valid {tc:12s}  : {v:>14,}  ({p:.2f}%)")
    print()
    print(f"  Date range          : {min_date} to {max_date}")
    print()
    print(f"  Missing target counts:")
    for tc, cnt in missing.items():
        print(f"      {tc:20s}: {cnt:>14,}")
    print()

    print(f"  Samples by commodity ({len(by_commodity)} commodities with targets):")
    for row in by_commodity[:25]:
        print(f"      {row[0]:30s}: {row[1]:>12,}")
    if len(by_commodity) > 25:
        print(f"      ... and {len(by_commodity) - 25} more commodities")
    print()

    print(f"  Samples by commodity-market pair:")
    print(f"      Total pairs with targets: {pair_stats[0]:>10,}")
    print(f"      Median samples/pair     : {pair_stats[1]:>10,.0f}")
    print(f"      Mean samples/pair       : {pair_stats[2]:>10,.0f}")
    print(f"      Min samples/pair        : {pair_stats[3]:>10,}")
    print(f"      Max samples/pair        : {pair_stats[4]:>10,}")
    print()
    print(f"  Processing time     : {elapsed:>11.1f} s")
    print("=" * 60)

if __name__ == "__main__":
    main()
