import duckdb
import json
import time

INPUT = 'data/processed/forecast_dataset.parquet'
OUTPUT_CONFIG = 'data/processed/split_config.json'

def main():
    t0 = time.perf_counter()
    con = duckdb.connect(':memory:')
    
    print("[1/3] Calculating global chronological splits (70/15/15)...")
    
    # Calculate splits based on row quantiles to ensure volume is roughly 70/15/15
    # (Since data density increases over time, duration-based splits skew heavily towards the end)
    dates = con.execute(f"""
        SELECT 
            min(Arrival_Date) as min_date,
            approx_quantile(Arrival_Date, 0.70) as split_train_val,
            approx_quantile(Arrival_Date, 0.85) as split_val_test,
            max(Arrival_Date) as max_date
        FROM '{INPUT}'
    """).fetchone()
    
    min_date, train_val_date, val_test_date, max_date = dates
    
    print(f"  Train: {min_date} to {train_val_date}")
    print(f"  Valid: {train_val_date} to {val_test_date}")
    print(f"  Test : {val_test_date} to {max_date}")

    print("\n[2/3] Identifying valid commodity-market pairs (>=1000 targets)...")
    # A pair is valid if it has at least 1000 samples for the horizons we care about.
    # The prompt says "at least 1,000 valid target samples" - we can filter where at least one horizon has >=1000.
    valid_pairs_query = f"""
        SELECT Commodity, Market
        FROM '{INPUT}'
        GROUP BY Commodity, Market
        HAVING count(target_30d) >= 1000 
           AND count(target_60d) >= 1000 
           AND count(target_90d) >= 1000
    """
    valid_pairs_df = con.execute(valid_pairs_query).df()
    valid_pairs = [tuple(x) for x in valid_pairs_df.to_numpy()]
    
    num_pairs = len(valid_pairs)
    num_commodities = valid_pairs_df['Commodity'].nunique()
    num_markets = valid_pairs_df['Market'].nunique()
    
    print(f"  Selected pairs: {num_pairs:,}")
    print(f"  Unique commodities: {num_commodities:,}")
    print(f"  Unique markets: {num_markets:,}")

    print("\n[3/3] Calculating sample counts for the selected universe...")
    # Register the valid pairs so we can join/filter
    con.register('valid_pairs', valid_pairs_df)
    
    sample_counts = con.execute(f"""
        SELECT 
            count(target_30d) as s30,
            count(target_60d) as s60,
            count(target_90d) as s90
        FROM '{INPUT}' d
        INNER JOIN valid_pairs v ON d.Commodity = v.Commodity AND d.Market = v.Market
    """).fetchone()

    s30, s60, s90 = sample_counts
    print(f"  Total valid 30d samples: {s30:,}")
    print(f"  Total valid 60d samples: {s60:,}")
    print(f"  Total valid 90d samples: {s90:,}")

    # Write config
    config = {
        "splits": {
            "train_start": str(min_date),
            "train_end": str(train_val_date),
            "val_start": str(train_val_date),
            "val_end": str(val_test_date),
            "test_start": str(val_test_date),
            "test_end": str(max_date)
        },
        "valid_pairs": valid_pairs,
        "metadata": {
            "num_commodities": num_commodities,
            "num_markets": num_markets,
            "num_pairs": num_pairs,
            "samples_30d": s30,
            "samples_60d": s60,
            "samples_90d": s90
        }
    }
    
    with open(OUTPUT_CONFIG, 'w') as f:
        json.dump(config, f)
        
    print(f"\nSaved split configuration to {OUTPUT_CONFIG}")
    print(f"Time taken: {time.perf_counter() - t0:.1f}s")

if __name__ == '__main__':
    main()
