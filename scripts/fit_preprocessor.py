"""
Fit preprocessing artifacts (scaler statistics, categorical encodings) on the
TRAIN split only.  Saves to data/processed/preprocessor.json.

Uses DuckDB to compute statistics out-of-core — never loads the full 75M-row
dataset into RAM.
"""

import json
import time
import duckdb

INPUT  = "data/processed/forecast_dataset.parquet"
CONFIG = "data/processed/split_config.json"
OUTPUT = "data/processed/preprocessor.json"

# ── Feature lists ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Modal_Price",
    "modal_price_lag_1",
    "modal_price_lag_3",
    "modal_price_lag_7",
    "modal_price_lag_14",
    "modal_price_lag_30",
    "modal_price_lag_60",
    "modal_price_lag_90",
    "modal_price_roll_mean_7",
    "modal_price_roll_mean_30",
    "modal_price_roll_std_7",
    "modal_price_roll_std_30",
    "modal_price_roll_min_7",
    "modal_price_roll_min_30",
    "modal_price_roll_max_7",
    "modal_price_roll_max_30",
    "modal_price_change_7",
    "modal_price_change_30",
]

TIME_FEATURES = [
    "month",
    "week",
    "day_of_year",
    "season",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]

BINARY_FEATURES = ["is_interpolated"]

TARGET_COLUMNS = ["target_30d", "target_60d", "target_90d"]


def main() -> None:
    t0 = time.perf_counter()

    with open(CONFIG) as f:
        config = json.load(f)

    train_end = config["splits"]["train_end"]

    con = duckdb.connect(":memory:")

    # ── 1. Numeric feature statistics (RobustScaler: median + IQR) ───────
    print("[1/3] Computing numeric feature statistics on TRAIN split ...")
    stats: dict = {}

    # Build a single query that computes median, Q1, Q3 for every numeric col
    agg_parts = []
    for col in NUMERIC_FEATURES:
        agg_parts.append(
            f"approx_quantile(\"{col}\", 0.25) AS \"{col}__q1\", "
            f"approx_quantile(\"{col}\", 0.50) AS \"{col}__median\", "
            f"approx_quantile(\"{col}\", 0.75) AS \"{col}__q3\", "
            f"avg(\"{col}\") AS \"{col}__mean\", "
            f"stddev(\"{col}\") AS \"{col}__std\""
        )

    agg_sql = ", ".join(agg_parts)
    query = f"""
        SELECT {agg_sql}
        FROM '{INPUT}'
        WHERE Arrival_Date <= TIMESTAMP '{train_end}'
    """
    row = con.execute(query).fetchone()
    col_names = [desc[0] for desc in con.description]

    result = dict(zip(col_names, row))

    numeric_stats = {}
    for col in NUMERIC_FEATURES:
        q1 = result[f"{col}__q1"]
        median = result[f"{col}__median"]
        q3 = result[f"{col}__q3"]
        iqr = q3 - q1 if (q1 is not None and q3 is not None) else 1.0
        numeric_stats[col] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": max(iqr, 1e-8),  # avoid division by zero
            "mean": result[f"{col}__mean"],
            "std": max(result[f"{col}__std"] or 1.0, 1e-8),
        }

    # Time features: these are bounded/cyclical, use min-max from known ranges
    time_stats = {}
    for col in TIME_FEATURES:
        if col.endswith("_sin") or col.endswith("_cos"):
            time_stats[col] = {"min": -1.0, "max": 1.0}
        elif col == "month":
            time_stats[col] = {"min": 1, "max": 12}
        elif col == "week":
            time_stats[col] = {"min": 1, "max": 53}
        elif col == "day_of_year":
            time_stats[col] = {"min": 1, "max": 366}
        elif col == "season":
            time_stats[col] = {"min": 1, "max": 4}

    print(f"       Computed stats for {len(numeric_stats)} numeric + {len(time_stats)} time features")

    # ── 2. Categorical encodings ─────────────────────────────────────────
    print("[2/3] Computing categorical encodings on TRAIN split ...")

    commodity_vals = [r[0] for r in con.execute(f"""
        SELECT DISTINCT Commodity FROM '{INPUT}'
        WHERE Arrival_Date <= TIMESTAMP '{train_end}'
        ORDER BY Commodity
    """).fetchall()]

    market_vals = [r[0] for r in con.execute(f"""
        SELECT DISTINCT Market FROM '{INPUT}'
        WHERE Arrival_Date <= TIMESTAMP '{train_end}'
        ORDER BY Market
    """).fetchall()]

    # Index 0 reserved for unseen categories
    commodity_to_idx = {v: i + 1 for i, v in enumerate(commodity_vals)}
    market_to_idx = {v: i + 1 for i, v in enumerate(market_vals)}

    print(f"       Commodities: {len(commodity_to_idx)}, Markets: {len(market_to_idx)}")

    # ── 3. Save ──────────────────────────────────────────────────────────
    print("[3/3] Saving preprocessor state ...")

    preprocessor = {
        "numeric_features": NUMERIC_FEATURES,
        "time_features": TIME_FEATURES,
        "binary_features": BINARY_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "numeric_stats": numeric_stats,
        "time_stats": time_stats,
        "commodity_to_idx": commodity_to_idx,
        "market_to_idx": market_to_idx,
        "n_commodities": len(commodity_to_idx) + 1,  # +1 for unknown
        "n_markets": len(market_to_idx) + 1,
        "train_end": train_end,
    }

    with open(OUTPUT, "w") as f:
        json.dump(preprocessor, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\nSaved to {OUTPUT}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
