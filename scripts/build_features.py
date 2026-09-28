"""
Build ML feature dataset from mandi_clean_provenance.parquet.

Outputs: data/processed/mandi_features.parquet

Uses Polars vectorised expressions partitioned by (Commodity, Market).
No row-level loops. No future leakage. No extra interpolation.
"""

import time
import polars as pl
import numpy as np

INPUT  = "data/processed/mandi_clean_provenance.parquet"
OUTPUT = "data/processed/mandi_features.parquet"

PARTITION = ["Commodity", "Market"]
PRICE_COL = "Modal_Price"
DATE_COL  = "Arrival_Date"

LAGS      = [1, 3, 7, 14, 30, 60, 90]
WINDOWS   = [7, 30]

# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    # ── 1. Read ──────────────────────────────────────────────────────────
    df = pl.read_parquet(INPUT)
    input_rows = df.height
    print(f"[1/5] Loaded {input_rows:,} rows  ({df.width} cols)")

    # ── 2. Sort (required for correct lag / rolling semantics) ───────────
    df = df.sort(PARTITION + [DATE_COL])

    # ── 3. Real-price column (nullify interpolated rows) ─────────────────
    #    Rolling/lag features must NOT treat interpolated values as real
    #    observations.  We create a masked copy and compute features from it.
    df = df.with_columns(
        pl.when(pl.col("is_interpolated") == 0)
          .then(pl.col(PRICE_COL))
          .otherwise(None)
          .alias("_real_price")
    )

    # ── 4. Price lag features ────────────────────────────────────────────
    print("[2/5] Computing price lags …")
    lag_exprs = [
        pl.col("_real_price")
          .shift(lag)
          .over(PARTITION)
          .alias(f"modal_price_lag_{lag}")
        for lag in LAGS
    ]
    df = df.with_columns(lag_exprs)

    # ── 5. Rolling features (mean, std, min, max) ────────────────────────
    print("[3/5] Computing rolling features …")
    rolling_exprs = []
    for w in WINDOWS:
        rolling_exprs.extend([
            pl.col("_real_price")
              .rolling_mean(window_size=w, min_samples=1)
              .over(PARTITION)
              .alias(f"modal_price_roll_mean_{w}"),

            pl.col("_real_price")
              .rolling_std(window_size=w, min_samples=2)
              .over(PARTITION)
              .alias(f"modal_price_roll_std_{w}"),

            pl.col("_real_price")
              .rolling_min(window_size=w, min_samples=1)
              .over(PARTITION)
              .alias(f"modal_price_roll_min_{w}"),

            pl.col("_real_price")
              .rolling_max(window_size=w, min_samples=1)
              .over(PARTITION)
              .alias(f"modal_price_roll_max_{w}"),
        ])
    df = df.with_columns(rolling_exprs)

    # ── 6. Price change features ─────────────────────────────────────────
    print("[4/5] Computing price changes & time features …")
    change_exprs = [
        (pl.col(PRICE_COL) -
         pl.col("_real_price").shift(w).over(PARTITION))
        .alias(f"modal_price_change_{w}")
        for w in WINDOWS
    ]
    df = df.with_columns(change_exprs)

    # Drop the helper column
    df = df.drop("_real_price")

    # ── 7. Time features ─────────────────────────────────────────────────
    date = pl.col(DATE_COL)
    df = df.with_columns([
        date.dt.month().alias("month"),
        date.dt.week().alias("week"),
        date.dt.ordinal_day().alias("day_of_year"),

        # Season: 1=Winter(Dec-Feb), 2=Spring(Mar-May),
        #          3=Summer(Jun-Aug), 4=Fall(Sep-Nov)
        ((date.dt.month() % 12) // 3 + 1).cast(pl.Int8).alias("season"),

        # Sin / cos annual seasonality
        (2.0 * np.pi * date.dt.ordinal_day() / 365.25).sin()
            .alias("day_of_year_sin"),
        (2.0 * np.pi * date.dt.ordinal_day() / 365.25).cos()
            .alias("day_of_year_cos"),
        (2.0 * np.pi * date.dt.month() / 12.0).sin()
            .alias("month_sin"),
        (2.0 * np.pi * date.dt.month() / 12.0).cos()
            .alias("month_cos"),
    ])

    # ── 8. Write ─────────────────────────────────────────────────────────
    print("[5/5] Writing parquet …")
    df.write_parquet(OUTPUT, use_pyarrow=True)

    elapsed = time.perf_counter() - t0

    # ── 9. Report ────────────────────────────────────────────────────────
    orig_cols = {
        "State", "District", "Market", "Commodity", "Variety", "Grade",
        "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price",
        "Commodity_Code", "is_interpolated",
    }
    feature_cols = sorted(set(df.columns) - orig_cols)
    null_counts = {
        col: df[col].null_count()
        for col in df.columns
        if df[col].null_count() > 0
    }

    print("\n" + "=" * 60)
    print("FEATURE BUILD REPORT")
    print("=" * 60)
    print(f"  Input rows        : {input_rows:>14,}")
    print(f"  Output rows       : {df.height:>14,}")
    print(f"  Feature cols added: {len(feature_cols):>14}")
    for c in feature_cols:
        print(f"      • {c}")
    print(f"  Commodities       : {df['Commodity'].n_unique():>14,}")
    print(f"  Markets           : {df['Market'].n_unique():>14,}")
    print(f"  Date range        : {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    print(f"  Missing values    :")
    if null_counts:
        for col, cnt in sorted(null_counts.items()):
            print(f"      {col:36s}: {cnt:>14,}")
    else:
        print("      (none)")
    print(f"  Processing time   : {elapsed:>11.1f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
