"""
Memory-efficient PyTorch IterableDataset for Mandi price forecasting.

Reads forecast_dataset.parquet via DuckDB, streams data one commodity-market
pair at a time, applies pre-fitted scaling from preprocessor.json, and yields
fixed-length chronological sequences.

Features: [sequence_length, num_features]  (float32)
Targets:  [3]  (target_30d, target_60d, target_90d)  (float32)
Cat IDs:  [2]  (commodity_idx, market_idx)  (int64)

No sequence ever crosses a pair boundary.
No future information enters any sequence.
"""

import json
import math
from typing import Optional

import duckdb
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────
# Feature lists (must match preprocessor.json)
# ─────────────────────────────────────────────────────────────────────────
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
    "month", "week", "day_of_year", "season",
    "day_of_year_sin", "day_of_year_cos",
    "month_sin", "month_cos",
]

BINARY_FEATURES = ["is_interpolated"]
TARGET_COLUMNS  = ["target_30d", "target_60d", "target_90d"]


class MandiDataset(IterableDataset):
    """
    Streams chronological sequences per (Commodity, Market) pair.

    Parameters
    ----------
    parquet_path : str
        Path to forecast_dataset.parquet.
    preprocessor_path : str
        Path to preprocessor.json (fitted on TRAIN).
    split_config_path : str
        Path to split_config.json.
    split : str
        One of "train", "val", "test".
    sequence_length : int
        Number of timesteps per input sequence (default 30).
    horizon : str
        Which target to return: "30d", "60d", "90d", or "all" (default "all").
    """

    def __init__(
        self,
        parquet_path: str,
        preprocessor_path: str,
        split_config_path: str,
        split: str = "train",
        sequence_length: int = 30,
        horizon: str = "all",
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.seq_len = sequence_length
        self.horizon = horizon

        # Load configs
        with open(split_config_path) as f:
            config = json.load(f)
        with open(preprocessor_path) as f:
            self.prep = json.load(f)

        # Determine date boundaries
        splits = config["splits"]
        if split == "train":
            self.date_start = splits["train_start"]
            self.date_end = splits["train_end"]
        elif split == "val":
            self.date_start = splits["val_start"]
            self.date_end = splits["val_end"]
        elif split == "test":
            self.date_start = splits["test_start"]
            self.date_end = splits["test_end"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Valid pairs
        self.valid_pairs = [tuple(p) for p in config["valid_pairs"]]

        # Feature column order
        self.all_features = NUMERIC_FEATURES + TIME_FEATURES + BINARY_FEATURES
        self.n_features = len(self.all_features)

        # Target column(s)
        if horizon == "all":
            self.target_cols = TARGET_COLUMNS
        else:
            self.target_cols = [f"target_{horizon}"]

        # Build SQL column list
        self.select_cols = (
            ["Commodity", "Market", "Arrival_Date"]
            + self.all_features
            + self.target_cols
        )

    # ─── Scaling helpers ─────────────────────────────────────────────
    def _scale_numeric(self, arr: np.ndarray, col_idx: int, col_name: str) -> None:
        """In-place RobustScaler: (x - median) / IQR."""
        stats = self.prep["numeric_stats"][col_name]
        median = stats["median"] or 0.0
        iqr = stats["iqr"]
        arr[:, col_idx] = (arr[:, col_idx] - median) / iqr

    def _scale_time(self, arr: np.ndarray, col_idx: int, col_name: str) -> None:
        """In-place min-max scaling to [0, 1]."""
        stats = self.prep["time_stats"][col_name]
        mn, mx = stats["min"], stats["max"]
        rng = mx - mn if mx != mn else 1.0
        arr[:, col_idx] = (arr[:, col_idx] - mn) / rng

    def _process_pair(self, commodity: str, market: str):
        """
        Query all rows for one (Commodity, Market) pair within the split
        date range, scale features, and yield sliding-window sequences.
        """
        col_list = ", ".join(f'"{c}"' for c in self.select_cols)
        query = f"""
            SELECT {col_list}
            FROM '{self.parquet_path}'
            WHERE Commodity = $1
              AND Market    = $2
              AND Arrival_Date >= TIMESTAMP '{self.date_start}'
              AND Arrival_Date <  TIMESTAMP '{self.date_end}'
            ORDER BY Arrival_Date
        """

        con = duckdb.connect(":memory:")
        rows = con.execute(query, [commodity, market]).fetchnumpy()
        con.close()

        n_rows = len(rows["Commodity"])
        if n_rows < self.seq_len + 1:
            return  # not enough data for even one sequence

        # Build feature matrix [n_rows, n_features]
        feat_matrix = np.empty((n_rows, self.n_features), dtype=np.float32)
        for i, col_name in enumerate(self.all_features):
            raw = rows[col_name]
            feat_matrix[:, i] = np.where(
                np.isnan(raw.astype(np.float64)) if hasattr(raw, 'astype') else False,
                0.0,
                raw.astype(np.float32),
            )

        # Build target matrix [n_rows, n_targets]
        n_targets = len(self.target_cols)
        target_matrix = np.full((n_rows, n_targets), np.nan, dtype=np.float32)
        for j, col_name in enumerate(self.target_cols):
            raw = rows[col_name]
            if raw is not None:
                vals = raw.astype(np.float64)
                target_matrix[:, j] = np.where(np.isnan(vals), np.nan, vals).astype(np.float32)

        # Apply scaling
        for i, col_name in enumerate(NUMERIC_FEATURES):
            self._scale_numeric(feat_matrix, i, col_name)
        base = len(NUMERIC_FEATURES)
        for i, col_name in enumerate(TIME_FEATURES):
            self._scale_time(feat_matrix, base + i, col_name)
        # Binary features (is_interpolated) stay as-is (0/1)

        # Categorical IDs
        commodity_idx = self.prep["commodity_to_idx"].get(commodity, 0)
        market_idx = self.prep["market_to_idx"].get(market, 0)
        cat_ids = np.array([commodity_idx, market_idx], dtype=np.int64)

        # Sliding window: the target comes from the LAST timestep in the window
        for start in range(n_rows - self.seq_len):
            end = start + self.seq_len
            target_row = end  # row whose target we predict

            target_vals = target_matrix[target_row]

            # Skip if ALL targets are NaN (no valid forecast target)
            if np.all(np.isnan(target_vals)):
                continue

            x = torch.from_numpy(feat_matrix[start:end].copy())
            y = torch.from_numpy(np.nan_to_num(target_vals, nan=0.0))
            y_mask = torch.from_numpy((~np.isnan(target_vals)).astype(np.float32))
            cat = torch.from_numpy(cat_ids.copy())

            yield x, y, y_mask, cat

    def __iter__(self):
        """
        Iterate through all valid pairs, yielding sequences.
        Supports multi-worker DataLoader via pair partitioning.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pairs = self.valid_pairs
        else:
            # Partition pairs across workers
            n = len(self.valid_pairs)
            per_worker = int(math.ceil(n / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            pairs = self.valid_pairs[start:end]

        for commodity, market in pairs:
            yield from self._process_pair(commodity, market)


def create_dataloader(
    parquet_path: str,
    preprocessor_path: str,
    split_config_path: str,
    split: str,
    sequence_length: int = 30,
    horizon: str = "all",
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the given split."""
    ds = MandiDataset(
        parquet_path=parquet_path,
        preprocessor_path=preprocessor_path,
        split_config_path=split_config_path,
        split=split,
        sequence_length=sequence_length,
        horizon=horizon,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
