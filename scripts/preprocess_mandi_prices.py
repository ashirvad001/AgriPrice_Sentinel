import os
import pandas as pd
import numpy as np

# Create output dir
os.makedirs("data/processed", exist_ok=True)

input_file = "data/raw/mandi_prices.csv"
output_file = "data/processed/mandi_features.parquet"

print("--- Mandi Prices Preprocessing ---")

# Load data
df = pd.read_csv(input_file)
rows_before = len(df)
print(f"Rows before cleaning: {rows_before}")

# 1. Drop complete duplicates
df = df.drop_duplicates()

# 2. Convert date to datetime, drop invalid dates
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['date'])

# 3. Numeric price columns: coerce to float, drop missing/invalid
price_cols = ['min_price', 'max_price', 'modal_price']
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows missing any of the essential prices
df = df.dropna(subset=price_cols)

# Ensure min <= modal <= max logically, drop invalid rows
df = df[(df['min_price'] > 0) & (df['max_price'] > 0) & (df['modal_price'] > 0)]
# Depending on data quality, sometimes max_price < min_price, but let's just do a basic sanity filter
df = df[df['min_price'] <= df['max_price']]

# 4. Standardize text columns (strip whitespace, title case)
text_cols = ['state', 'district', 'mandi', 'commodity', 'variety', 'grade']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
    # Replace 'Nan' or empty string with pandas NA for optional fields (like variety, grade)
    df.loc[df[col].isin(['Nan', 'None', '']), col] = np.nan

# 5. Drop rows missing essential categorical fields
df = df.dropna(subset=['state', 'district', 'mandi', 'commodity'])

# 6. Handle arrival_quantity (numeric, keep missing as null)
if 'arrival_quantity' in df.columns:
    df['arrival_quantity'] = pd.to_numeric(df['arrival_quantity'], errors='coerce')

# 7. Feature Engineering
# Sort values to prevent future data leakage
df = df.sort_values(by=['commodity', 'mandi', 'date']).reset_index(drop=True)

# Time features
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week.astype(int)
df['day_of_year'] = df['date'].dt.dayofyear
df['season'] = (df['month'] % 12 // 3 + 1) # 1:Winter, 2:Spring, 3:Summer, 4:Fall

# Cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

grouped = df.groupby(['commodity', 'mandi'])

# Lags
lags = [1, 3, 7, 14, 30, 60, 90]
for lag in lags:
    df[f'modal_price_lag_{lag}'] = grouped['modal_price'].shift(lag)

# Rolling mean/std
windows = [7, 30]
for w in windows:
    # Use transform with a lambda to apply rolling within groups
    df[f'modal_price_roll_mean_{w}'] = grouped['modal_price'].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
    df[f'modal_price_roll_std_{w}'] = grouped['modal_price'].transform(lambda x: x.rolling(window=w, min_periods=1).std())

# Price change (absolute diff)
for w in windows:
    df[f'modal_price_change_{w}'] = df['modal_price'] - df[f'modal_price_lag_{w}']

# Save to parquet
df.to_parquet(output_file, index=False)

rows_after = len(df)

# Print Summary
print(f"\n--- Summary Report ---")
print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning: {rows_after}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Number of commodities: {df['commodity'].nunique()}")
print(f"Number of states: {df['state'].nunique()}")
print(f"Number of mandis: {df['mandi'].nunique()}")
print(f"Output saved to: {output_file}")
