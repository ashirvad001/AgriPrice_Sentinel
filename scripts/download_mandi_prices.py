import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAGOV_API_KEY")
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

if not API_KEY:
    print("Error: DATAGOV_API_KEY not found in environment.")
    exit(1)

all_records = []
offset = 0
limit = 1000  # max limit for data.gov.in

print(f"Fetching data from Data.gov.in with pagination...")

while True:
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
        "offset": offset
    }
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}", flush=True)
        try:
            print(f"Response: {response.text}")
        except:
            pass
        exit(1)
        
    data = response.json()
    records = data.get("records", [])
    
    if not records:
        break
        
    all_records.extend(records)
    print(f"Fetched {len(records)} records (total: {len(all_records)})...", flush=True)
    offset += limit
    
    # Optional: limit to 5000 records for testing so we don't fetch indefinitely
    # But user said "Implement pagination so all available records can be fetched."
    # data.gov.in mandi dataset can be around 5k - 10k records daily. 
    # Let's break if we get less than limit, meaning it's the last page.
    if len(records) < limit:
        break

if len(all_records) == 0:
    print("Error: No records found from API.")
    exit(1)

# Normalize columns
normalized_records = []
for r in all_records:
    # Handle missing keys safely
    rec = {
        "date": r.get("arrival_date"),
        "state": r.get("state"),
        "district": r.get("district"),
        "mandi": r.get("market"),
        "commodity": r.get("commodity"),
        "variety": r.get("variety"),
        "grade": r.get("grade"),
        "min_price": r.get("min_price"),
        "max_price": r.get("max_price"),
        "modal_price": r.get("modal_price"),
        "arrival_quantity": r.get("arrival_quantity") or r.get("arrivals") or None
    }
    normalized_records.append(rec)

df = pd.DataFrame(normalized_records)

# Validation
required_fields = ["min_price", "date", "mandi"]
if not all(field in df.columns for field in required_fields):
    print("Error: Required fields missing from API response.")
    exit(1)

if len(df) == 0:
    print("Error: Dataset has 0 rows after processing.")
    exit(1)

out_file = "data/raw/mandi_prices.csv"
os.makedirs("data/raw", exist_ok=True)

df.to_csv(out_file, index=False)

print("\n--- Summary Report ---")
print(f"Files created/modified: {out_file}")
print("Exact command to download the data: python scripts/download_mandi_prices.py")
print(f"Row count: {len(df)}")
print(f"Columns: {', '.join(df.columns)}")
if pd.notna(df['date']).any():
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
else:
    print("Date range: No valid dates found")
print(f"Number of states: {df['state'].nunique()}")
print(f"Number of commodities: {df['commodity'].nunique()}")
print(f"Number of mandis: {df['mandi'].nunique()}")
print("Dataset is valid for the next preprocessing step.")
