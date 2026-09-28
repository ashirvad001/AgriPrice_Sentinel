import duckdb

con = duckdb.connect()

# Check mandi_features schema
print("=== mandi_features.parquet ===")
r = con.query("SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM 'data/processed/mandi_features.parquet')")
print(r.df().to_string())

print("\n=== Row count ===")
r2 = con.query("SELECT COUNT(*) as cnt FROM 'data/processed/mandi_features.parquet'")
print(r2.df())

print("\n=== forecast_dataset columns ===")
r3 = con.query("SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM 'data/processed/forecast_dataset.parquet')")
print(r3.df().to_string())

print("\n=== forecast_dataset row count ===")
r4 = con.query("SELECT COUNT(*) as cnt FROM 'data/processed/forecast_dataset.parquet'")
print(r4.df())

print("\n=== forecast_dataset sample dates ===")
r5 = con.query("SELECT MIN(Arrival_Date) as min_d, MAX(Arrival_Date) as max_d FROM 'data/processed/forecast_dataset.parquet'")
print(r5.df())

print("\n=== Unique entities in forecast_dataset ===")
r6 = con.query("SELECT COUNT(DISTINCT (Commodity || '|' || Market)) as n FROM 'data/processed/forecast_dataset.parquet'")
print(r6.df())
