from prometheus_client import Counter, Histogram, Gauge

# (1) Prediction count by crop
PREDICTION_COUNT = Counter(
    "api_prediction_count_total", 
    "Total accurate predictions made", 
    ["crop"]
)

# (2) Latency p50/p95/p99 (Histograms are used in PromQL with histogram_quantile)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "API Request Latency in seconds",
    # Specific buckets help calculate accurate p50, p95, p99
    buckets=[0.05, 0.1, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0, 2.5, 5.0]
)

# (3) Cache hit rate (Hit and Miss counters)
CACHE_HITS = Counter("api_cache_hits_total", "Total Redis cache hits")
CACHE_MISSES = Counter("api_cache_misses_total", "Total Redis cache misses")

# (4) API error rate
ERROR_COUNT = Counter("api_error_count_total", "Total API Errors", ["endpoint", "status_code"])

# Support for Alertmanager rule: model RMSE
MODEL_RMSE = Gauge("model_rmse", "Current RMSE of the forecasting model")
