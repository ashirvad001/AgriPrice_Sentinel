"""
celery_app.py
─────────────
Celery application configuration with Redis broker and
celery-beat schedule for weekly LSTM model retraining.

Start the worker:
    celery -A celery_app worker --loglevel=info

Start the beat scheduler:
    celery -A celery_app beat --loglevel=info
"""

import os
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()

# ── Celery instance ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "agriprice_sentinel",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.retrain", "tasks.alerts"],
)

# ── Celery configuration ────────────────────────────────────────────────────
app.conf.update(
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=14400,       # 4-hour hard limit per task
    task_soft_time_limit=10800,  # 3-hour soft limit (raises SoftTimeLimitExceeded)
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=4,  # restart worker after 4 tasks to prevent memory leaks
)

# ── Beat schedule: retrain every Sunday at 2:00 AM IST ───────────────────────
app.conf.beat_schedule = {
    "weekly-lstm-retrain": {
        "task": "tasks.retrain.retrain_all_models",
        "options": {"queue": "retrain"},
    },
    "daily-price-alerts": {
        "task": "tasks.alerts.send_daily_alerts",
        "schedule": crontab(hour=6, minute=0),
        "options": {"queue": "alerts"},
    },
}
