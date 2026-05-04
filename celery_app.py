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
from kombu import Queue
from dotenv import load_dotenv

load_dotenv()

# ── Celery instance ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "agriprice_sentinel",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.retrain", "tasks.alerts", "tasks.drift_detection"],
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

# ── Queue declarations ──────────────────────────────────────────────────────
app.conf.task_queues = [
    Queue("default"),
    Queue("retrain", routing_key="retrain"),
    Queue("alerts",  routing_key="alerts"),
]
app.conf.task_default_queue = "default"

# ── Beat schedule ────────────────────────────────────────────────────────────
# All times in Asia/Kolkata (IST = UTC+5:30)
# Sunday 2:00 AM IST = Saturday 20:30 UTC
# Monday 3:00 AM IST = Sunday 21:30 UTC
app.conf.beat_schedule = {
    "weekly-lstm-retrain": {
        "task": "tasks.retrain.retrain_all_models",
        "schedule": crontab(hour=2, minute=0, day_of_week="sunday"),
        "options": {"queue": "retrain"},
    },
    "daily-price-alerts": {
        "task": "tasks.alerts.send_daily_alerts",
        "schedule": crontab(hour=6, minute=0),
        "options": {"queue": "alerts"},
    },
    "weekly-drift-detection": {
        "task": "tasks.drift_detection.detect_drift_weekly",
        "schedule": crontab(hour=3, minute=0, day_of_week="monday"),
        "options": {"queue": "default"},
    },
}
