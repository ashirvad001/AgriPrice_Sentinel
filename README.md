<div align="center">
  <h1>🌾 AgriPrice Sentinel</h1>
  <p><strong>AI-Powered Commodity Price Forecasting & Alert System for Indian Mandis</strong></p>
  
  ![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
  ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
  ![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
  ![Celery](https://img.shields.io/badge/Celery-5.3-37814A?logo=celery&logoColor=white)
</div>

---

## 📖 Overview

**AgriPrice Sentinel** is a comprehensive predictive analytics platform designed to forecast agricultural commodity prices across Indian wholesale markets (Mandis). By combining historical price tracking, weather data, and advanced deep learning (BiLSTM + Bahdanau Attention), it empowers farmers and traders to make data-driven selling decisions up to 90 days in advance.

The platform provides a Next.js dashboard for visualizing trends, SHAP-based model explainability, and proactive WhatsApp alerts powered by Twilio and Celery.

---

## ✨ Key Features

- 🔮 **Advanced Forecasting Engine**: Multi-horizon predictions (30, 60, 90 days) using a state-of-the-art **BiLSTM network with Bahdanau Attention** and Monte Carlo Dropout for 95% confidence intervals.
- 📱 **WhatsApp Bot & Proactive Alerts**: automated daily price alerts and threshold notifications sent directly to users' WhatsApp via Twilio.
- 🧠 **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations) values translate complex machine learning features into farmer-friendly insights (e.g., "Price dropped due to heavy rainfall 2 weeks ago").
- 🔄 **Automated MLOps Pipeline**: Celery beat workers automatically retrain models weekly, replacing the production model only if the new RMSE shows improvement.
- ⚡ **High-Performance API**: Asynchronous FastAPI backend backed by PostgreSQL (asyncpg) and Redis caching for millisecond-latency responses.

---

## 🏛️ System Architecture

### Tech Stack
*   **Backend:** Python, FastAPI, SQLAlchemy 2.0 (Async), Redis, Celery
*   **Machine Learning:** TensorFlow/Keras, Scikit-learn, SHAP, KerasTuner, Statsmodels (for baseline comparisons)
*   **Frontend:** React, Next.js 14, TailwindCSS, Recharts
*   **Database:** PostgreSQL (Core Data), Redis (Caching & Task Broker)
*   **External APIs:** Twilio (WhatsApp), OpenWeatherMap / IMD (Weather Data)

### Data Pipeline (48-Feature Matrix)
The feature engineering pipeline transforms raw price and weather data into a 48-feature dataset:
1.  **Price Lags & Rolling Stats**: Capturing short-term momentum (7d, 14d, 30d, 60d, 90d).
2.  **Seasonality**: Fourier encodings (sine/cosine) for month and week to capture harvest cycles.
3.  **Market Signals**: Distance from Minimum Support Price (MSP), price spread, and futures basis.
4.  **Agro-Climatology**: 10-day rolling rainfall accumulators and temperature ranges with dynamic drought flags.

---

## 📊 Model Evaluation Suite

The project includes a comprehensive evaluation script (`model_evaluation.py`) that benchmarks 5 algorithms across **16 crops × 3 mandis × 3 horizons (30/60/90 days)** for a total of 144 experiment runs.

**Compared Models:**
1. ARIMA(5,1,2)
2. SARIMA (Seasonal)
3. Facebook Prophet
4. Vanilla LSTM
5. **BiLSTM + Attention (Production Model)**

*The BiLSTM+Attention model consistently outperforms baselines by effectively capturing non-linear relationships between weather anomalies and price spikes, while MC Dropout provides robust uncertainty bands (95% CI).*

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.11+
*   Node.js 18+
*   PostgreSQL 15+
*   Redis server running locally or via Docker

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-username/AgriPrice_Sentinel.git
cd AgriPrice_Sentinel

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
cp .env.example .env
```

**Required `.env` Variables:**
```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/mandi_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your_jwt_secret_here
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
```

### 2. Database Initialization & Running the API
```bash
# The lifespan hook auto-creates SQLAlchemy tables on first run.
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*API Docs available at: `http://localhost:8000/docs`*

### 3. Background Workers (Celery)
To enable daily WhatsApp alerts and weekly model retraining:
```bash
# Terminal 1: Start the Celery worker
celery -A celery_app worker --loglevel=info

# Terminal 2: Start the Celery beat scheduler
celery -A celery_app beat --loglevel=info
```

### 4. Frontend Dashboard Setup
```bash
cd dashboard
npm install
npm run dev
```
*Dashboard available at: `http://localhost:3000`*

---

## 🧪 Model Training & Evaluation
To run the automated model evaluation suite and generate the LaTeX tables, heatmaps, and improvement charts:
```bash
python model_evaluation.py --crops Wheat Rice --horizons 30  # Quick run
# Or full 720-experiment run:
python model_evaluation.py
```
*Results are saved in the `evaluation_results/` directory and persisted to the `model_evaluations` PostgreSQL table.*

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
