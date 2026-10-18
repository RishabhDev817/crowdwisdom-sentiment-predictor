# CrowdWisdom Sentiment Predictor

Quantitative machine-learning pipeline designed for the CrowdWisdomTrading Quantitative Data Scientist assessment.

## Project Goal
The primary objective of this project is to build an end-to-end, reproducible forecasting pipeline that combines social media sentiment signals (from X / Twitter) with historical asset pricing data to forecast BTC-USD prices across multi-step horizons ($n+0$ through $n+5$). The evaluation relies on rigorous walk-forward validation and quantitative trading metrics rather than standard random train-test splits.

## Planned Pipeline Architecture
1. **Data Ingestion (`src/ingestion/`)**:
   - Market data ingestion: Historical daily OHLCV bars for BTC-USD (180-day lookback window) using `yfinance`.
   - Sentiment data ingestion: Structured extraction of sentiment signals from X (Twitter) using search/scraping APIs (e.g. Apify, Tavily, Exa).
2. **Feature Engineering & Processing (`src/processing/`)**:
   - Price-derived technical indicators, log returns, and volatility measures.
   - Sentiment signal aggregation, rolling polarity/volume metrics, and timestamp alignment to market close.
3. **Forecasting Models (`src/models/`)**:
   - Baseline statistical benchmarks (e.g., Naive persistence, Moving Average).
   - Machine learning multi-horizon regressors (e.g., LightGBM, XGBoost, Ridge/Linear baselines) forecasting $n+0$ to $n+5$.
4. **Validation & Quantitative Evaluation (`src/evaluation/`)**:
   - Expanding / rolling walk-forward backtesting (strictly avoiding future lookahead bias).
   - Statistical forecasting metrics (RMSE, MAE, MAPE, Directional Accuracy).
   - Quantitative trading metrics (Sharpe ratio, cumulative returns, max drawdown, win rate).
5. **Visualization & Reporting (`src/visualization/`, `reports/figures/`)**:
   - Multi-horizon forecast vs. actual plots.
   - Cumulative returns and drawdown curves across walk-forward folds.
   - Feature importance and sentiment attribution diagnostics.

## Current Setup Status
- **Status**: Data ingestion phase in progress.
- **Implemented Components**:
  - Directory hierarchy established (`config/`, `data/`, `src/`, `reports/`, `tests/`, `notebooks/`).
  - Base configuration defined in `config/config.yaml` (configured for BTC-USD, daily bars, 180-day lookback, and Apify sentiment settings).
  - Environment variable template created (`.env.example`).
  - Core package dependencies specified (`requirements.txt`).
  - Historical BTC-USD price ingestion pipeline (`src/ingestion/price_data.py`) with comprehensive validation and unit test suite.
  - X (Twitter) post ingestion pipeline (`src/ingestion/x_posts.py`) with schema normalization, deduplication, and offline test suite.
- **Pending Implementation**:
  - Social sentiment scoring and polarity analysis.
  - Feature engineering and timestamp alignment to daily market close.
  - Model architectures, training routines, and walk-forward validation.
  - Evaluation reporting and visualization scripts.

*(Note: No forecasting models have been trained and no sentiment scores have been generated yet.)*

## Project Structure
```text
crowdwisdom-sentiment-predictor/
├── config/
│   └── config.yaml            # Pipeline configuration parameters
├── data/
│   ├── raw/                   # Raw ingested market & sentiment data (gitignored)
│   └── processed/             # Cleaned, aligned feature datasets (gitignored)
├── notebooks/                 # Exploratory research notebooks
├── reports/
│   └── figures/               # Generated evaluation charts & plots
├── src/
│   ├── ingestion/             # Data retrieval modules (price_data.py, x_posts.py)
│   ├── processing/            # Data cleaning & feature engineering
│   ├── models/                # Multi-horizon forecasting models
│   ├── evaluation/            # Walk-forward validation & quant metrics
│   └── visualization/         # Diagnostic and reporting charts
├── tests/                     # Unit and integration tests
├── .env.example               # Environment variables template
├── .gitignore                 # Repository exclusions
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

## Getting Started
### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration & Credentials
Copy `.env.example` to `.env` and populate your API credentials when ready:
```bash
cp .env.example .env
```
Ensure parameters in `config/config.yaml` match your target settings.

### 3. Running Data Ingestion
```bash
# Ingest historical BTC price bars (yfinance)
python -m src.ingestion.price_data

# Ingest historical X posts (Apify - requires APIFY_API_TOKEN in .env)
python -m src.ingestion.x_posts
```
