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

## Current Implementation Status
- **Phase 1 (Completed)**: Offline, deterministic lexicon-based sentiment analysis (`src/processing/sentiment.py`) with negation handling, intensity modifiers, and multi-word phrase matching.
- **Phase 2 (Completed)**: Daily UTC sentiment aggregation and price bar alignment (`src/processing/features.py`), engagement-weighted sentiment calculation, and look-ahead isolation.
- **Phase 3 & 3.1 (Completed)**: Multi-horizon target generation ($n+1 \dots n+5$), backward-looking price-derived features, forward-return target modeling, price reconstruction, calendar continuity validation, and contemporaneous nowcast reference isolation (`src/models/baselines.py`).
- **Phase 4.2, 4.3 & 4.4 (Completed)**: Strategy-target alignment, dual-track evaluation framework, and targeted backtest accounting corrections. Dedicated executable trade return target (`target_trade_return = Open.shift(-2) / Open.shift(-1) - 1.0`), strategy target boundary purging (88 eligible training observations per 90-day fold), 10 bps turnover fee deduction with terminal liquidation accounting, drawdown initial capital baseline, gross-of-fees benchmark disclosure, and explicit active vs. all-period win rate reporting.
- **Phase 5.0 — 5.4 (Completed)**: End-to-end sentiment-to-forecasting pipeline integration and controlled live collection. Integrated price ingestion, genuine Apify post collection (`apidojo/tweet-scraper`), deterministic sentiment scoring, daily UTC aggregation, price calendar alignment, leakage-safe feature selection, and purged walk-forward backtesting. Ingested 10 genuine X posts via Apify and verified full pipeline execution, honest reporting of data coverage limitations, and zero data leakage.

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
│   ├── processing/            # Sentiment scoring, daily aggregation, target generation
│   │   ├── sentiment.py       # Deterministic rule-based sentiment engine
│   │   └── features.py        # Daily aggregation, forecast & strategy trade target shifting
│   ├── models/                # Forecasting models and baselines
│   │   └── baselines.py       # Naive persistence, Ridge pipeline, multi-horizon orchestrator
│   ├── evaluation/            # Walk-forward validation & quant metrics (Phases 4 & 4.3)
│   │   ├── walk_forward.py    # Rolling chronological walk-forward splits & dual-track evaluation
│   │   ├── metrics.py         # Statistical forecast metrics (RMSE, MAE, MAPE, DA)
│   │   ├── backtest.py        # Quantitative trading backtest engine & performance metrics
│   │   └── run_evaluation.py  # Standalone evaluation & backtest execution runner
│   └── visualization/         # Diagnostic and reporting charts
├── tests/                     # Unit and integration tests (177 offline tests)
│   ├── fixtures/              # Synthetic fixtures for offline validation
│   ├── test_price_data.py     # Price ingestion & validation tests
│   ├── test_x_posts.py        # X post ingestion & normalization tests
│   ├── test_sentiment.py      # Sentiment scoring & negation tests
│   ├── test_features.py       # Aggregation & alignment tests
│   ├── test_targets.py        # Forecasting & strategy trade target tests
│   ├── test_baselines.py      # Baseline models & price reconstruction tests
│   ├── test_walk_forward.py   # Walk-forward purging & dual-track separation tests
│   ├── test_backtest.py       # Strategy execution & benchmark alignment tests
│   └── test_pipeline.py       # Integrated pipeline orchestration & missing-data guards
├── .env.example               # Environment variables template
├── .gitignore                 # Repository exclusions
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

## Forecasting Framework & Target Definitions

### Horizon Definitions ($n+0$ through $n+5$)
Target variables are separated into a contemporaneous nowcast reference and forward forecast horizons:
- **$n+0$ (`target_Close_n0`, `target_return_h0`) — Contemporaneous Nowcast Reference**:
  Defined explicitly as the closing price of the current prediction cutoff date ($Close_t$). It represents zero time shift ($h=0$) and zero return ($0.0$). It is **not** a future forecast, but serves as a same-time reference benchmark. It is **strictly quarantined from Ridge model training and from aggregate forecast metric summaries**, preventing a trivial $0.0$ nowcast error from deflating future forecast error metrics.
- **$n+1$ through $n+5$ (`target_return_h1` ... `target_return_h5`, `target_Close_n1` ... `target_Close_n5`) — Forward Return & Future Closing Targets**:
  - **Training Target for Ridge**: Forward cumulative percentage return:
    $$\text{target\_return\_h}_h = \frac{Close_{t+h}}{Close_t} - 1.0$$
    Forward returns are stationary, zero-centered, and suitable for linear regularized regression, eliminating the non-stationarity and sample-mean shrinkage issues of raw price levels.
  - **Verification Target**: Future nominal close price $Close_{t+h}$ obtained via row shift (`shift(-h)`).
  - The last $h$ rows of the dataset evaluate to `NaN` as future prices beyond the dataset boundary are unobserved, and are masked during training.

### Daily Calendar Continuity Validation
Because row-based shifts (`shift(-h)`) assume consecutive daily observations:
- The pipeline rigorously validates that the UTC date series is **unique, strictly ascending, and spaced exactly 1 calendar day apart** via `validate_calendar_continuity()`.
- If missing calendar dates (gaps), duplicate dates, or unsorted rows are detected, a clear `ValueError` is raised.
- Data bars are never silently dropped, interpolated, or fabricated.

### Prediction Cutoff & Information Availability
To ensure complete absence of look-ahead bias and temporal leakage, the prediction convention is strictly enforced:
- **Prediction Cutoff**: Predictions for future dates ($D+1 \dots D+5$) are generated at or immediately following **$D\text{ 23:59:59 UTC}$** (the close of day $D$).
- **Feature Eligibility**: A feature vector for observation date $D$ contains **only** data generated at or before $D\text{ 23:59:59 UTC}$.
  - Sentiment features aggregated from posts published strictly within $[D\text{ 00:00:00 UTC}, D\text{ 23:59:59 UTC}]$.
  - Technical and return indicators derived exclusively from day $D$ and earlier bars.
- Raw nominal price levels ($Close, Open, High, Low$) and future target columns are strictly quarantined from the model feature matrix $X$.
- $Close_D$ is used strictly as the prediction-time price anchor to reconstruct nominal price forecasts from predicted returns.

### Feature Column Specification
Model input features ($X$) consist of strictly stationary price-derived signals and bounded sentiment signals:
1. **Price-Derived Features (Strictly Backward-Looking & Stationary)**:
   - `return_1d`: 1-day percentage change ($\frac{Close_t}{Close_{t-1}} - 1$).
   - `return_3d`: 3-day percentage change ($\frac{Close_t}{Close_{t-3}} - 1$).
   - `return_7d`: 7-day percentage change ($\frac{Close_t}{Close_{t-7}} - 1$).
   - `volatility_7d`: 7-day rolling standard deviation of 1-day returns.
   - `high_low_ratio`: Intra-day range ratio ($\frac{High_t}{Low_t}$).
   - `close_open_ratio`: Intra-day direction ratio ($\frac{Close_t}{Open_t}$).
   - `Volume`: Daily trading volume.
2. **Sentiment Features**:
   - `sentiment_mean`: Arithmetic mean of post polarity scores on day $D$.
   - `post_count`: Total number of valid posts on day $D$.
   - `sentiment_weighted`: Engagement-weighted polarity score (using verified likes, retweets, replies, and quotes).
   - `sentiment_weighted_available`: Boolean indicator specifying whether valid engagement metrics were present.
   - `sentiment_pos_ratio`, `sentiment_neg_ratio`, `sentiment_neu_ratio`: Proportions of positive, negative, and neutral posts.
   - `has_posts`: Boolean indicator distinguishing days with genuine post activity from days with missing sentiment data.
3. **Excluded Columns (Leakage & Non-Stationarity Quarantine)**:
   - All target columns (`target_*`, `forward_return*`).
   - Raw nominal price levels (`Close`, `Open`, `High`, `Low`).
   - Identifiers, URLs, raw text strings, and metadata (`id`, `url`, `text`, `Date`, `is_synthetic`).

### Missing Sentiment Handling
When no social posts are available for a given day:
- `has_posts` is set to `False`, and `sentiment_mean` is set to `NaN`.
- Missing sentiment is **not** silently converted to $0.0$ (neutral), preserving the true distinction between missing data and genuine neutrality.
- The machine learning pipeline applies median imputation (`SimpleImputer(strategy="median")`) learned **exclusively on the training fold**, preventing data snooping.

## Baseline Models

### 1. Naive Price-Persistence Baseline (`NaivePersistenceModel`)
- **Mechanism**: Predicts a forward return of zero ($\hat{r}_{t, h} = 0.0$), which converts to the latest available closing price at the prediction cutoff: $\hat{y}_{t+h} = Close_t \times (1.0 + 0.0) = Close_t$.
- **Purpose**: Establishes the primary statistical benchmark for random walk asset price series.
- **Horizon $n+0$**: Evaluated as a reference benchmark ($\hat{y}_t = Close_t$), yielding zero error, but strictly quarantined from future forecast metric aggregations.

### 2. Ridge Regression Baseline (`create_ridge_pipeline`)
- **Mechanism**: L2-regularized linear model trained on scaled stationary price-derived and sentiment features to predict forward return $\hat{r}_{t, h}$.
- **Pipeline Architecture**:
  ```text
  Stationary Features (X) ──> SimpleImputer(median) ──> StandardScaler() ──> Ridge(alpha=1.0) ──> Forecast Return (r_hat)
  ```
- **Price Reconstruction**:
  Forecast returns are converted back to nominal price levels via:
  $$\hat{y}_{t, h} = Close_t \times (1.0 + \hat{r}_{t, h})$$
  The price anchor $Close_t$ is strictly validated for positive, finite values.
- **Leakage Prevention**: All preprocessing stages (imputation medians, standard scaling means/variances) are fitted strictly on the training partition within a scikit-learn `Pipeline`.
- **Reproducibility**: Parameterized with fixed random state (`random_state=42`) and configurable regularization strength $\alpha$.

### 3. Multi-Horizon Forecaster (`MultiHorizonForecaster`)
- **Mechanism**: Orchestrates horizon-specific models across future horizons $h \in \{1, 2, 3, 4, 5\}$.
- **Return & Price Outputs**: Provides explicit methods `predict_returns()` and `predict_prices()` as well as `predict(return_type='price')`.
- **Nowcast Safeguard**: Rejects any attempt to fit a Ridge model on $h=0$ with a clear `ValueError`. Provides `predict_nowcast()` strictly for reference.
- **Evaluation Safeguard**: `aggregate_forecast_metrics()` automatically filters out $h=0$ so nowcast reference results cannot artificially deflate future forecast error metrics.

## Walk-Forward Cross-Validation & Strategy Backtesting (Phases 4, 4.1, 4.2 & 4.3)

### 1. Dual-Track Evaluation Architecture
The evaluation framework maintains a strict mathematical and conceptual separation between two evaluation tracks:
- **Track 1: Econometric Multi-Horizon Close-to-Close Forecasting Track**:
  - Focuses on predicting future close price drift for multi-step horizons ($n+1$ through $n+5$).
  - Target: $\text{target\_return\_h}_h = \frac{Close_{t+h}}{Close_t} - 1.0$.
  - Baseline: Naïve zero-return persistence ($\hat{r}_{t, h} = 0.0 \implies \hat{y}_{t, h} = Close_t$).
  - Evaluated on RMSE, MAE, and Directional Accuracy.
- **Track 2: Quantitative Trading Strategy Next-Open Backtesting Track**:
  - Focuses on actual executable return generated by a causal Next-Open trading policy.
  - Dedicated Strategy Target:
    $$\text{target\_trade\_return} = \frac{Open_{t+2}}{Open_{t+1}} - 1.0 = \frac{\text{Open.shift}(-2)}{\text{Open.shift}(-1)} - 1.0$$
  - Signal Timing: Generated at Close[D] ($t$) after observing technical indicators and daily sentiment finalized by 23:59:59 UTC.
  - Execution: Entered at $\text{Open}[D+1]$ ($t+1$) and held until exit/rebalance at $\text{Open}[D+2]$ ($t+2$).
  - Evaluated via out-of-sample backtest with a 10 bps transaction fee per unit turnover and synchronized Buy-and-Hold benchmark over identical return intervals.

### 2. Chronological Walk-Forward Design & Target Boundary Purging
- **Rolling Window Configuration**:
  - `train_window_days = 90`: 90-day rolling lookback for fold-local model fitting (`expanding=False`).
  - `test_window_days = 15`: 15-day out-of-sample evaluation per fold.
  - `step_days = 15`: Non-overlapping test blocks ensuring each observation is evaluated exactly once in aggregate metrics.
- **Exact Purge Rule for Close-to-Close Targets ($h \in 1..5$)**:
  - Training observation at index $t$ is eligible for horizon $h$ if and only if its forward target date strictly precedes the test fold: $(t + h) < \text{test\_start\_idx}$.
  - Rows where $(t + h) \ge \text{test\_start\_idx}$ are masked to `NaN`.
  - Eligible training counts for 90-day folds: $h=1 \implies 89$, $h=2 \implies 88$, $h=3 \implies 87$, $h=4 \implies 86$, $h=5 \implies 85$.
- **Exact Purge Rule for Strategy Trade Target**:
  - Signal row at index $t$ uses $Open_{t+1}$ and $Open_{t+2}$; its target endpoint is $t+2$.
  - To ensure its target endpoint is strictly before the test fold's first row ($\text{test\_start\_idx}$):
    $$(t + 2) < \text{test\_start\_idx} \iff t \le \text{test\_start\_idx} - 3$$
  - Training rows at indices $\text{test\_start\_idx} - 2$ and $\text{test\_start\_idx} - 1$ are purged (masked to `NaN`).
  - For every 90-day rolling training fold, exactly **2 rows are purged**, leaving exactly **88 eligible training observations**.
- **Fold-Local Preprocessing Isolation**:
  - Median imputation and standard scaling are fitted strictly on eligible non-leaking training rows.
  - Test observations are transformed using training statistics.

### 3. Causal Next-Open Strategy Backtest Mechanics
A deterministic, lookahead-free simulation using actual OHLC prices:
- **Signal Timing**: Prediction $\hat{r}_{\text{trade}}[D]$ is calculated after Close of day $D$ (23:59:59 UTC) using only features known up to Close $D$.
- **Execution Timing**: Market order executes at the next available trading price: **$\text{Open}_{D+1}$** (at $D+1\text{ 00:00:00 UTC}$).
- **Holding Period**: Position is held across the 24-hour daily cycle until **$\text{Open}_{D+2}$** (rebalance time).
- **Realized Asset Return**: Strictly computed from actual Open prices:
  $$R_{\text{asset}} = \frac{\text{Open}_{D+2} - \text{Open}_{D+1}}{\text{Open}_{D+1}}$$
- **Position Mapping**:
  - Long ($+1.0$) if $\hat{r}_{\text{trade}} > \text{threshold}$.
  - Flat ($0.0$) if $|\hat{r}_{\text{trade}}| \le \text{threshold}$ or if prediction is NaN.
  - Short ($-1.0$) if $\hat{r}_{\text{trade}} < -\text{threshold}$ (when `allow_short=True`).
  - Default: Long-only / Cash (`threshold=0.0`, `allow_short=False`).
- **Transaction Costs, Turnover & Terminal Liquidation**:
  - Rebalance occurs at $\text{Open}_{D+1}$: turnover $= |\text{pos}_{D+1} - \text{pos}_D|$.
  - Fee: 10 bps ($0.0010$) per unit turnover deducted from realized return.
  - Terminal liquidation: If the evaluation concludes with an open position ($\text{pos} \neq 0$), a modeled terminal liquidation fee is charged at the evaluation endpoint.
- **Benchmark Synchronization & Fee Disclosure**:
  - The Buy-and-Hold benchmark executes over identical return intervals ($\text{Open}_{D+1}$ to $\text{Open}_{D+2}$).
  - Benchmark performance is reported **gross of transaction fees** (0 initial fee, 0 trades), whereas the strategy is reported **net of 10 bps turnover fees**.
- **Win-Rate Accounting**:
  - Reports active-day win rate (positive returns while in market: $\text{win\_days} / \text{active\_days}$) alongside all-period win rate ($\text{win\_days} / N$).
- **Maximum Drawdown Baseline**:
  - Equity path explicitly includes the starting capital baseline of $1.0$, accurately capturing any initial drawdown.
- **Insufficient Future Observations Handling**:
  - If a signal date near the end of the series lacks future Open prices ($t+1$ or $t+2$ exceeds dataset boundary), it is cleanly skipped without index errors or silent row misalignment.

### 4. Real-Data vs Synthetic-Data Separation
- **Offline Unit Tests**: All 177 automated unit tests run strictly offline, using mocks and explicitly tagged in-memory synthetic fixtures (`is_synthetic=True`).
- **Real BTC Price Evaluation**: Tested on local real historical BTC daily prices (`data/raw/btc_usd_daily_raw.csv`, 179 contiguous days).
- **Collected Real Social Sentiment**: 10 genuine posts collected via Apify (`data/raw/x_posts_btc_raw.json` and `data/raw/x_posts_btc_normalized.csv`).
- **Coverage Limitation**: The 10 collected posts represent a point snapshot (`2026-10-18`). Because continuous daily historical social sentiment across the full 180-day backtest window (`2026-04-21` to `2026-10-16`) was not available, the missing-data imputer cleanly preserved price indicators for days without posts. As an honest consequence, the historical backtest performance of the sentiment model matches the price-only baseline (`diff = 0.0`). No fabricated performance claims or synthetic substitutions are made.
- **Backtest Limitations**: A short 75-day historical backtest does not imply long-term profitability or trading edge.

## Quantitative Walk-Forward Evaluation Results

### Forecast Accuracy Across Horizons ($n+1$ through $n+5$)

| Horizon | Sample Count | Ridge RMSE | Persistence RMSE | Diff (RMSE) | Ridge MAE | Persistence MAE | Diff (MAE) | Directional Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$n+1$** | 75 | 1651.87 | 1566.98 | +84.89 | 1050.70 | 974.69 | +76.01 | 48.00% |
| **$n+2$** | 75 | 2431.60 | 2285.36 | +146.24 | 1697.98 | 1496.33 | +201.66 | 49.33% |
| **$n+3$** | 75 | 3308.31 | 2916.35 | +391.96 | 2266.52 | 1899.71 | +366.81 | 54.67% |
| **$n+4$** | 75 | 4066.77 | 3563.61 | +503.16 | 2725.15 | 2283.68 | +441.47 | 56.00% |
| **$n+5$** | 75 | 4696.78 | 4130.50 | +566.29 | 3107.85 | 2611.18 | +496.67 | 54.67% |

### Strategy Performance: Price-Only Baseline vs. Sentiment-Enabled Model

| Metric | Price-Only Baseline | Sentiment-Enabled Model | Difference | Buy-and-Hold Benchmark |
|---|:---:|:---:|:---:|:---:|
| **Cumulative Return** | +1.04% | +1.04% | 0.00% | +29.95% (Gross) |
| **Annualized Return** | +5.15% | +5.15% | 0.00% | +257.87% |
| **Annualized Volatility** | 27.33% | 27.33% | 0.00% | 40.33% |
| **Sharpe Ratio** | 0.1883 | 0.1883 | 0.0000 | 6.3936 |
| **Sortino Ratio** | 0.3157 | 0.3157 | 0.0000 | 14.3043 |
| **Max Drawdown** | 10.44% | 10.44% | 0.00% | 6.96% |
| **Number of Trades** | 32 | 32 | 0 | 0 |
| **Market Exposure** | 44.00% | 44.00% | 0.00% | 100.00% |
| **Win Rate (Active Days)** | 48.48% (16/33) | 48.48% (16/33) | 0.00% | 52.00% (39/75) |
| **Win Rate (All Periods)** | 21.33% (16/75) | 21.33% (16/75) | 0.00% | 52.00% (39/75) |

*Accounting Notes:*
1. Strategy metrics are **net of modeled 10 bps ($0.0010$) turnover fees** and terminal liquidation fee.
2. The Buy-and-Hold benchmark executes over the same Next-Open trading cycle ($\text{Open}_{D+1}$ to $\text{Open}_{D+2}$) and is reported **gross of transaction fees**.
3. All-period win rate accounts for all 75 out-of-sample days, ensuring flat (cash) days are not mislabeled as wins.

## Getting Started

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration & Credentials Management
Copy `.env.example` to `.env` and populate your API credentials:
```bash
cp .env.example .env
```
Inside `.env`:
```text
APIFY_API_TOKEN=your_actual_token_here
```
> [!IMPORTANT]
> **Credential Security**: `.env` is strictly excluded by `.gitignore` and must never be committed. The codebase loads tokens via `dotenv` without logging, printing, or including secrets in error messages.

### 3. Running Automated Tests
Run the complete test suite (177 offline tests passing):
```bash
pytest -v
```

### 4. Running Walk-Forward Evaluation
Execute the integrated walk-forward evaluation and backtest:
```bash
# Auto mode (evaluates price baseline and sentiment track):
python -m src.evaluation.run_evaluation --mode auto

# Price-only baseline:
python -m src.evaluation.run_evaluation --mode price_only

# Sentiment-only track:
python -m src.evaluation.run_evaluation --mode sentiment

# Offline test with synthetic fixture (isolated testing only):
python -m src.evaluation.run_evaluation \
  --posts-path tests/fixtures/synthetic_x_posts.json \
  --allow-synthetic \
  --mode auto
```

### 5. Running Controlled Data Ingestion
```bash
# Ingest historical BTC price bars (yfinance):
python -m src.ingestion.price_data

# Ingest historical X posts (Apify - uses configured single query & 10 item cap):
python -m src.ingestion.x_posts --search-term Bitcoin --max-items 10
```
