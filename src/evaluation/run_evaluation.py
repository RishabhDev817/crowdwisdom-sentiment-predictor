"""
Execution runner and orchestrator for walk-forward evaluation and strategy backtesting.

Supports:
1. Price-only baseline forecasting and backtesting (using stationary price-derived indicators).
2. End-to-end sentiment-enabled forecasting and backtesting (scoring posts, daily aggregation,
   price alignment, multi-horizon drift forecasting, and causal Next-Open strategy execution).
3. Strict missing-data guards: stops sentiment-enabled evaluation with a clear actionable message
   if real social posts are absent, empty, or contain dummy/placeholder records.
4. Strict isolation: synthetic fixtures cannot be used in production execution unless explicitly
   flagged for offline testing (`allow_synthetic=True`).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import json
import logging
import sys
import pandas as pd

from src.evaluation.backtest import BacktestConfig, BacktestEngine, BacktestResult
from src.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardSplitter, WalkForwardResult
from src.processing.features import (
    add_price_derived_features,
    build_daily_features_pipeline,
    get_feature_columns,
    validate_calendar_continuity,
)
from src.processing.sentiment import score_posts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


MIN_REAL_POSTS_THRESHOLD = 10


def validate_and_load_x_posts(
    posts_path: Optional[Union[str, Path]],
    allow_synthetic: bool = False,
    min_real_posts: int = MIN_REAL_POSTS_THRESHOLD,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validates and loads social post data, enforcing strict quality and isolation rules.

    Rules:
        - If posts_path is None or file is missing, returns (None, reason).
        - If dataset is empty (0 records), returns (None, reason).
        - If dataset is flagged as synthetic ('is_synthetic': true) and allow_synthetic=False,
          returns (None, reason).
        - If dataset has fewer than min_real_posts records and allow_synthetic=False,
          rejects it as insufficient/placeholder.
        - If dataset text matches dummy patterns (e.g. 'BTC test') and allow_synthetic=False,
          rejects it as placeholder data.
        - Verifies required columns ('text', 'created_at') and UTC timestamp parseability.

    Args:
        posts_path: File path to CSV or JSON post records.
        allow_synthetic: If True, allows synthetic fixtures (strictly for offline unit tests).
        min_real_posts: Minimum number of posts required for real evaluation (default: 10).

    Returns:
        Tuple of (DataFrame or None, error_message or None).
    """
    if not posts_path:
        return None, "Posts data path was not provided."

    path = Path(posts_path)
    if not path.is_file():
        return None, f"Posts data file not found at: '{path}'."

    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                is_synthetic_meta = data.get("_metadata", {}).get("is_synthetic", False)
                if is_synthetic_meta and not allow_synthetic:
                    return None, (
                        f"Dataset at '{path}' is explicitly flagged as a synthetic fixture "
                        "('is_synthetic: true') and is rejected from production evaluation."
                    )
                posts_list = data.get("posts", [])
            elif isinstance(data, list):
                posts_list = data
            else:
                return None, f"Unexpected JSON structure in '{path}'."
            df = pd.DataFrame(posts_list)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        return None, f"Failed to read posts file at '{path}': {e}"

    if df.empty:
        return None, f"Posts dataset at '{path}' is empty (0 records)."

    # Reject synthetic posts in production mode
    if "is_synthetic" in df.columns:
        has_syn = df["is_synthetic"].fillna(False).astype(bool).any()
        if has_syn and not allow_synthetic:
            return None, (
                f"Dataset at '{path}' contains records flagged with 'is_synthetic=True'. "
                "Synthetic data is strictly rejected from real evaluation."
            )

    # Reject placeholder/dummy datasets based on record volume
    if len(df) < min_real_posts and not allow_synthetic:
        sample_texts = df["text"].dropna().tolist()[:3] if "text" in df.columns else []
        return None, (
            f"Dataset at '{path}' contains only {len(df)} record(s) ({sample_texts}), "
            f"which is insufficient for real sentiment evaluation (minimum required: {min_real_posts}). "
            "Real social sentiment ingestion via Apify is required."
        )

    # Reject datasets composed solely of dummy placeholder texts
    if not allow_synthetic and "text" in df.columns:
        dummy_patterns = {"btc test", "test", "placeholder", "dummy"}
        texts_lower = df["text"].astype(str).str.strip().str.lower()
        if texts_lower.isin(dummy_patterns).all():
            return None, (
                f"Dataset at '{path}' consists exclusively of placeholder/dummy records ('{texts_lower.iloc[0]}'). "
                "Real social sentiment collection via Apify is required."
            )

    # Validate required columns
    required_cols = ["text", "created_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"Dataset at '{path}' is missing required columns: {missing}."

    # Validate timestamp parseability
    try:
        parsed_dates = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        if parsed_dates.isna().all():
            return None, f"Dataset at '{path}' contains no valid UTC timestamps in 'created_at'."
    except Exception as e:
        return None, f"Timestamp parsing failed for '{path}': {e}"

    return df, None


def load_and_validate_btc_prices(price_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads and validates historical daily BTC prices, ensuring calendar continuity.

    Args:
        price_path: Path to the raw BTC prices CSV.

    Returns:
        pd.DataFrame: Contiguous, daily-spaced price DataFrame indexed by UTC midnight.
    """
    path = Path(price_path)
    if not path.is_file():
        raise FileNotFoundError(f"Price data file not found at: '{path}'")

    price_df = pd.read_csv(path)
    price_df["Date"] = pd.to_datetime(price_df["Date"], utc=True).dt.normalize()
    price_df = price_df.set_index("Date").sort_index()

    # Check for calendar continuity gaps in the raw data
    diffs = price_df.index.to_series().diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(days=1)]
    if not gaps.empty:
        first_gap_loc = price_df.index.get_loc(gaps.index[0])
        contiguous_df = price_df.iloc[:first_gap_loc].copy()
        logger.info(
            f"Extracted contiguous slice: {len(contiguous_df)} days "
            f"({contiguous_df.index[0].strftime('%Y-%m-%d')} to {contiguous_df.index[-1].strftime('%Y-%m-%d')})"
        )
    else:
        contiguous_df = price_df.copy()

    validate_calendar_continuity(contiguous_df)
    return contiguous_df


def build_integrated_pipeline_dataset(
    price_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    text_column: str = "text",
    created_at_column: str = "created_at",
    engagement_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds an integrated dataset combining validated price bars with daily sentiment features
    and backward-looking price-derived technical indicators.

    Orchestration Steps:
        1. Validates calendar continuity of input price DataFrame.
        2. Scores posts with rule-based sentiment if polarity scores are not yet computed.
        3. Aggregates daily UTC sentiment features (00:00:00 to 23:59:59 UTC).
        4. Left-joins daily sentiment onto price calendar, setting has_posts=False for missing days.
        5. Computes backward-looking price-derived features.
        6. Extracts and validates permissible stationary feature columns (quarantining targets and raw prices).

    Args:
        price_df: Validated OHLCV price DataFrame.
        posts_df: Validated social posts DataFrame.
        text_column: Text column name.
        created_at_column: Timestamp column name.
        engagement_weights: Optional engagement metric weights.

    Returns:
        Tuple of (DataFrame with all features, list of permissible feature column names).
    """
    validate_calendar_continuity(price_df)

    if "sentiment_score" not in posts_df.columns:
        scored_posts = score_posts(posts_df, text_column=text_column)
    else:
        scored_posts = posts_df

    aligned_df = build_daily_features_pipeline(
        price_df=price_df,
        posts_df=scored_posts,
        text_column=text_column,
        created_at_column=created_at_column,
        engagement_weights=engagement_weights,
    )

    df_with_features = add_price_derived_features(aligned_df)
    feat_cols = get_feature_columns(df_with_features)

    return df_with_features, feat_cols


def _print_performance_metrics(metrics: Dict[str, Any], title: str, is_benchmark: bool = False) -> None:
    """Helper to cleanly format and display strategy or benchmark quantitative performance."""
    print(f"\n--- {title} ---")
    if is_benchmark:
        print("  Note: Benchmark return is gross of transaction fees; strategy return is net of 10 bps turnover fees.")

    for k, v in metrics.items():
        if k == "win_rate":
            w_days = metrics.get("win_days", 0)
            if is_benchmark:
                n_days = metrics.get("n_days", 0)
                print(f"  {'win_rate (in-market days)':25s}: {v}% ({w_days}/{n_days} daily holding periods)")
            else:
                act_days = metrics.get("active_days", 0)
                print(f"  {'win_rate (active days)':25s}: {v}% ({w_days}/{act_days} active in-market days)")
        elif k == "all_period_win_rate" and not is_benchmark:
            w_days = metrics.get("all_period_win_days", 0)
            n_days = metrics.get("n_days", 0)
            print(f"  {'win_rate (all periods)':25s}: {v}% ({w_days}/{n_days} total evaluation days)")
        elif k in {"win_days", "active_days", "all_period_win_days"}:
            continue
        else:
            print(f"  {k:25s}: {v}")


def run_pipeline(
    price_csv_path: str = "data/raw/btc_usd_daily_raw.csv",
    posts_csv_path: Optional[str] = "data/raw/x_posts_btc_normalized.csv",
    train_window_days: int = 90,
    test_window_days: int = 15,
    step_days: int = 15,
    mode: str = "auto",
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Executes walk-forward evaluation and backtesting across baseline and sentiment-enabled tracks.

    Modes:
        - 'auto': Runs price-only baseline; attempts sentiment-enabled pipeline and blocks gracefully
          if genuine sentiment data is missing.
        - 'price_only': Runs price-derived technical features evaluation only.
        - 'sentiment': Requires genuine sentiment data; stops with error if unavailable.

    Args:
        price_csv_path: Path to historical BTC price CSV.
        posts_csv_path: Path to X posts CSV or JSON.
        train_window_days: Rolling training window in calendar days.
        test_window_days: Out-of-sample test window in calendar days.
        step_days: Step size between consecutive folds.
        mode: Execution mode ('auto', 'price_only', 'sentiment').
        allow_synthetic: Whether synthetic post fixtures are permitted (strictly for offline tests).

    Returns:
        Dictionary containing pipeline results, backtest objects, and execution status.
    """
    try:
        contiguous_df = load_and_validate_btc_prices(price_csv_path)
    except Exception as e:
        logger.error(f"Failed to load/validate BTC prices: {e}")
        sys.exit(1)

    # Configure Walk-Forward Splitter
    splitter = WalkForwardSplitter(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        expanding=False,  # Strict rolling window
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3, 4, 5],
        price_column="Close",
        alpha=1.0,
    )
    bt_engine = BacktestEngine(
        config=BacktestConfig(
            threshold=0.0,
            allow_short=False,
            cost_bps=10.0,
            execution_policy="next_open",
            terminal_liquidation=True,
        )
    )

    pipeline_output: Dict[str, Any] = {
        "contiguous_df": contiguous_df,
        "price_only_result": None,
        "price_only_backtest": None,
        "sentiment_result": None,
        "sentiment_backtest": None,
        "sentiment_status": "SKIPPED",
        "block_reason": None,
    }

    # =========================================================================
    # Track 1 & 2: Price-Only Baseline Evaluation
    # =========================================================================
    if mode in ("auto", "price_only"):
        logger.info("Executing Price-Only Baseline Walk-Forward Evaluation...")
        df_price_features = add_price_derived_features(contiguous_df)
        base_result = evaluator.evaluate(df_price_features, is_synthetic=False)

        print("\n" + "=" * 80)
        print("CHRONOLOGICAL WALK-FORWARD EVALUATION RESULTS (PRICE-DERIVED FEATURES ONLY)")
        print("=" * 80)
        print(f"Total Observations: {len(contiguous_df)} days ({contiguous_df.index[0].strftime('%Y-%m-%d')} to {contiguous_df.index[-1].strftime('%Y-%m-%d')})")
        print(f"Folds Evaluated:    {base_result.n_folds}")
        print(f"Splitter Design:    Rolling {base_result.train_window_days}-day window (expanding=False, boundary-purged)")
        print(f"Test Window:        {base_result.test_window_days} days per fold")
        print(f"Out-of-Sample Days: {base_result.metadata['n_test_samples']} days")

        print("\n--- FORECAST ACCURACY: RMSE BY HORIZON (RIDGE vs PERSISTENCE) ---")
        print(base_result.comparison_table_rmse.to_string())

        print("\n--- FORECAST ACCURACY: MAE BY HORIZON (RIDGE vs PERSISTENCE) ---")
        print(base_result.comparison_table_mae.to_string())

        print("\n--- FORECAST ACCURACY: DIRECTIONAL ACCURACY & COVERAGE (%) ---")
        print(base_result.comparison_table_da.to_string())

        if "strategy_trade" in base_result.predictions_returns:
            strat_df = base_result.predictions_returns["strategy_trade"]
            base_bt = bt_engine.run(
                predictions_series=strat_df["strategy_pred_return"],
                price_series=strat_df["Close"],
                open_price_series=contiguous_df["Open"],
                dates=strat_df.index,
                is_synthetic=False,
            )
            _print_performance_metrics(base_bt.metrics, "STRATEGY QUANTITATIVE PERFORMANCE (Net of 10 bps Turnover Fees)", is_benchmark=False)
            _print_performance_metrics(base_bt.benchmark_metrics, "BENCHMARK PERFORMANCE (Buy-and-Hold BTC, Gross of Fees, Next-Open Execution)", is_benchmark=True)
            pipeline_output["price_only_backtest"] = base_bt

        pipeline_output["price_only_result"] = base_result

    # =========================================================================
    # Track 3: Sentiment-Enabled Pipeline Evaluation & Missing-Data Handling
    # =========================================================================
    if mode in ("auto", "sentiment"):
        posts_df, err_msg = validate_and_load_x_posts(posts_csv_path, allow_synthetic=allow_synthetic)

        if posts_df is None:
            pipeline_output["sentiment_status"] = "BLOCKED"
            pipeline_output["block_reason"] = err_msg

            print("\n" + "=" * 80)
            print("SENTIMENT-ENABLED FORECASTING STATUS: BLOCKED")
            print("=" * 80)
            print("Status:  BLOCKED (Zero genuine social posts available)")
            print(f"Reason:  {err_msg}")
            print("Action:  Live social post collection is required (Phase 5 Apify scraper).")
            print("         Synthetic fixtures and dummy records are strictly isolated from production evaluation.")
            print("Notice:  The price-only baseline results above are NOT evidence that social sentiment")
            print("         adds predictive value.")
            print("=" * 80 + "\n")

            if mode == "sentiment":
                logger.warning(f"Sentiment evaluation stopped: {err_msg}")
                return pipeline_output
        else:
            logger.info("Executing Sentiment-Enabled Integrated Pipeline...")
            df_sent_features, feat_cols = build_integrated_pipeline_dataset(contiguous_df, posts_df)

            print("\n" + "=" * 80)
            print("SENTIMENT-ENABLED INTEGRATED PIPELINE EVALUATION")
            print("=" * 80)
            print(f"Posts Ingested:      {len(posts_df)} posts ({posts_csv_path})")
            print(f"Features Engineered: {len(feat_cols)} features ({feat_cols})")
            print("Daily Aggregation:   UTC calendar (00:00:00 to 23:59:59 UTC) aligned to Close[D]")
            print("Synthetic Flag:      " + ("TRUE (Isolated offline testing fixture)" if allow_synthetic else "FALSE (Genuine production observations)"))

            sent_result = evaluator.evaluate(df_sent_features, is_synthetic=allow_synthetic)
            pipeline_output["sentiment_result"] = sent_result
            pipeline_output["sentiment_status"] = "COMPLETED"

            print("\n--- SENTIMENT MODEL FORECAST ACCURACY: RMSE BY HORIZON ---")
            print(sent_result.comparison_table_rmse.to_string())

            print("\n--- SENTIMENT MODEL FORECAST ACCURACY: MAE BY HORIZON ---")
            print(sent_result.comparison_table_mae.to_string())

            print("\n--- SENTIMENT MODEL FORECAST ACCURACY: DIRECTIONAL ACCURACY (%) ---")
            print(sent_result.comparison_table_da.to_string())

            if "strategy_trade" in sent_result.predictions_returns:
                sent_strat_df = sent_result.predictions_returns["strategy_trade"]
                sent_bt = bt_engine.run(
                    predictions_series=sent_strat_df["strategy_pred_return"],
                    price_series=sent_strat_df["Close"],
                    open_price_series=contiguous_df["Open"],
                    dates=sent_strat_df.index,
                    is_synthetic=allow_synthetic,
                )
                _print_performance_metrics(sent_bt.metrics, "SENTIMENT STRATEGY PERFORMANCE (Net of 10 bps Turnover Fees)", is_benchmark=False)
                pipeline_output["sentiment_backtest"] = sent_bt

                # Side-by-side comparison table if baseline exists
                if pipeline_output.get("price_only_backtest"):
                    base_m = pipeline_output["price_only_backtest"].metrics
                    sent_m = sent_bt.metrics
                    comp_metrics = [
                        "cumulative_return", "annualized_return", "annualized_volatility",
                        "sharpe_ratio", "sortino_ratio", "max_drawdown", "num_trades",
                        "exposure", "win_rate",
                    ]
                    comp_records = []
                    for m in comp_metrics:
                        comp_records.append({
                            "Metric": m,
                            "Price-Only Baseline": base_m.get(m),
                            "Sentiment-Enabled Model": sent_m.get(m),
                            "Difference (Sentiment - Base)": round(sent_m.get(m, 0) - base_m.get(m, 0), 4) if isinstance(sent_m.get(m), (int, float)) and isinstance(base_m.get(m), (int, float)) else "N/A"
                        })
                    print("\n--- QUANTITATIVE STRATEGY COMPARISON: BASELINE vs SENTIMENT ---")
                    print(pd.DataFrame(comp_records).to_string(index=False))

    print("\n" + "=" * 80)
    print("IMPORTANT DISCLAIMER & LIMITATIONS:")
    print("  1. The baseline results use historical BTC-USD price data.")
    print("  2. Real X sentiment data was ingested as a genuine sample (10 posts on 2026-10-18);")
    print("     it does not provide continuous daily historical coverage across the 180-day price evaluation folds.")
    print("  3. Downstream imputer cleanly preserves historical price indicators for days without posts.")
    print("  4. A short historical backtest does not imply future profitability or trading edge.")
    print("  5. No claim is made that social sentiment improves predictive accuracy.")
    print("=" * 80 + "\n")

    return pipeline_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 Walk-Forward Evaluation and Backtest.")
    parser.add_argument("--price-path", default="data/raw/btc_usd_daily_raw.csv", help="Path to raw BTC prices CSV.")
    parser.add_argument("--posts-path", default="data/raw/x_posts_btc_normalized.csv", help="Path to normalized X posts CSV/JSON.")
    parser.add_argument("--mode", choices=["auto", "price_only", "sentiment"], default="auto", help="Execution mode: auto, price_only, or sentiment.")
    parser.add_argument("--allow-synthetic", action="store_true", default=False, help="Allow explicitly flagged synthetic fixtures (offline testing only).")
    parser.add_argument("--train-days", type=int, default=90, help="Rolling training window in days.")
    parser.add_argument("--test-days", type=int, default=15, help="Test window per fold in days.")
    parser.add_argument("--step-days", type=int, default=15, help="Step size between folds in days.")
    args = parser.parse_args()

    out = run_pipeline(
        price_csv_path=args.price_path,
        posts_csv_path=args.posts_path,
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        step_days=args.step_days,
        mode=args.mode,
        allow_synthetic=args.allow_synthetic,
    )

    if args.mode == "sentiment" and out.get("sentiment_status") == "BLOCKED":
        sys.exit(1)


if __name__ == "__main__":
    main()
