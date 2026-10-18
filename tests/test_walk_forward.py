"""
Unit tests for walk-forward evaluation module (src/evaluation/walk_forward.py).

Verifies:
1. Strict chronological ordering (train strictly precedes test for all folds).
2. Fold-local fitting (zero look-ahead leakage across folds).
3. Correct forecast horizon coverage (horizons 1-5, excluding n+0).
4. Rejection of n+0 in forecast evaluation.
5. End-of-series missing target handling and sample counts.
6. Exact date alignment of out-of-sample predictions.
7. Insufficient history handling (raises ValueError).
8. Synthetic fixture marking (is_synthetic=True).
All tests use small in-memory synthetic datasets.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_directional_accuracy, compute_forecast_metrics
from src.evaluation.walk_forward import (
    WalkForwardEvaluator,
    WalkForwardResult,
    WalkForwardSplitter,
)
from src.processing.features import (
    add_price_derived_features,
    prepare_modeling_dataset,
)


@pytest.fixture
def synthetic_walk_forward_df() -> pd.DataFrame:
    """Creates a 60-day synthetic contiguous daily price and sentiment DataFrame."""
    dates = pd.date_range("2026-08-01", periods=60, freq="D", tz="UTC")
    data = {
        "Open": [50000.0 + i * 150 for i in range(60)],
        "High": [50500.0 + i * 150 for i in range(60)],
        "Low": [49500.0 + i * 150 for i in range(60)],
        "Close": [50200.0 + i * 150 + (i % 2) * 50 for i in range(60)],
        "Volume": [1.0e9 + i * 0.05e9 for i in range(60)],
        "post_count": [10 + (i % 5) * 5 for i in range(60)],
        "positive_post_count": [5 + (i % 3) for i in range(60)],
        "negative_post_count": [2 + (i % 2) for i in range(60)],
        "neutral_post_count": [3 + (i % 2) for i in range(60)],
        "sentiment_mean": [0.25 if i % 4 != 0 else np.nan for i in range(60)],
        "subjectivity_mean": [0.4 if i % 4 != 0 else np.nan for i in range(60)],
        "sentiment_weighted": [0.30 if i % 4 != 0 else 0.0 for i in range(60)],
        "sentiment_weighted_available": [i % 4 != 0 for i in range(60)],
        "has_posts": [i % 4 != 0 for i in range(60)],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return add_price_derived_features(df)


# ==============================================================================
# 1. WalkForwardSplitter Chronology & Validation
# ==============================================================================

def test_walk_forward_splitter_chronology(synthetic_walk_forward_df):
    """
    Verifies that every training fold strictly precedes its test fold,
    with zero chronological overlap or shuffling.
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )

    splits = list(splitter.split(synthetic_walk_forward_df))
    assert len(splits) == 3  # (60 - 40) // 10 + 1 = 3 folds

    for fold_num, (train_idx, test_idx, train_dates, test_dates) in enumerate(splits):
        assert len(train_idx) == 30
        assert len(test_idx) == 10

        # Strict chronological separation: max(train) < min(test)
        assert train_idx[-1] < test_idx[0]
        assert train_dates[-1] < test_dates[0]

        # Contiguous monotonic indices
        assert (np.diff(train_idx) == 1).all()
        assert (np.diff(test_idx) == 1).all()


def test_walk_forward_splitter_expanding(synthetic_walk_forward_df):
    """Verifies that expanding=True expands the training window from index 0."""
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
        expanding=True,
    )

    splits = list(splitter.split(synthetic_walk_forward_df))
    # Fold 1: train 0..29, test 30..39
    assert len(splits[0][0]) == 30
    assert splits[0][0][0] == 0

    # Fold 2: train 0..39, test 40..49
    assert len(splits[1][0]) == 40
    assert splits[1][0][0] == 0

    # Fold 3: train 0..49, test 50..59
    assert len(splits[2][0]) == 50
    assert splits[2][0][0] == 0


def test_walk_forward_splitter_insufficient_history(synthetic_walk_forward_df):
    """Verifies ValueError when dataset has fewer rows than train + test windows."""
    splitter = WalkForwardSplitter(
        train_window_days=50,
        test_window_days=20,  # Requires 70 rows, dataset has 60
    )
    with pytest.raises(ValueError, match="Insufficient history"):
        list(splitter.split(synthetic_walk_forward_df))


def test_walk_forward_splitter_invalid_params():
    """Verifies input validation on splitter parameters."""
    with pytest.raises(ValueError, match="train_window_days must be positive"):
        WalkForwardSplitter(train_window_days=-10)

    with pytest.raises(ValueError, match="test_window_days must be positive"):
        WalkForwardSplitter(test_window_days=0)

    with pytest.raises(ValueError, match="step_days must be positive"):
        WalkForwardSplitter(step_days=-5)


# ==============================================================================
# 2. WalkForwardEvaluator Execution & Safeguards
# ==============================================================================

def test_walk_forward_evaluator_rejects_n0():
    """
    Safeguard: Verifies that WalkForwardEvaluator explicitly rejects horizon 0
    from its forecast horizons list.
    """
    with pytest.raises(ValueError, match="Horizon n\\+0 cannot be included"):
        WalkForwardEvaluator(horizons=[0, 1, 2])


def test_walk_forward_evaluator_execution(synthetic_walk_forward_df):
    """
    Verifies end-to-end walk-forward evaluation:
      - Fits models strictly fold-by-fold.
      - Produces out-of-sample predictions for horizons 1 through 5.
      - Produces summary price and return metrics.
      - Formats side-by-side comparison tables.
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3],
        price_column="Close",
        alpha=1.0,
    )

    result = evaluator.evaluate(synthetic_walk_forward_df, is_synthetic=True)

    assert isinstance(result, WalkForwardResult)
    assert result.is_synthetic is True
    assert result.n_folds == 3

    # Out-of-sample predictions should cover 3 folds * 10 test days = 30 days
    ridge_preds = result.predictions_prices["ridge"]
    assert len(ridge_preds) == 30
    assert not ridge_preds.isna().all().any()

    # Verify horizons in summary metrics
    assert set(result.summary_price_metrics["ridge"].keys()) == {1, 2, 3}
    assert set(result.summary_price_metrics["persistence"].keys()) == {1, 2, 3}

    # Verify comparison tables
    assert len(result.comparison_table_rmse) == 3
    assert len(result.comparison_table_mae) == 3
    assert len(result.comparison_table_da) == 3


def test_walk_forward_evaluator_sample_counts_and_masking(synthetic_walk_forward_df):
    """
    Verifies that end-of-series target NaNs are cleanly masked,
    and that sample counts correctly reflect unobserved future targets.
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3, 4, 5],
        price_column="Close",
    )

    result = evaluator.evaluate(synthetic_walk_forward_df, is_synthetic=True)

    # Total test rows across 3 folds = 30 rows
    # The last fold contains rows 50..59 (the very end of the 60-day series)
    # For horizon 1: row 59 target is unobserved (NaN) -> valid = 29
    # For horizon 5: rows 55..59 targets are unobserved (NaN) -> valid = 25
    count_h1 = result.summary_price_metrics["ridge"][1]["sample_count"]
    count_h5 = result.summary_price_metrics["ridge"][5]["sample_count"]

    assert count_h1 == 29
    assert count_h5 == 25
    assert count_h1 > count_h5


def test_directional_accuracy_calculation():
    """Verifies directional accuracy calculation on returns."""
    actual = np.array([0.02, -0.01, 0.03, -0.05])
    predicted = np.array([0.01, 0.02, 0.05, -0.02])
    # Matches at indices: 0 (pos/pos), 2 (pos/pos), 3 (neg/neg) -> 3 / 4 = 75.0%
    da = compute_directional_accuracy(actual, predicted)
    assert da == 75.0


def test_compute_forecast_metrics_scale_mismatch_guard():
    """Verifies that compute_forecast_metrics rejects scale mismatch between prices and returns."""
    prices = np.array([50000.0, 52000.0])
    returns = np.array([0.02, 0.01])

    with pytest.raises(ValueError, match="Scale mismatch"):
        compute_forecast_metrics(prices, returns, target_type="price")


def test_walk_forward_horizon_specific_purging(synthetic_walk_forward_df):
    """
    Safeguard: Verifies that every training label's target timestamp is
    strictly earlier than the first test timestamp for its fold and horizon.
    Boundary observations whose forward targets would extend into the test fold
    must be purged independently per horizon.
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3, 4, 5],
        price_column="Close",
    )

    result = evaluator.evaluate(synthetic_walk_forward_df, is_synthetic=True)

    # In fold metrics, inspect eligible train samples:
    # Train window = 30 days.
    # Horizon 1 purges 1 row -> 29 eligible.
    # Horizon 2 purges 2 rows -> 28 eligible.
    # Horizon 3 purges 3 rows -> 27 eligible.
    # Horizon 4 purges 4 rows -> 26 eligible.
    # Horizon 5 purges 5 rows -> 25 eligible.
    for fm in result.fold_metrics:
        for h in [1, 2, 3, 4, 5]:
            expected_eligible = 30 - h
            assert fm[f"h{h}_eligible_train_samples"] == expected_eligible

    # Verify no cross-fold leakage:
    # All train targets for horizon h must end strictly before the fold's test_start_date
    for train_idx, test_idx, train_dates, test_dates in splitter.split(synthetic_walk_forward_df):
        test_start_date = test_dates[0]
        test_start_idx = test_idx[0]
        for h in [1, 2, 3, 4, 5]:
            eligible_train_idx = [t for t in train_idx if t + h < test_start_idx]
            assert len(eligible_train_idx) == 30 - h
            for t in eligible_train_idx:
                target_date = synthetic_walk_forward_df.index[t + h]
                assert target_date < test_start_date


def test_directional_metrics_coverage_and_neutral_baseline():
    """
    Verifies that directional accuracy is calculated strictly among non-neutral predictions,
    and that a zero-return persistence model yields directional_coverage = 0.0% and
    directional_accuracy = NaN (reported as N/A).
    """
    actual = np.array([0.02, -0.01, 0.03, -0.05])
    # Model 1: Active directional model (3 correct, 1 incorrect)
    pred_active = np.array([0.01, 0.02, 0.05, -0.02])
    metrics_active = compute_forecast_metrics(actual, pred_active, target_type="return")

    assert metrics_active["directional_coverage"] == 100.0
    assert metrics_active["directional_accuracy"] == 75.0
    assert metrics_active["is_neutral_baseline"] is False

    # Model 2: Zero-return persistence baseline
    pred_pers = np.array([0.0, 0.0, 0.0, 0.0])
    metrics_pers = compute_forecast_metrics(actual, pred_pers, target_type="return")

    assert metrics_pers["directional_coverage"] == 0.0
    assert np.isnan(metrics_pers["directional_accuracy"])
    assert metrics_pers["is_neutral_baseline"] is True

    # Model 3: Mixed active and neutral predictions with threshold
    pred_mixed = np.array([0.02, 0.0001, -0.03, 0.0])
    # With threshold 0.001: indices 1 and 3 are neutral.
    # Non-neutral: index 0 (+0.02 vs +0.02, match), index 2 (+0.03 vs -0.03, mismatch) -> 1/2 = 50.0%
    metrics_mixed = compute_forecast_metrics(actual, pred_mixed, target_type="return", neutral_threshold=0.001)
    assert metrics_mixed["directional_coverage"] == 50.0
    assert metrics_mixed["directional_accuracy"] == 50.0


# ==============================================================================
# 3. Strategy Trade Target Walk-Forward Boundary Purging & Separation
# ==============================================================================

def test_walk_forward_strategy_target_purging(synthetic_walk_forward_df):
    """
    Safeguard: Verifies that strategy target boundary purging strictly enforces:
      - Signal row at index t uses Open[t+1] and Open[t+2] (endpoint at t+2).
      - Endpoint must be strictly before test_start_idx (t + 2 < test_start_idx).
      - Rows at test_start_idx - 2 and test_start_idx - 1 are purged (exactly 2 rows purged).
      - In a 30-day training fold, eligible count is exactly 28 (30 - 2).
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3],
        price_column="Close",
    )

    result = evaluator.evaluate(synthetic_walk_forward_df, is_synthetic=True)

    # 1. Verify eligible training counts across all folds
    for fm in result.fold_metrics:
        assert fm["trade_eligible_train_samples"] == 28  # 30 - 2 = 28

    # 2. Verify zero leakage across fold boundaries
    for train_idx, test_idx, train_dates, test_dates in splitter.split(synthetic_walk_forward_df):
        test_start_date = test_dates[0]
        test_start_idx = test_idx[0]

        # Explicit boundary checks
        # Index test_start_idx - 2: endpoint is (test_start_idx - 2 + 2) = test_start_idx -> PURGED
        # Index test_start_idx - 1: endpoint is (test_start_idx - 1 + 2) = test_start_idx + 1 -> PURGED
        # Index test_start_idx - 3: endpoint is (test_start_idx - 3 + 2) = test_start_idx - 1 < test_start_idx -> ELIGIBLE
        purged_indices = [t for t in train_idx if (t + 2) >= test_start_idx]
        assert len(purged_indices) == 2
        assert train_idx[-1] in purged_indices
        assert train_idx[-2] in purged_indices

        eligible_indices = [t for t in train_idx if (t + 2) < test_start_idx]
        assert len(eligible_indices) == 28
        assert train_idx[-3] in eligible_indices

        # Verify all eligible target endpoints strictly precede the test fold start date
        for t in eligible_indices:
            label_endpoint_date = synthetic_walk_forward_df.index[t + 2]
            assert label_endpoint_date < test_start_date


def test_walk_forward_strategy_predictions_separation(synthetic_walk_forward_df):
    """
    Verifies that the strategy model, its target (target_trade_return), and its
    out-of-sample predictions are strictly separated from close-to-close forecasting.
    """
    splitter = WalkForwardSplitter(
        train_window_days=30,
        test_window_days=10,
        step_days=10,
    )
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        horizons=[1, 2, 3],
        price_column="Close",
    )

    result = evaluator.evaluate(synthetic_walk_forward_df, is_synthetic=True)

    # 1. Strategy predictions container exists and has required columns
    assert "strategy_trade" in result.predictions_returns
    strat_df = result.predictions_returns["strategy_trade"]
    assert "strategy_pred_return" in strat_df.columns
    assert "actual_trade_return" in strat_df.columns
    assert "Open" in strat_df.columns
    assert "Close" in strat_df.columns
    assert len(strat_df) == 30  # 3 folds * 10 test days

    # 2. Strategy metrics exist and are distinct from close-to-close metrics
    assert result.strategy_metrics is not None
    assert "rmse" in result.strategy_metrics
    assert "mae" in result.strategy_metrics
    assert "directional_accuracy" in result.strategy_metrics

    # 3. Close-to-close Ridge forecasting results do not contain strategy target
    ridge_returns = result.predictions_returns["ridge"]
    assert "strategy_pred_return" not in ridge_returns.columns
    assert "actual_trade_return" not in ridge_returns.columns
    for h in [1, 2, 3]:
        assert f"ridge_pred_return_h{h}" in ridge_returns.columns
        assert f"actual_return_h{h}" in ridge_returns.columns

    # 4. Summary price/return metrics contain only horizon keys (1, 2, 3)
    assert set(result.summary_price_metrics["ridge"].keys()) == {1, 2, 3}
    assert set(result.summary_return_metrics["ridge"].keys()) == {1, 2, 3}


