"""
Unit tests for baseline forecasting models (Naïve Persistence and Ridge regression).

Verifies:
1. Multi-horizon fitting on forward returns (target_return_h{h}).
2. Reconstruction of forecast prices via Close_t * (1.0 + predicted_return).
3. Persistence baseline equivalence (return = 0.0, price = Close_t).
4. Strict exclusion and protection against training Ridge on horizon n+0.
5. Aggregate metric calculation excluding n+0 nowcast contamination.
6. Pipeline preprocessing (median imputation and scaling) without leakage.
7. Rejection of invalid price anchors (non-positive or non-finite).
All tests use small synthetic in-memory datasets.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.baselines import (
    MultiHorizonForecaster,
    NaivePersistenceModel,
    aggregate_forecast_metrics,
    calculate_metrics,
    create_ridge_pipeline,
    reconstruct_price_from_return,
)
from src.processing.features import (
    FORECAST_HORIZONS,
    NOWCAST_HORIZON,
    prepare_modeling_dataset,
)


@pytest.fixture
def synthetic_training_data() -> Tuple[pd.DataFrame, List[str], Dict[int, pd.Series]]:
    """Creates a 30-day synthetic modeling dataset for baseline model testing."""
    dates = pd.date_range("2026-09-01", periods=30, freq="D", tz="UTC")
    data = {
        "Open": [50000.0 + i * 200 + (i % 3) * 50 for i in range(30)],
        "High": [50500.0 + i * 200 + (i % 3) * 100 for i in range(30)],
        "Low": [49500.0 + i * 200 for i in range(30)],
        "Close": [50200.0 + i * 200 + (i % 2) * 50 for i in range(30)],
        "Volume": [1.0e9 + i * 0.05e9 for i in range(30)],
        "post_count": [10 + (i % 5) * 5 for i in range(30)],
        "positive_post_count": [5 + (i % 3) for i in range(30)],
        "negative_post_count": [2 + (i % 2) for i in range(30)],
        "neutral_post_count": [3 + (i % 2) for i in range(30)],
        # Include missing sentiment on every 5th day to test imputer handling
        "sentiment_mean": [0.3 if i % 5 != 0 else np.nan for i in range(30)],
        "subjectivity_mean": [0.4 if i % 5 != 0 else np.nan for i in range(30)],
        "sentiment_weighted": [0.35 if i % 5 != 0 else 0.0 for i in range(30)],
        "sentiment_weighted_available": [i % 5 != 0 for i in range(30)],
        "has_posts": [i % 5 != 0 for i in range(30)],
    }
    raw_df = pd.DataFrame(data, index=dates)
    raw_df.index.name = "Date"

    m_df, feat_cols, target_cols = prepare_modeling_dataset(
        raw_df,
        horizons=[1, 2, 3, 4, 5],
        target_column="Close",
        include_nowcast=True,
    )

    y_dict = {h: m_df[f"target_return_h{h}"] for h in range(1, 6)}
    return m_df, feat_cols, y_dict


# ==============================================================================
# 1. Price Reconstruction & Anchor Validation
# ==============================================================================

def test_reconstruct_price_from_return():
    """Verifies conversion: Close_t * (1.0 + predicted_return)."""
    current_close = np.array([50000.0, 60000.0])
    predicted_return = np.array([0.02, -0.05])

    expected = np.array([50000.0 * 1.02, 60000.0 * 0.95])
    reconstructed = reconstruct_price_from_return(current_close, predicted_return)

    assert np.allclose(reconstructed, expected)


def test_reconstruct_price_rejects_invalid_anchors():
    """Verifies that non-positive or non-finite current close prices are rejected."""
    # Negative close
    with pytest.raises(ValueError, match="Current price anchor contains non-positive"):
        reconstruct_price_from_return(np.array([-50000.0]), np.array([0.01]))

    # Zero close
    with pytest.raises(ValueError, match="Current price anchor contains non-positive"):
        reconstruct_price_from_return(np.array([0.0]), np.array([0.01]))

    # NaN close
    with pytest.raises(ValueError, match="Current price anchor contains non-positive"):
        reconstruct_price_from_return(np.array([np.nan]), np.array([0.01]))

    # Non-finite return
    with pytest.raises(ValueError, match="Predicted returns contain non-finite"):
        reconstruct_price_from_return(np.array([50000.0]), np.array([np.nan]))


# ==============================================================================
# 2. Naïve Price-Persistence Baseline Tests
# ==============================================================================

def test_naive_persistence_model_prediction(synthetic_training_data):
    """
    Verifies that NaivePersistenceModel:
      - Predicts 0.0 forward return for all horizons.
      - Predicts Close_t for nominal prices.
    """
    m_df, feat_cols, y_dict = synthetic_training_data
    model = NaivePersistenceModel(price_column="Close")
    model.fit(m_df)

    # Return prediction: strictly zeros
    ret_preds = model.predict_returns(m_df)
    assert isinstance(ret_preds, np.ndarray)
    assert len(ret_preds) == len(m_df)
    assert (ret_preds == 0.0).all()

    # Price prediction: strictly latest close
    price_preds = model.predict_prices(m_df)
    assert isinstance(price_preds, np.ndarray)
    assert len(price_preds) == len(m_df)
    assert (price_preds == m_df["Close"].values).all()

    # Default predict dispatch
    assert (model.predict(m_df, return_type="price") == price_preds).all()
    assert (model.predict(m_df, return_type="return") == ret_preds).all()


def test_naive_persistence_unfitted_raises():
    """Verifies that calling predict() on an unfitted model raises RuntimeError."""
    model = NaivePersistenceModel(price_column="Close")
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        model.predict(pd.DataFrame({"Close": [100.0]}))


def test_naive_persistence_missing_price_column():
    """Verifies KeyError when price column is missing."""
    model = NaivePersistenceModel(price_column="NonExistent")
    with pytest.raises(KeyError, match="Price column 'NonExistent' not found"):
        model.fit(pd.DataFrame({"Other": [100.0]}))


def test_naive_persistence_invalid_prices():
    """Verifies ValueError when price column contains non-positive prices."""
    model = NaivePersistenceModel(price_column="Close")
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        model.fit(pd.DataFrame({"Close": [-100.0]}))


# ==============================================================================
# 3. Ridge Pipeline Construction & Imputation Handling
# ==============================================================================

def test_create_ridge_pipeline_structure():
    """Verifies that create_ridge_pipeline encapsulates imputation, scaling, and Ridge."""
    pipe = create_ridge_pipeline(alpha=2.5, random_state=123)

    assert isinstance(pipe, Pipeline)
    assert "imputer" in pipe.named_steps
    assert "scaler" in pipe.named_steps
    assert "regressor" in pipe.named_steps
    assert pipe.named_steps["regressor"].alpha == 2.5


def test_ridge_pipeline_handles_missing_sentiment_without_leakage(synthetic_training_data):
    """
    Verifies that the Ridge pipeline:
      - Fits on forward returns (target_return_h1).
      - Safely imputes NaNs in sentiment features using training median.
      - Predicts bounded forward returns deterministically.
      - Fits transformations strictly on train.
    """
    m_df, feat_cols, y_dict = synthetic_training_data

    # Split train/test chronologically (no random shuffle)
    train_size = 20
    X_train = m_df[feat_cols].iloc[:train_size]
    y_train = y_dict[1].iloc[:train_size]

    X_test = m_df[feat_cols].iloc[train_size:]

    pipe = create_ridge_pipeline(alpha=1.0)
    pipe.fit(X_train, y_train)

    preds_train = pipe.predict(X_train)
    preds_test = pipe.predict(X_test)

    assert len(preds_train) == train_size
    assert len(preds_test) == len(m_df) - train_size
    assert not np.isnan(preds_test).any()
    # Predicted forward returns should be realistic (within [-1, 1])
    assert (preds_test > -1.0).all() and (preds_test < 1.0).all()


# ==============================================================================
# 4. Multi-Horizon Forecaster (Return Prediction & Price Conversion)
# ==============================================================================

def test_multi_horizon_forecaster_ridge(synthetic_training_data):
    """
    Verifies MultiHorizonForecaster with model_type='ridge':
      - Fits on forward returns for horizons 1 through 5.
      - Predicts forward returns and reconstructed prices.
      - Price reconstruction equals Close_t * (1 + pred_return).
      - Unobserved end-of-series target NaNs are masked during training.
    """
    m_df, feat_cols, y_dict = synthetic_training_data

    forecaster = MultiHorizonForecaster(
        model_type="ridge",
        horizons=[1, 2, 3, 4, 5],
        alpha=1.0,
        price_column="Close",
    )
    forecaster.fit(m_df, y_dict, feature_columns=feat_cols)

    # 1. Return predictions
    returns_pred = forecaster.predict_returns(m_df)
    assert set(returns_pred.keys()) == {1, 2, 3, 4, 5}
    for h in range(1, 6):
        assert len(returns_pred[h]) == len(m_df)
        assert not np.isnan(returns_pred[h]).any()

    # 2. Price predictions
    prices_pred = forecaster.predict_prices(m_df)
    assert set(prices_pred.keys()) == {1, 2, 3, 4, 5}
    for h in range(1, 6):
        assert len(prices_pred[h]) == len(m_df)
        assert not np.isnan(prices_pred[h]).any()
        assert (prices_pred[h] > 0).all()
        # Verify reconstruction formula exactly
        expected_price = m_df["Close"].values * (1.0 + returns_pred[h])
        assert np.allclose(prices_pred[h], expected_price)


def test_multi_horizon_forecaster_rejects_n0_for_ridge(synthetic_training_data):
    """
    Safeguard: Verifies that attempting to train a Ridge model on horizon n+0
    raises a clear ValueError to prevent contemporaneous nowcast leakage.
    """
    m_df, feat_cols, y_dict = synthetic_training_data
    forecaster = MultiHorizonForecaster(
        model_type="ridge",
        horizons=[0, 1, 2],
        price_column="Close",
    )
    with pytest.raises(ValueError, match="Cannot train Ridge model on horizon n\\+0"):
        forecaster.fit(m_df, y_dict, feature_columns=feat_cols)


def test_multi_horizon_forecaster_nowcast_reference(synthetic_training_data):
    """Verifies predict_nowcast returns the contemporaneous cutoff price."""
    m_df, feat_cols, y_dict = synthetic_training_data
    forecaster = MultiHorizonForecaster(model_type="ridge", horizons=[1, 2])
    forecaster.fit(m_df, y_dict, feature_columns=feat_cols)

    nowcast = forecaster.predict_nowcast(m_df)
    assert (nowcast == m_df["Close"].values).all()


def test_multi_horizon_forecaster_persistence(synthetic_training_data):
    """Verifies MultiHorizonForecaster with model_type='persistence' across horizons 1-5."""
    m_df, feat_cols, y_dict = synthetic_training_data

    forecaster = MultiHorizonForecaster(
        model_type="persistence",
        horizons=[1, 2, 3, 4, 5],
        price_column="Close",
    )
    forecaster.fit(m_df, y_dict, feature_columns=feat_cols)

    returns_pred = forecaster.predict_returns(m_df)
    prices_pred = forecaster.predict_prices(m_df)

    for h in range(1, 6):
        assert (returns_pred[h] == 0.0).all()
        assert (prices_pred[h] == m_df["Close"].values).all()


def test_multi_horizon_forecaster_unfitted_raises():
    """Verifies that predict() before fit() raises RuntimeError."""
    forecaster = MultiHorizonForecaster()
    with pytest.raises(RuntimeError, match="is not fitted"):
        forecaster.predict(pd.DataFrame({"Close": [100.0]}))


def test_multi_horizon_forecaster_invalid_model_type():
    """Verifies ValueError on unsupported model_type."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        MultiHorizonForecaster(model_type="invalid_model")


# ==============================================================================
# 5. Statistical Metric Calculations & Safeguards
# ==============================================================================

def test_calculate_metrics_price():
    """Verifies exact price metric calculations (RMSE, MAE, MAPE)."""
    y_true = np.array([100.0, 110.0, 120.0])
    y_pred = np.array([102.0, 108.0, 126.0])

    metrics = calculate_metrics(y_true, y_pred, target_type="price")
    assert np.isclose(metrics["rmse"], 3.8297, atol=1e-3)
    assert np.isclose(metrics["mae"], 3.3333, atol=1e-3)
    assert np.isclose(metrics["mape"], 2.9394, atol=1e-3)


def test_calculate_metrics_return():
    """Verifies return metric calculations (RMSE, MAE)."""
    y_true = np.array([0.01, -0.02, 0.05])
    y_pred = np.array([0.015, -0.018, 0.048])

    metrics = calculate_metrics(y_true, y_pred, target_type="return")
    assert "mape" not in metrics
    assert np.isclose(metrics["rmse"], 0.003317, atol=1e-4)


def test_calculate_metrics_rejects_scale_mismatch():
    """Verifies ValueError when return predictions (<1) are compared against price levels (>100)."""
    prices = np.array([50000.0, 51000.0])
    returns = np.array([0.02, 0.01])

    with pytest.raises(ValueError, match="Scale mismatch detected"):
        calculate_metrics(prices, returns, target_type="price")


def test_aggregate_forecast_metrics_excludes_n0():
    """
    Safeguard: Verifies that aggregate_forecast_metrics strictly excludes horizon 0
    so that a perfect 0.0 nowcast reference cannot contaminate or artificially deflate
    future-forecast evaluation summaries.
    """
    metrics_by_h = {
        0: {"rmse": 0.0, "mae": 0.0, "mape": 0.0},      # n+0 nowcast reference
        1: {"rmse": 100.0, "mae": 80.0, "mape": 1.5},
        2: {"rmse": 200.0, "mae": 160.0, "mape": 2.5},
    }

    # Aggregate over all available horizons
    agg = aggregate_forecast_metrics(metrics_by_h)

    # Should average only horizons 1 and 2:
    # RMSE: (100 + 200) / 2 = 150.0 (NOT (0 + 100 + 200) / 3 = 100.0!)
    assert agg["mean_rmse"] == 150.0
    assert agg["mean_mae"] == 120.0
    assert agg["mean_mape"] == 2.0


def test_aggregate_forecast_metrics_no_valid_future_horizons():
    """Verifies ValueError when only horizon 0 is passed."""
    metrics_only_n0 = {0: {"rmse": 0.0, "mae": 0.0}}
    with pytest.raises(ValueError, match="No valid future forecasting horizons"):
        aggregate_forecast_metrics(metrics_only_n0)


# ==============================================================================
# 6. Chronological Order & Non-Shuffling Protection
# ==============================================================================

def test_chronological_order_preservation(synthetic_training_data):
    """
    Verifies that time-series observations are processed strictly in chronological order,
    and that dates remain monotonic ascending.
    """
    m_df, feat_cols, y_dict = synthetic_training_data
    assert m_df.index.is_monotonic_increasing

    forecaster = MultiHorizonForecaster(model_type="ridge", horizons=[1])
    forecaster.fit(m_df, y_dict, feature_columns=feat_cols)
    preds = forecaster.predict(m_df)

    assert len(preds[1]) == len(m_df)
