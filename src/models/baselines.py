"""
Baseline forecasting models for multi-horizon price and return prediction.

This module provides baseline forecasting models across future horizons n+1 through n+5:
1. Naïve Price-Persistence Baseline:
   Predicts a forward return of 0.0, which converts to the latest known close price
   at prediction cutoff: Close_t * (1.0 + 0.0) = Close_t.
2. Ridge Regression Baseline:
   Predicts forward returns (target_return_h{h} = Close_{t+h} / Close_t - 1.0) using
   strictly stationary price-derived and sentiment features. Preprocessing (median
   imputation and standard scaling) is encapsulated within a scikit-learn Pipeline.
   Predictions are reconstructed to price levels via Close_t * (1.0 + predicted_return).
3. Multi-Horizon Orchestrator:
   Coordinates multi-horizon forecasting across future horizons (n+1 through n+5),
   strictly excluding contemporaneous nowcast reference n+0 from training and
   aggregate forecasting evaluations.

Temporal Leakage Prevention:
    - Prediction Cutoff: Date D at 23:59:59 UTC.
    - Horizon n+0: Contemporaneous nowcast reference price (Close_D), not a learned forecast.
    - Horizons n+1 to n+5: Future targets based on Close_{D+h} / Close_D - 1.0.
    - Feature Matrix X: Strictly stationary; raw Close, Open, High, Low and targets are excluded.
    - Reconstruction Anchor: Prediction-time Close_D used strictly to rescale returns to prices.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.processing.features import (
    DEFAULT_HORIZONS,
    FORECAST_HORIZONS,
    NOWCAST_HORIZON,
    get_feature_columns,
)

logger = logging.getLogger(__name__)


def reconstruct_price_from_return(
    current_close: Union[pd.Series, np.ndarray, float],
    predicted_return: Union[pd.Series, np.ndarray, float],
) -> np.ndarray:
    """
    Converts predicted forward return back to nominal price:
        predicted_price = current_close * (1.0 + predicted_return)

    Validation:
        - current_close must be strictly positive (> 0) and finite.
        - predicted_return must be finite.

    Args:
        current_close: Known closing price at prediction cutoff.
        predicted_return: Model predicted forward return.

    Returns:
        1D numpy array of reconstructed price forecasts.

    Raises:
        ValueError: If current_close is non-positive, non-finite, or predicted_return is non-finite.
    """
    c_arr = np.asarray(current_close, dtype=float)
    r_arr = np.asarray(predicted_return, dtype=float)

    if not np.isfinite(c_arr).all() or (c_arr <= 0).any():
        raise ValueError("Current price anchor contains non-positive or non-finite values.")
    if not np.isfinite(r_arr).all():
        raise ValueError("Predicted returns contain non-finite values.")

    return c_arr * (1.0 + r_arr)


class NaivePersistenceModel(BaseEstimator, RegressorMixin):
    """
    Naïve Price-Persistence Baseline Forecaster.

    Convention:
        Forecasts zero forward return (hat{r}_{t, h} = 0.0) for every horizon.
        When converted to price:
            hat{y}_{t, h} = Close_t * (1.0 + 0.0) = Close_t
    """

    def __init__(self, price_column: str = "Close"):
        self.price_column = price_column
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "NaivePersistenceModel":
        """
        Fits the persistence model by validating the presence of the price column.

        Args:
            X: Feature DataFrame containing the price column.
            y: Ignored (persists latest price regardless of target).

        Returns:
            Fitted model instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")
        if self.price_column not in X.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in X columns: {list(X.columns)}")
        if X.empty:
            raise ValueError("Cannot fit NaivePersistenceModel on empty DataFrame.")

        prices = X[self.price_column].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"Price column '{self.price_column}' contains non-positive or non-finite values.")

        self.is_fitted_ = True
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        """Forecasts zero forward return for persistence baseline."""
        self._check_is_fitted(X)
        return np.zeros(len(X), dtype=float)

    def predict_prices(self, X: pd.DataFrame) -> np.ndarray:
        """Forecasts latest close price (Close_t * (1.0 + 0.0))."""
        self._check_is_fitted(X)
        prices = X[self.price_column].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"Price column '{self.price_column}' contains non-positive or non-finite values.")
        return prices

    def predict(self, X: pd.DataFrame, return_type: str = "price") -> np.ndarray:
        """
        Generates predictions as reconstructed prices (default) or forward returns.

        Args:
            X: Input DataFrame containing price_column.
            return_type: 'price' for price level, 'return' for forward return.

        Returns:
            1D numpy array of predictions.
        """
        if return_type == "price":
            return self.predict_prices(X)
        elif return_type in {"return", "returns"}:
            return self.predict_returns(X)
        else:
            raise ValueError(f"Invalid return_type: '{return_type}'. Supported: 'price', 'return'.")

    def _check_is_fitted(self, X: pd.DataFrame) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")
        if self.price_column not in X.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in X columns: {list(X.columns)}")


def create_ridge_pipeline(
    alpha: float = 1.0,
    impute_strategy: str = "median",
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs a leakage-free scikit-learn Pipeline for Ridge regression.

    Pipeline stages:
        1. SimpleImputer: Imputes missing sentiment features (e.g. no-post days)
           strictly using training statistics.
        2. StandardScaler: Centers and scales features using training statistics.
        3. Ridge: L2-regularized linear regressor.

    Args:
        alpha: L2 regularization strength (default: 1.0).
        impute_strategy: Strategy for missing value imputation (default: 'median').
        random_state: Random state seed for reproducibility.

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha, random_state=random_state)),
    ])
    return pipeline


class MultiHorizonForecaster:
    """
    Coordinates multi-horizon forecasting across future horizons (default: n+1 through n+5).

    Methodology:
        - Targets: Forward returns (target_return_h{h} = Close.shift(-h) / Close - 1.0).
        - Feature Matrix X: Stationary price-derived features and sentiment features (raw Close is excluded).
        - Price Reconstruction: hat{y}_{t, h} = Close_t * (1.0 + hat{r}_{t, h}).
        - Horizon n+0: Strictly quarantined from training and evaluation summaries.
          It is a contemporaneous nowcast reference, not a learned forecast.
    """

    def __init__(
        self,
        model_type: str = "ridge",  # 'ridge' or 'persistence'
        horizons: Optional[List[int]] = None,
        alpha: float = 1.0,
        price_column: str = "Close",
        random_state: int = 42,
    ):
        if model_type not in {"ridge", "persistence"}:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Supported: 'ridge', 'persistence'.")

        self.model_type = model_type
        # Default to future forecasting horizons only: [1, 2, 3, 4, 5]
        self.horizons = list(horizons) if horizons is not None else list(FORECAST_HORIZONS)
        self.alpha = alpha
        self.price_column = price_column
        self.random_state = random_state

        self.models_: Dict[int, Union[NaivePersistenceModel, Pipeline]] = {}
        self.feature_columns_: Optional[List[str]] = None
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y_dict: Dict[int, Union[pd.Series, np.ndarray]],
        feature_columns: Optional[List[str]] = None,
    ) -> "MultiHorizonForecaster":
        """
        Fits models for each configured horizon.

        Args:
            X: Input feature DataFrame (must contain price_column for validation and anchoring).
            y_dict: Dictionary mapping horizon int h -> forward return target Series/array.
            feature_columns: Optional explicit list of feature column names.
                             If None, automatically inferred and validated via get_feature_columns(X).

        Returns:
            Fitted forecaster instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame for X, got {type(X).__name__}")
        if X.empty:
            raise ValueError("Cannot fit MultiHorizonForecaster on empty DataFrame X.")
        if self.price_column not in X.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in X columns.")

        # Guard against n+0 model training for Ridge
        if 0 in self.horizons and self.model_type == "ridge":
            raise ValueError(
                "Cannot train Ridge model on horizon n+0. Horizon n+0 is a contemporaneous nowcast "
                "reference, not a future forecast horizon. Valid future forecasting horizons are 1 through 5."
            )

        # Validate feature columns (strictly excludes raw Close and targets)
        if feature_columns is None:
            self.feature_columns_ = get_feature_columns(X)
        else:
            prohibited = {"Close", "Open", "High", "Low", "Date"}
            self.feature_columns_ = [
                c for c in feature_columns
                if c in X.columns and c not in prohibited and not c.startswith("target_") and not c.startswith("forward_return")
            ]

        if not self.feature_columns_:
            raise ValueError("No valid numeric feature columns found in X.")

        X_feats = X[self.feature_columns_]

        self.models_ = {}
        for h in self.horizons:
            if h not in y_dict:
                raise KeyError(f"Target for horizon n+{h} missing from y_dict.")

            y_h = pd.Series(y_dict[h])

            # Mask out unobserved future rows (NaNs at end of series) and purged boundary targets
            valid_mask = y_h.notna()
            if valid_mask.sum() < 2:
                raise ValueError(
                    f"Insufficient training observations for horizon n+{h}: only {valid_mask.sum()} eligible "
                    f"samples remain after purging boundary target leakage and masking unobserved values."
                )

            X_valid = X_feats.loc[valid_mask]
            y_valid = y_h.loc[valid_mask]

            if self.model_type == "persistence" or h == 0:
                model = NaivePersistenceModel(price_column=self.price_column)
                model.fit(X.loc[valid_mask], y_valid)
            else:
                model = create_ridge_pipeline(
                    alpha=self.alpha,
                    random_state=self.random_state,
                )
                model.fit(X_valid, y_valid)

            self.models_[h] = model

        self.is_fitted_ = True
        return self

    def predict_returns(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Generates predicted forward returns for all configured horizons.

        Returns:
            Dictionary mapping horizon h -> 1D numpy array of predicted returns.
        """
        self._check_is_fitted(X)
        X_feats = X[self.feature_columns_]

        preds_returns: Dict[int, np.ndarray] = {}
        for h in self.horizons:
            model = self.models_[h]
            if isinstance(model, NaivePersistenceModel):
                preds_returns[h] = model.predict_returns(X)
            else:
                preds_returns[h] = model.predict(X_feats)

        return preds_returns

    def predict_prices(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Reconstructs predicted future prices from forward returns and current close:
            predicted_price_{t, h} = Close_t * (1.0 + predicted_return_{t, h})

        Returns:
            Dictionary mapping horizon h -> 1D numpy array of reconstructed prices.
        """
        self._check_is_fitted(X)
        current_close = X[self.price_column].to_numpy(dtype=float)
        if not np.isfinite(current_close).all() or (current_close <= 0).any():
            raise ValueError(f"Current price anchor in '{self.price_column}' contains non-positive or non-finite values.")

        returns_dict = self.predict_returns(X)
        preds_prices: Dict[int, np.ndarray] = {}

        for h, ret_arr in returns_dict.items():
            preds_prices[h] = reconstruct_price_from_return(current_close, ret_arr)

        return preds_prices

    def predict(self, X: pd.DataFrame, return_type: str = "price") -> Dict[int, np.ndarray]:
        """
        Generates predictions as reconstructed prices (default) or forward returns.

        Args:
            X: Input feature DataFrame (must contain price_column).
            return_type: 'price' to return reconstructed prices (default),
                         'return' to return predicted forward returns.

        Returns:
            Dictionary mapping horizon h -> 1D numpy array of predictions.
        """
        if return_type == "price":
            return self.predict_prices(X)
        elif return_type in {"return", "returns"}:
            return self.predict_returns(X)
        else:
            raise ValueError(f"Invalid return_type: '{return_type}'. Supported: 'price', 'return'.")

    def predict_nowcast(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns contemporaneous nowcast reference price Close_t at prediction cutoff.
        This represents the reference price at cutoff D, NOT a future prediction.
        """
        self._check_is_fitted(X)
        prices = X[self.price_column].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"Price column '{self.price_column}' contains non-positive or non-finite values.")
        return prices

    def _check_is_fitted(self, X: pd.DataFrame) -> None:
        if not self.is_fitted_:
            raise RuntimeError("MultiHorizonForecaster is not fitted. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame for X, got {type(X).__name__}")
        if self.price_column not in X.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in X columns.")


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_type: str = "price",
) -> Dict[str, float]:
    """
    Computes statistical forecast accuracy metrics.

    Args:
        y_true: Ground truth target array.
        y_pred: Model predictions array.
        target_type: 'price' for nominal prices, 'return' for forward returns.

    Returns:
        Dictionary of computed metrics:
          - For 'price': rmse, mae, mape
          - For 'return': rmse, mae
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(yt) | np.isnan(yp))
    if not valid.any():
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan} if target_type == "price" else {"rmse": np.nan, "mae": np.nan}

    yt_v = yt[valid]
    yp_v = yp[valid]

    # Guard against scale mismatch (e.g. comparing returns to nominal prices)
    if target_type == "price":
        mean_true = float(np.mean(np.abs(yt_v)))
        mean_pred = float(np.mean(np.abs(yp_v)))
        if mean_true > 100.0 and mean_pred < 1.0:
            raise ValueError(
                "Scale mismatch detected: y_true appears to be price levels (>100) while y_pred appears to be returns (<1). "
                "Reconstruct forecast prices before computing price metrics, or specify target_type='return'."
            )

    diff = yp_v - yt_v
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    if target_type == "return":
        return {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
        }

    # For price: compute MAPE
    denom = np.where(np.abs(yt_v) < 1e-8, np.nan, np.abs(yt_v))
    mape = float(np.nanmean(np.abs(diff) / denom) * 100.0)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
    }


def aggregate_forecast_metrics(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Computes summary forecast metrics across forecasting horizons (1 to 5).

    Safety Safeguard:
        Horizon n+0 is strictly excluded from aggregate forecast metrics.
        If horizon 0 is present in metrics_by_horizon, it is filtered out so that
        the perfect 0.0 error of the nowcast reference cannot contaminate or artificially
        deflate future-forecast evaluation summaries.

    Args:
        metrics_by_horizon: Dictionary mapping horizon h -> metrics dict.
        horizons: Optional explicit list of horizons to aggregate (default: all h >= 1).

    Returns:
        Dictionary of mean metrics across future horizons.
    """
    if horizons is None:
        target_horizons = [h for h in metrics_by_horizon.keys() if h >= 1]
    else:
        # Strictly exclude 0
        target_horizons = [h for h in horizons if h >= 1 and h in metrics_by_horizon]

    if not target_horizons:
        raise ValueError("No valid future forecasting horizons (h >= 1) found to aggregate.")

    metric_keys = list(next(iter(metrics_by_horizon.values())).keys())
    agg: Dict[str, float] = {}

    for k in metric_keys:
        vals = [
            metrics_by_horizon[h][k]
            for h in target_horizons
            if k in metrics_by_horizon[h] and not np.isnan(metrics_by_horizon[h][k])
        ]
        if vals:
            agg[f"mean_{k}"] = round(float(np.mean(vals)), 4)
        else:
            agg[f"mean_{k}"] = np.nan

    return agg
