"""
Walk-forward cross-validation module for multi-horizon forecasting.

Implements deterministic chronological rolling/expanding walk-forward splits
strictly without look-ahead bias or random shuffling.

Core Guarantees:
1. Strict Temporal Order:
   Every training fold strictly precedes its test fold (max(train_date) < min(test_date)).
2. Pipeline Fitting Isolation:
   Preprocessing (imputation and scaling) and model parameters are fit strictly on each fold's
   training observations; future and test observations are transformed using training statistics.
3. Multi-Horizon Coverage (n+1 through n+5):
   Evaluates forward return forecasts and reconstructed price forecasts for horizons 1 through 5.
   Horizon n+0 is strictly excluded from forecast evaluation.
4. Missing Target Quarantine:
   End-of-series unobserved targets (NaN due to forward shifts) are cleanly masked per horizon
   without shifting predictions away from their true calendar timestamps.
5. Exact Date Indexing:
   All predictions and evaluation metrics preserve date indexing and timestamp integrity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_directional_accuracy,
    compute_forecast_metrics,
    format_metrics_comparison_table,
)
from src.models.baselines import (
    MultiHorizonForecaster,
    NaivePersistenceModel,
    create_ridge_pipeline,
    reconstruct_price_from_return,
)
from src.processing.features import (
    FORECAST_HORIZONS,
    add_forecast_targets,
    add_price_derived_features,
    add_strategy_trade_target,
    get_feature_columns,
    validate_calendar_continuity,
)

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """
    Generates deterministic chronological train/test fold splits.

    Split Design:
        - train_window_days: Number of calendar days in each training partition.
        - test_window_days: Number of calendar days in each test partition.
        - step_days: Step size between successive folds. Default equals test_window_days
          for non-overlapping out-of-sample test evaluation.
        - expanding: If True, training window expands from index 0; if False (default),
          rolling window with fixed train_window_days (e.g. 90-day rolling window).
    """

    def __init__(
        self,
        train_window_days: int = 90,
        test_window_days: int = 15,
        step_days: Optional[int] = None,
        expanding: bool = False,
    ):
        if train_window_days <= 0:
            raise ValueError(f"train_window_days must be positive, got {train_window_days}")
        if test_window_days <= 0:
            raise ValueError(f"test_window_days must be positive, got {test_window_days}")

        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days if step_days is not None else test_window_days
        if self.step_days <= 0:
            raise ValueError(f"step_days must be positive, got {self.step_days}")

        self.expanding = expanding

    def get_n_splits(self, n_samples: int) -> int:
        """Calculates total number of valid chronological folds possible."""
        min_required = self.train_window_days + self.test_window_days
        if n_samples < min_required:
            return 0
        return (n_samples - min_required) // self.step_days + 1

    def split(
        self,
        df: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
        """
        Yields (train_indices, test_indices, train_dates, test_dates) for each fold.

        Raises:
            ValueError: If total samples are insufficient for even one fold.
        """
        n_samples = len(df)
        min_required = self.train_window_days + self.test_window_days

        if n_samples < min_required:
            raise ValueError(
                f"Insufficient history: dataset has {n_samples} observations, "
                f"but walk-forward requires at least {min_required} "
                f"(train_window_days={self.train_window_days} + test_window_days={self.test_window_days})."
            )

        n_splits = self.get_n_splits(n_samples)
        dates = pd.DatetimeIndex(df.index)

        for fold in range(n_splits):
            start_offset = fold * self.step_days
            train_start = 0 if self.expanding else start_offset
            train_end = start_offset + self.train_window_days
            test_start = train_end
            test_end = test_start + self.test_window_days

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx, dates[train_idx], dates[test_idx]


@dataclass
class WalkForwardResult:
    """Holds complete results, out-of-sample predictions, and metrics from walk-forward validation."""
    predictions_prices: Dict[str, pd.DataFrame]  # model_name -> DataFrame of predictions and actuals
    predictions_returns: Dict[str, pd.DataFrame]
    summary_price_metrics: Dict[str, Dict[int, Dict[str, Any]]]
    summary_return_metrics: Dict[str, Dict[int, Dict[str, Any]]]
    fold_metrics: List[Dict[str, Any]]
    comparison_table_rmse: pd.DataFrame
    comparison_table_mae: pd.DataFrame
    comparison_table_da: pd.DataFrame
    n_folds: int
    train_window_days: int
    test_window_days: int
    step_days: int
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_metrics: Optional[Dict[str, Any]] = None


class WalkForwardEvaluator:
    """
    Executes chronological walk-forward evaluation across multiple forecasting horizons.

    Workflow:
        1. Validates calendar continuity of input price/feature DataFrame.
        2. Generates forward return targets (target_return_h{h}) and price targets (target_Close_n{h}).
        3. Iterates through chronological folds using WalkForwardSplitter.
        4. Trains Ridge (forward returns) and Naive Persistence models strictly on fold train data.
        5. Computes out-of-sample return forecasts and reconstructed price forecasts on fold test data.
        6. Aggregates out-of-sample forecasts without double-counting when step_days == test_window_days.
        7. Computes MAE, RMSE, MAPE, and Directional Accuracy per horizon (1-5) and model.
    """

    def __init__(
        self,
        splitter: Optional[WalkForwardSplitter] = None,
        horizons: Optional[List[int]] = None,
        price_column: str = "Close",
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.splitter = splitter if splitter is not None else WalkForwardSplitter(
            train_window_days=90,
            test_window_days=15,
            step_days=15,  # Non-overlapping test blocks
        )
        self.horizons = list(horizons) if horizons is not None else list(FORECAST_HORIZONS)
        # Ensure n+0 is never included in forecasting evaluation
        if 0 in self.horizons:
            raise ValueError(
                "Horizon n+0 cannot be included in WalkForwardEvaluator horizons. "
                "Forecast horizons must be future horizons (h >= 1)."
            )

        self.price_column = price_column
        self.alpha = alpha
        self.random_state = random_state

    def evaluate(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        is_synthetic: bool = False,
    ) -> WalkForwardResult:
        """
        Runs walk-forward evaluation on the given dataset.

        Args:
            df: DataFrame containing price and feature columns with DatetimeIndex or Date column.
            feature_columns: Optional explicit list of features to pass to models.
            is_synthetic: Whether the dataset is synthetic (must be flagged accurately).

        Returns:
            WalkForwardResult containing complete predictions, metrics, and comparisons.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Ensure index is DatetimeIndex and validate calendar continuity
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.copy()
                df.index = pd.to_datetime(df["Date"], utc=True).dt.normalize()
            else:
                df = df.copy()
                df.index = pd.to_datetime(df.index, utc=True).dt.normalize()

        validate_calendar_continuity(df)

        # Prepare targets (forward returns and price targets)
        prepared_df = add_forecast_targets(
            df,
            horizons=self.horizons,
            target_column=self.price_column,
            include_nowcast=False,
        )
        if "Open" in df.columns and len(df) >= 3:
            prepared_df = add_strategy_trade_target(prepared_df, open_column="Open")

        # Identify feature columns (strictly stationary, no raw prices or targets)
        if feature_columns is None:
            feat_cols = get_feature_columns(prepared_df)
        else:
            prohibited = {"Close", "Open", "High", "Low", "Date"}
            feat_cols = [
                c for c in feature_columns
                if c in prepared_df.columns and c not in prohibited and not c.startswith("target_") and not c.startswith("forward_return") and not c.startswith("trade_return")
            ]

        if not feat_cols:
            raise ValueError("No valid numeric feature columns found for model input.")

        # Out-of-sample prediction containers
        # Records: Date -> {model: {horizon: pred}}
        test_predictions_records: List[Dict[str, Any]] = []
        fold_metrics_list: List[Dict[str, Any]] = []

        fold_idx = 0
        for train_idx, test_idx, train_dates, test_dates in self.splitter.split(prepared_df):
            fold_idx += 1
            train_df = prepared_df.iloc[train_idx]
            test_df = prepared_df.iloc[test_idx]

            # Construct target dictionary for training (forward returns) with horizon-specific purging:
            # Observation t is eligible for horizon h if and only if its forward target timestamp
            # index (train_idx + h) < test_start_idx (strictly precedes test fold start date).
            test_start_idx = test_idx[0]
            y_train_dict = {}
            fold_eligible_counts: Dict[str, Any] = {"fold": fold_idx}
            for h in self.horizons:
                y_h = train_df[f"target_return_h{h}"].copy()
                # Purge training observations whose forward target date falls into or beyond test fold
                purge_mask = (train_idx + h) >= test_start_idx
                y_h.iloc[purge_mask] = np.nan
                y_train_dict[h] = y_h
                fold_eligible_counts[f"h{h}_eligible_train_samples"] = int(y_h.notna().sum())

            # Strategy Trade Target Purging:
            # Signal at index t uses Open[t+1] and Open[t+2]; label endpoint is t + 2.
            # To ensure the label endpoint is strictly before the test fold's first row (test_start_idx),
            # observation t is eligible if and only if (train_idx + 2) < test_start_idx.
            has_strategy_target = "target_trade_return" in prepared_df.columns
            if has_strategy_target:
                y_trade = train_df["target_trade_return"].copy()
                trade_purge_mask = (train_idx + 2) >= test_start_idx
                y_trade.iloc[trade_purge_mask] = np.nan
                fold_eligible_counts["trade_eligible_train_samples"] = int(y_trade.notna().sum())

                # Fit dedicated Strategy Ridge Model strictly on eligible non-leaking samples
                valid_trade_mask = y_trade.notna()
                if valid_trade_mask.sum() > 0:
                    strategy_model = create_ridge_pipeline(
                        alpha=self.alpha,
                        random_state=self.random_state,
                    )
                    strategy_model.fit(train_df.loc[valid_trade_mask, feat_cols], y_trade.loc[valid_trade_mask])
                    strategy_pred_ret = strategy_model.predict(test_df[feat_cols])
                else:
                    strategy_pred_ret = np.full(len(test_df), np.nan)
            else:
                strategy_pred_ret = np.full(len(test_df), np.nan)

            # 1. Fit Ridge Forecaster strictly on training fold (using only eligible non-leaking samples)
            ridge_forecaster = MultiHorizonForecaster(
                model_type="ridge",
                horizons=self.horizons,
                alpha=self.alpha,
                price_column=self.price_column,
                random_state=self.random_state,
            )
            ridge_forecaster.fit(train_df, y_train_dict, feature_columns=feat_cols)

            # 2. Fit Persistence Forecaster strictly on training fold
            pers_forecaster = MultiHorizonForecaster(
                model_type="persistence",
                horizons=self.horizons,
                price_column=self.price_column,
            )
            pers_forecaster.fit(train_df, y_train_dict, feature_columns=feat_cols)

            # 3. Generate out-of-sample predictions on test fold
            ridge_ret_preds = ridge_forecaster.predict_returns(test_df)
            ridge_price_preds = ridge_forecaster.predict_prices(test_df)

            pers_ret_preds = pers_forecaster.predict_returns(test_df)
            pers_price_preds = pers_forecaster.predict_prices(test_df)

            # Store predictions with date alignment
            for i, test_date in enumerate(test_dates):
                row_dict: Dict[str, Any] = {
                    "Date": test_date,
                    "fold": fold_idx,
                    "Close": test_df[self.price_column].iloc[i],
                }
                for h in self.horizons:
                    # Ground truths
                    row_dict[f"actual_return_h{h}"] = test_df[f"target_return_h{h}"].iloc[i]
                    row_dict[f"actual_price_h{h}"] = test_df[f"target_{self.price_column}_n{h}"].iloc[i]
                    # Ridge predictions
                    row_dict[f"ridge_pred_return_h{h}"] = ridge_ret_preds[h][i]
                    row_dict[f"ridge_pred_price_h{h}"] = ridge_price_preds[h][i]
                    # Persistence predictions
                    row_dict[f"persistence_pred_return_h{h}"] = pers_ret_preds[h][i]
                    row_dict[f"persistence_pred_price_h{h}"] = pers_price_preds[h][i]

                if has_strategy_target:
                    row_dict["actual_trade_return"] = test_df["target_trade_return"].iloc[i]
                    row_dict["strategy_pred_return"] = strategy_pred_ret[i]
                    if "Open" in test_df.columns:
                        row_dict["Open"] = test_df["Open"].iloc[i]

                test_predictions_records.append(row_dict)

            # Compute fold-level metrics
            fold_metric: Dict[str, Any] = {
                "fold": fold_idx,
                "train_end": str(train_dates[-1].date()),
                "test_end": str(test_dates[-1].date()),
            }
            fold_metric.update(fold_eligible_counts)
            for h in self.horizons:
                actual_p = test_df[f"target_{self.price_column}_n{h}"]
                actual_r = test_df[f"target_return_h{h}"]
                c_close = test_df[self.price_column]

                r_p_metrics = compute_forecast_metrics(actual_p, ridge_price_preds[h], target_type="price", current_close=c_close)
                p_p_metrics = compute_forecast_metrics(actual_p, pers_price_preds[h], target_type="price", current_close=c_close)

                fold_metric[f"h{h}_ridge_rmse"] = r_p_metrics["rmse"]
                fold_metric[f"h{h}_pers_rmse"] = p_p_metrics["rmse"]
                fold_metric[f"h{h}_ridge_da"] = r_p_metrics["directional_accuracy"]
                fold_metric[f"h{h}_pers_da"] = p_p_metrics["directional_accuracy"]

            if has_strategy_target and not np.isnan(strategy_pred_ret).all():
                actual_trade_fold = test_df["target_trade_return"].values
                fold_trade_metrics = compute_forecast_metrics(actual_trade_fold, strategy_pred_ret, target_type="return")
                fold_metric["trade_ridge_rmse"] = fold_trade_metrics["rmse"]
                fold_metric["trade_ridge_mae"] = fold_trade_metrics["mae"]
                fold_metric["trade_ridge_da"] = fold_trade_metrics["directional_accuracy"]

            fold_metrics_list.append(fold_metric)

        if not test_predictions_records:
            raise ValueError(f"No walk-forward folds could be generated with {len(prepared_df)} rows.")

        # Convert predictions to DataFrame indexed by Date
        pred_df = pd.DataFrame(test_predictions_records).set_index("Date")

        # Organize predictions_prices and predictions_returns
        price_cols_ridge = [f"actual_price_h{h}" for h in self.horizons] + [f"ridge_pred_price_h{h}" for h in self.horizons] + ["Close"]
        price_cols_pers = [f"actual_price_h{h}" for h in self.horizons] + [f"persistence_pred_price_h{h}" for h in self.horizons] + ["Close"]

        return_cols_ridge = [f"actual_return_h{h}" for h in self.horizons] + [f"ridge_pred_return_h{h}" for h in self.horizons] + ["Close"]
        return_cols_pers = [f"actual_return_h{h}" for h in self.horizons] + [f"persistence_pred_return_h{h}" for h in self.horizons] + ["Close"]

        predictions_prices = {
            "ridge": pred_df[price_cols_ridge].copy(),
            "persistence": pred_df[price_cols_pers].copy(),
        }
        predictions_returns = {
            "ridge": pred_df[return_cols_ridge].copy(),
            "persistence": pred_df[return_cols_pers].copy(),
        }

        if "actual_trade_return" in pred_df.columns and "strategy_pred_return" in pred_df.columns:
            trade_cols = ["actual_trade_return", "strategy_pred_return", "Close"]
            if "Open" in pred_df.columns:
                trade_cols.append("Open")
            predictions_returns["strategy_trade"] = pred_df[trade_cols].copy()
            summary_strategy_metrics = compute_forecast_metrics(
                pred_df["actual_trade_return"].values,
                pred_df["strategy_pred_return"].values,
                target_type="return",
            )
        else:
            summary_strategy_metrics = None

        # Compute full out-of-sample summary metrics across all folds
        summary_price_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {"ridge": {}, "persistence": {}}
        summary_return_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {"ridge": {}, "persistence": {}}

        c_close_all = pred_df["Close"].values

        for h in self.horizons:
            act_p = pred_df[f"actual_price_h{h}"].values
            act_r = pred_df[f"actual_return_h{h}"].values

            # Ridge price & return metrics
            ridge_p = pred_df[f"ridge_pred_price_h{h}"].values
            ridge_r = pred_df[f"ridge_pred_return_h{h}"].values
            summary_price_metrics["ridge"][h] = compute_forecast_metrics(act_p, ridge_p, target_type="price", current_close=c_close_all)
            summary_return_metrics["ridge"][h] = compute_forecast_metrics(act_r, ridge_r, target_type="return")

            # Persistence price & return metrics
            pers_p = pred_df[f"persistence_pred_price_h{h}"].values
            pers_r = pred_df[f"persistence_pred_return_h{h}"].values
            summary_price_metrics["persistence"][h] = compute_forecast_metrics(act_p, pers_p, target_type="price", current_close=c_close_all)
            summary_return_metrics["persistence"][h] = compute_forecast_metrics(act_r, pers_r, target_type="return")

        # Format comparison tables
        comp_rmse = format_metrics_comparison_table(summary_price_metrics["ridge"], summary_price_metrics["persistence"], metric_name="rmse")
        comp_mae = format_metrics_comparison_table(summary_price_metrics["ridge"], summary_price_metrics["persistence"], metric_name="mae")
        comp_da = format_metrics_comparison_table(summary_price_metrics["ridge"], summary_price_metrics["persistence"], metric_name="directional_accuracy")

        return WalkForwardResult(
            predictions_prices=predictions_prices,
            predictions_returns=predictions_returns,
            summary_price_metrics=summary_price_metrics,
            summary_return_metrics=summary_return_metrics,
            fold_metrics=fold_metrics_list,
            comparison_table_rmse=comp_rmse,
            comparison_table_mae=comp_mae,
            comparison_table_da=comp_da,
            n_folds=fold_idx,
            train_window_days=self.splitter.train_window_days,
            test_window_days=self.splitter.test_window_days,
            step_days=self.splitter.step_days,
            is_synthetic=is_synthetic,
            metadata={
                "horizons": self.horizons,
                "n_test_samples": len(pred_df),
                "features_used": feat_cols,
                "strategy_metrics": summary_strategy_metrics,
            },
            strategy_metrics=summary_strategy_metrics,
        )
