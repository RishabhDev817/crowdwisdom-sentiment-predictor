"""
Evaluation metrics module for multi-horizon price and return forecasting.

Computes:
1. Forecast Accuracy Metrics:
   - MAE: Mean Absolute Error
   - RMSE: Root Mean Squared Error
   - MAPE: Mean Absolute Percentage Error (with documented zero/invalid policy)
   - Directional Accuracy: Proportion of correct directional movement forecasts
     relative to prediction-time close.
2. Sample count reporting per horizon and model.
3. Metric comparison tables for side-by-side benchmarking (Ridge vs Persistence).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_directional_coverage(
    predicted_returns: Union[pd.Series, np.ndarray],
    neutral_threshold: float = 0.0,
) -> float:
    """
    Computes directional coverage: percentage of predictions that take a non-neutral stance.

    Args:
        predicted_returns: Model predicted forward returns.
        neutral_threshold: Absolute return threshold below which prediction is considered neutral.
                           Default: 0.0 (strictly non-zero).

    Returns:
        Coverage as a percentage in [0.0, 100.0].
        Returns NaN if no valid predictions exist.
    """
    y_pred = np.asarray(predicted_returns, dtype=float)
    valid = ~np.isnan(y_pred)
    if not valid.any():
        return np.nan
    yp_v = y_pred[valid]
    non_neutral = np.abs(yp_v) > neutral_threshold
    return float(round((non_neutral.sum() / len(yp_v)) * 100.0, 2))


def compute_directional_accuracy(
    actual_returns: Union[pd.Series, np.ndarray],
    predicted_returns: Union[pd.Series, np.ndarray],
    neutral_threshold: float = 0.0,
) -> float:
    """
    Computes directional accuracy strictly among non-neutral predictions.

    Directional match is defined as:
        sign(predicted_return) == sign(actual_return)
    evaluated strictly on non-neutral predictions (|predicted_return| > neutral_threshold).

    If no non-neutral predictions exist (e.g. zero-return persistence baseline where
    predicted return is always 0.0), directional accuracy evaluates to np.nan (reported as N/A).

    Args:
        actual_returns: Ground truth forward returns.
        predicted_returns: Model predicted forward returns.
        neutral_threshold: Threshold below which predictions are neutral (default: 0.0).

    Returns:
        Directional accuracy as a float percentage in [0.0, 100.0], or np.nan if no
        non-neutral predictions exist.
    """
    y_true = np.asarray(actual_returns, dtype=float)
    y_pred = np.asarray(predicted_returns, dtype=float)

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid.any():
        return np.nan

    yt_v = y_true[valid]
    yp_v = y_pred[valid]

    non_neutral = np.abs(yp_v) > neutral_threshold
    if not non_neutral.any():
        return np.nan

    yt_stance = yt_v[non_neutral]
    yp_stance = yp_v[non_neutral]

    sign_true = np.sign(yt_stance)
    sign_pred = np.sign(yp_stance)

    correct = (sign_true == sign_pred).sum()
    return float(round((correct / len(yt_stance)) * 100.0, 2))


def compute_forecast_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_type: str = "price",
    current_close: Optional[Union[pd.Series, np.ndarray]] = None,
    neutral_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes statistical forecast accuracy metrics for a single horizon.

    Metrics computed:
        - sample_count: Number of non-NaN evaluation pairs
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error (percentage, for price targets)
        - directional_accuracy: Directional match percentage among non-neutral predictions
        - directional_coverage: Percentage of predictions taking non-neutral stance
        - is_neutral_baseline: Boolean flag indicating if model takes no directional bets

    Zero / Invalid Handling:
        - NaNs in predictions or ground truth are masked before calculation.
        - In MAPE, actual values with absolute magnitude < 1e-8 evaluate to NaN to
          prevent division by zero.
        - If no valid pairs remain, metrics evaluate to NaN and sample_count to 0.
        - If model has 0.0% directional coverage, directional_accuracy evaluates to np.nan.

    Args:
        y_true: Ground truth actual values (prices or forward returns).
        y_pred: Model predicted values (prices or forward returns).
        target_type: 'price' for nominal prices, 'return' for forward returns.
        current_close: Optional known close price at prediction cutoff (used for directional
                       accuracy when evaluating prices).
        neutral_threshold: Threshold below which predictions are neutral (default: 0.0).

    Returns:
        Dictionary containing computed metrics and sample count.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(yt) | np.isnan(yp))
    n_valid = int(valid.sum())

    if n_valid == 0:
        base_res = {
            "sample_count": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "directional_accuracy": np.nan,
            "directional_coverage": np.nan,
            "is_neutral_baseline": True,
        }
        if target_type == "price":
            base_res["mape"] = np.nan
        return base_res

    yt_v = yt[valid]
    yp_v = yp[valid]

    diff = yp_v - yt_v
    mae = float(round(np.mean(np.abs(diff)), 4))
    rmse = float(round(np.sqrt(np.mean(diff ** 2)), 4))

    result: Dict[str, Any] = {
        "sample_count": n_valid,
        "rmse": rmse,
        "mae": mae,
    }

    if target_type == "price":
        # Guard against passing returns into price metric calculation
        mean_true = float(np.mean(np.abs(yt_v)))
        mean_pred = float(np.mean(np.abs(yp_v)))
        if mean_true > 100.0 and mean_pred < 1.0:
            raise ValueError(
                "Scale mismatch: y_true is nominal price (>100) while y_pred appears to be return (<1). "
                "Specify target_type='return' or reconstruct prices before computing price metrics."
            )

        # MAPE calculation
        denom = np.where(np.abs(yt_v) < 1e-8, np.nan, np.abs(yt_v))
        mape_vals = np.abs(diff) / denom
        result["mape"] = float(round(np.nanmean(mape_vals) * 100.0, 4))

        # Directional accuracy from prices: requires current_close
        if current_close is not None:
            c_arr = np.asarray(current_close, dtype=float)[valid]
            actual_move_ret = (yt_v - c_arr) / c_arr
            predicted_move_ret = (yp_v - c_arr) / c_arr
            result["directional_accuracy"] = compute_directional_accuracy(
                actual_move_ret, predicted_move_ret, neutral_threshold=neutral_threshold
            )
            result["directional_coverage"] = compute_directional_coverage(
                predicted_move_ret, neutral_threshold=neutral_threshold
            )
        else:
            result["directional_accuracy"] = np.nan
            result["directional_coverage"] = np.nan

    elif target_type in {"return", "returns"}:
        # Directional accuracy directly from returns
        result["directional_accuracy"] = compute_directional_accuracy(
            yt_v, yp_v, neutral_threshold=neutral_threshold
        )
        result["directional_coverage"] = compute_directional_coverage(
            yp_v, neutral_threshold=neutral_threshold
        )

    da_val = result.get("directional_accuracy", np.nan)
    cov_val = result.get("directional_coverage", np.nan)
    result["is_neutral_baseline"] = bool(np.isnan(da_val) or (cov_val == 0.0))

    return result


def format_metrics_comparison_table(
    ridge_metrics: Dict[int, Dict[str, Any]],
    persistence_metrics: Dict[int, Dict[str, Any]],
    metric_name: str = "rmse",
) -> pd.DataFrame:
    """
    Creates a side-by-side comparison DataFrame between Ridge and Persistence across horizons.

    Args:
        ridge_metrics: Dict mapping horizon h -> metric dict.
        persistence_metrics: Dict mapping horizon h -> metric dict.
        metric_name: Metric key to compare (e.g. 'rmse', 'mae', 'directional_accuracy').

    Returns:
        DataFrame indexed by Horizon (n+1 through n+5) with columns:
        ['Horizon', 'Sample Count', 'Ridge', 'Persistence', 'Difference'].
    """
    horizons = sorted([h for h in ridge_metrics.keys() if h >= 1 and h in persistence_metrics])

    records = []
    for h in horizons:
        r_val = ridge_metrics[h].get(metric_name, np.nan)
        p_val = persistence_metrics[h].get(metric_name, np.nan)
        count = ridge_metrics[h].get("sample_count", 0)

        diff = round(r_val - p_val, 4) if pd.notna(r_val) and pd.notna(p_val) else np.nan

        row_dict: Dict[str, Any] = {
            "Horizon": f"n+{h}",
            "Sample Count": count,
            f"Ridge_{metric_name}": r_val,
            f"Persistence_{metric_name}": p_val,
            "Difference (Ridge - Persistence)": diff,
        }
        if metric_name == "directional_accuracy":
            row_dict["Ridge_coverage(%)"] = ridge_metrics[h].get("directional_coverage", np.nan)
            row_dict["Persistence_coverage(%)"] = persistence_metrics[h].get("directional_coverage", 0.0)
            row_dict["Baseline_Type"] = "Neutral (no direction)"

        records.append(row_dict)

    return pd.DataFrame(records).set_index("Horizon")
