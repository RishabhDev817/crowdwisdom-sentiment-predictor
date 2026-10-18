"""Evaluation, walk-forward validation, and quantitative trading backtest modules."""

from src.evaluation.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from src.evaluation.metrics import (
    compute_directional_accuracy,
    compute_directional_coverage,
    compute_forecast_metrics,
    format_metrics_comparison_table,
)
from src.evaluation.walk_forward import (
    WalkForwardEvaluator,
    WalkForwardResult,
    WalkForwardSplitter,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "compute_directional_accuracy",
    "compute_directional_coverage",
    "compute_forecast_metrics",
    "format_metrics_comparison_table",
]
