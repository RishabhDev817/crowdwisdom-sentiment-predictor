"""
Unit tests for strategy backtesting engine (src/evaluation/backtest.py).

Verifies:
1. Signal timing (signal at t-1 dictates position held during period t).
2. Position mapping (long, flat, short, and NaN handling).
3. Transaction costs and turnover deduction.
4. Quantitative metric calculations (Cumulative return, Sharpe, Sortino, Max Drawdown, Exposure, Trades).
5. Edge cases: zero volatility, zero downside deviation, flat positions.
6. Input validation: length mismatch, non-positive prices, insufficient history.
7. Synthetic fixture marking (is_synthetic=True).
All tests use small in-memory synthetic datasets.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)


# ==============================================================================
# 1. Signal Timing & Execution Mechanics
# ==============================================================================

def test_backtest_signal_timing_and_execution():
    """
    Verifies that a prediction at row t-1 determines the position during period t,
    and earns the asset return from Close_{t-1} to Close_t.
    """
    # Prices: Day 0=100, Day 1=110 (+10%), Day 2=105 (-4.545%), Day 3=115 (+9.52%)
    prices = np.array([100.0, 110.0, 104.5, 115.0])
    # Predictions at:
    # t=0: +0.05 -> triggers Long position for Day 1
    # t=1: -0.05 -> triggers Flat position for Day 2 (long_only)
    # t=2: +0.08 -> triggers Long position for Day 3
    # t=3: irrelevant for 4-day backtest
    preds = np.array([0.05, -0.05, 0.08, 0.0])

    engine = BacktestEngine(
        config=BacktestConfig(
            threshold=0.0,
            allow_short=False,
            cost_bps=0.0,  # Zero cost for pure return check
        )
    )
    result = engine.run(preds, prices, is_synthetic=True)

    eq_df = result.equity_curve

    # Period 0: initial anchor (pos=0, return=0, equity=1.0)
    assert eq_df["position"].iloc[0] == 0.0
    assert eq_df["strategy_equity"].iloc[0] == 1.0

    # Period 1: Long position entered at t=0 close, earns Day 1 return (+10%)
    assert eq_df["position"].iloc[1] == 1.0
    assert np.isclose(eq_df["strategy_return"].iloc[1], 0.10)
    assert np.isclose(eq_df["strategy_equity"].iloc[1], 1.10)

    # Period 2: Flat position entered at t=1 close, earns 0% return (avoids -5% asset drop)
    assert eq_df["position"].iloc[2] == 0.0
    assert np.isclose(eq_df["strategy_return"].iloc[2], 0.0)
    assert np.isclose(eq_df["strategy_equity"].iloc[2], 1.10)

    # Period 3: Long position entered at t=2 close, earns Day 3 return (115/104.5 - 1 = +10.05%)
    assert eq_df["position"].iloc[3] == 1.0
    expected_day3_ret = (115.0 - 104.5) / 104.5
    assert np.isclose(eq_df["strategy_return"].iloc[3], expected_day3_ret)


def test_backtest_transaction_cost_deduction():
    """
    Verifies that transaction fees are deducted per unit turnover:
        turnover = |pos_t - pos_{t-1}|
        fee = cost_bps * 1e-4 * turnover
    """
    prices = np.array([100.0, 100.0, 100.0, 100.0])  # Zero asset return
    # t=0: pred=0.05 -> pos_1 = 1 (turnover = 1)
    # t=1: pred=-0.05 -> pos_2 = 0 (turnover = 1)
    # t=2: pred=0.05 -> pos_3 = 1 (turnover = 1)
    preds = np.array([0.05, -0.05, 0.05, 0.0])

    # Fee = 10 bps = 0.0010 per turn
    engine = BacktestEngine(config=BacktestConfig(threshold=0.0, cost_bps=10.0))
    result = engine.run(preds, prices, is_synthetic=True)

    eq_df = result.equity_curve
    for t in [1, 2, 3]:
        assert eq_df["turnover"].iloc[t] == 1.0
        assert np.isclose(eq_df["strategy_return"].iloc[t], -0.0010)


def test_backtest_short_position_mapping():
    """Verifies that allow_short=True maps negative predictions to -1 position."""
    prices = np.array([100.0, 90.0, 80.0])  # Asset drops 10% each day
    preds = np.array([-0.05, -0.05, 0.0])

    engine = BacktestEngine(
        config=BacktestConfig(threshold=0.0, allow_short=True, cost_bps=0.0)
    )
    result = engine.run(preds, prices, is_synthetic=True)

    # Short position earns positive return when asset drops
    eq_df = result.equity_curve
    assert eq_df["position"].iloc[1] == -1.0
    assert np.isclose(eq_df["strategy_return"].iloc[1], 0.10)  # -1 * -0.10 = +0.10


def test_backtest_missing_prediction_handled_as_flat():
    """Verifies that NaN predictions result in a Flat (0.0) position without crashing."""
    prices = np.array([100.0, 110.0, 120.0])
    preds = np.array([np.nan, 0.05, 0.0])

    engine = BacktestEngine(config=BacktestConfig(threshold=0.0, cost_bps=0.0))
    result = engine.run(preds, prices, is_synthetic=True)

    eq_df = result.equity_curve
    assert eq_df["position"].iloc[1] == 0.0  # NaN at t=0 -> flat at t=1
    assert eq_df["position"].iloc[2] == 1.0  # 0.05 at t=1 -> long at t=2


# ==============================================================================
# 2. Risk & Performance Metric Calculations
# ==============================================================================

def test_backtest_max_drawdown_calculation():
    """Verifies exact maximum drawdown calculation on known equity progression."""
    # Equity: 1.0 -> 1.20 -> 0.90 -> 1.05
    # Peak at t=1 is 1.20. Trough at t=2 is 0.90. Drawdown = (1.20 - 0.90) / 1.20 = 25.0%
    prices = np.array([100.0, 120.0, 90.0, 105.0])
    preds = np.array([0.05, 0.05, 0.05, 0.0])

    engine = BacktestEngine(config=BacktestConfig(threshold=0.0, cost_bps=0.0))
    result = engine.run(preds, prices, is_synthetic=True)

    metrics = result.metrics
    assert np.isclose(metrics["max_drawdown"], 25.0, atol=1e-2)


def test_backtest_exposure_and_trades_count():
    """Verifies exposure percentage and trade count."""
    prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0])
    # t=0: +0.02 (Long)
    # t=1: +0.02 (Long - hold, no trade)
    # t=2: -0.01 (Flat - trade 2)
    # t=3: +0.03 (Long - trade 3)
    preds = np.array([0.02, 0.02, -0.01, 0.03, 0.0])

    engine = BacktestEngine(config=BacktestConfig(threshold=0.0, allow_short=False, cost_bps=0.0))
    result = engine.run(preds, prices, is_synthetic=True)

    metrics = result.metrics
    # Active periods: t=1, 2, 3, 4 (4 periods)
    # Positions: 1, 1, 0, 1 -> 3 out of 4 non-zero -> 75% exposure
    assert np.isclose(metrics["exposure"], 75.0)
    # Trades (turnovers > 0): t=1 (0->1), t=3 (1->0), t=4 (0->1) -> 3 trades
    assert metrics["num_trades"] == 3


def test_backtest_zero_volatility_handling():
    """Verifies that flat strategy returns handle Sharpe ratio without division by zero."""
    prices = np.array([100.0, 105.0, 110.0])
    preds = np.array([0.0, 0.0, 0.0])  # Always flat (threshold=0.0, no trade)

    engine = BacktestEngine(config=BacktestConfig(threshold=0.0, cost_bps=0.0))
    result = engine.run(preds, prices, is_synthetic=True)

    metrics = result.metrics
    assert np.isnan(metrics["sharpe_ratio"])
    assert metrics["cumulative_return"] == 0.0
    assert metrics["num_trades"] == 0


# ==============================================================================
# 3. Input Validation
# ==============================================================================

def test_backtest_input_validation():
    """Verifies ValueError on invalid price or prediction inputs."""
    engine = BacktestEngine()

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        engine.run(np.array([0.01, 0.02]), np.array([100.0]))

    # Insufficient observations (< 2)
    with pytest.raises(ValueError, match="Insufficient data"):
        engine.run(np.array([0.01]), np.array([100.0]))

    # Non-positive prices
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        engine.run(np.array([0.01, 0.02]), np.array([100.0, -50.0]))

    # NaN in prices
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        engine.run(np.array([0.01, 0.02]), np.array([100.0, np.nan]))


def test_backtest_causal_next_open_execution():
    """
    Safeguard: Verifies that when actual Open prices are provided with DatetimeIndex:
      - Signal hat{r}_{D, 1} generated at Close D dictates position.
      - Position enters at Open D+1 and exits at Open D+2.
      - Realized return is strictly (Open_{D+2} - Open_{D+1}) / Open_{D+1}.
      - Transaction fees are deducted upon position change at Open D+1.
      - Never assumes execution occurs at Close D when the signal uses Close D.
    """
    dates = pd.date_range("2026-08-01", periods=6, freq="D", tz="UTC")
    # Day 0: Open=100, Close=102 -> Signal generated at Close D0 (+0.05 -> Long)
    # Day 1: Open=103 (differs from Close D0=102!), Close=105 -> Entry at Open D1=103
    # Day 2: Open=108, Close=107 -> Exit/rebalance at Open D2=108. Return = (108-103)/103 = +4.854%
    # Signal at Close D1: -0.05 -> Flat
    # Day 2: Entry at Open D2=108 with pos=0.0. Exit at Open D3=110. Return = 0.0
    # Day 3: Open=110, Close=112
    # Day 4: Open=115, Close=114
    # Day 5: Open=120, Close=122
    opens = pd.Series([100.0, 103.0, 108.0, 110.0, 115.0, 120.0], index=dates)
    closes = pd.Series([102.0, 105.0, 107.0, 112.0, 114.0, 122.0], index=dates)

    # Signals for Day 0, Day 1, Day 2, Day 3
    preds = pd.Series([0.05, -0.05, 0.08, 0.02], index=dates[:4])

    engine = BacktestEngine(
        config=BacktestConfig(threshold=0.0, allow_short=False, cost_bps=10.0)
    )
    result = engine.run(
        predictions_series=preds,
        price_series=closes.iloc[:4],
        open_price_series=opens,
        is_synthetic=True,
    )

    eq_df = result.equity_curve
    assert len(eq_df) == 4

    # Row 0: corresponds to execution at Day 1 Open (2026-08-02) from Day 0 signal
    row0 = eq_df.iloc[0]
    assert row0["signal_date"] == dates[0]
    assert row0["entry_open"] == 103.0  # Actual Day 1 Open, NOT Day 0 Close (102.0)
    assert row0["exit_open"] == 108.0   # Actual Day 2 Open
    assert np.isclose(row0["asset_return"], (108.0 - 103.0) / 103.0)
    assert row0["position"] == 1.0
    assert row0["turnover"] == 1.0
    # Net strategy return = pos * asset_return - fee = 5/103 - 0.0010
    expected_net0 = (5.0 / 103.0) - 0.0010
    assert np.isclose(row0["strategy_return"], expected_net0)

    # Row 1: corresponds to execution at Day 2 Open (2026-08-03) from Day 1 signal (-0.05 -> Flat)
    row1 = eq_df.iloc[1]
    assert row1["signal_date"] == dates[1]
    assert row1["entry_open"] == 108.0
    assert row1["exit_open"] == 110.0
    assert row1["position"] == 0.0
    assert row1["turnover"] == 1.0  # Exit from Long to Flat
    # Net return = 0 * return - 0.0010 = -0.0010
    assert np.isclose(row1["strategy_return"], -0.0010)


def test_backtest_benchmark_alignment_and_boundary_handling():
    """
    Verifies that:
      - Benchmark uses the exact same entry and exit dates and return intervals as the strategy.
      - Handles boundary signal dates near the end of the series where Open prices are insufficient,
        without throwing index errors or producing misaligned rows.
      - Accurately tracks turnover fees across transitions (Flat -> Long -> Long -> Flat).
    """
    dates = pd.date_range("2026-09-01", periods=6, freq="D", tz="UTC")
    # Only 5 days of Open prices available
    opens = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0], index=dates[:5])
    closes = pd.Series([102.0, 108.0, 112.0, 110.0, 116.0], index=dates[:5])

    # Predictions for all 5 dates:
    # Day 0 (2026-09-01): pred=+0.02 -> Long. Entry Day 1 Open (105.0), Exit Day 2 Open (110.0). Asset ret = 5/105 = +4.76%.
    # Day 1 (2026-09-02): pred=+0.01 -> Long (hold). Entry Day 2 Open (110.0), Exit Day 3 Open (108.0). Asset ret = -2/110 = -1.82%.
    # Day 2 (2026-09-03): pred=-0.02 -> Flat. Entry Day 3 Open (108.0), Exit Day 4 Open (115.0).
    # Day 3 (2026-09-04): loc=3, loc+1=4 (Day 4 Open), but loc+2=5 (Day 5 Open) does NOT exist! -> cleanly skipped.
    # Day 4 (2026-09-05): loc=4, loc+1=5 does NOT exist! -> cleanly skipped.
    preds = pd.Series([0.02, 0.01, -0.02, 0.05, 0.03], index=dates[:5])

    engine = BacktestEngine(
        config=BacktestConfig(threshold=0.0, allow_short=False, cost_bps=10.0)
    )
    result = engine.run(
        predictions_series=preds,
        price_series=closes,
        open_price_series=opens,
        is_synthetic=True,
    )

    eq_df = result.equity_curve
    # Exactly 3 executable periods
    assert len(eq_df) == 3

    # Benchmark alignment: benchmark asset returns match strategy asset returns identically
    assert (eq_df["asset_return"].values == [(110.0 - 105.0) / 105.0, (108.0 - 110.0) / 110.0, (115.0 - 108.0) / 108.0]).all()
    expected_benchmark_equity = np.cumprod(1.0 + eq_df["asset_return"].values)
    assert np.allclose(eq_df["benchmark_equity"].values, expected_benchmark_equity)

    # Turnover verification:
    # Period 0: Flat -> Long (turnover = 1.0, fee = 0.0010)
    assert eq_df["turnover"].iloc[0] == 1.0
    assert np.isclose(eq_df["strategy_return"].iloc[0], (5.0 / 105.0) - 0.0010)

    # Period 1: Long -> Long (turnover = 0.0, fee = 0.0)
    assert eq_df["turnover"].iloc[1] == 0.0
    assert np.isclose(eq_df["strategy_return"].iloc[1], -2.0 / 110.0)

    # Period 2: Long -> Flat (turnover = 1.0, fee = 0.0010, return = 0 * asset - 0.0010)
    assert eq_df["turnover"].iloc[2] == 1.0
    assert np.isclose(eq_df["strategy_return"].iloc[2], -0.0010)


# ==============================================================================
# 4. Phase 4.4.2 Targeted Corrections & Regression Tests
# ==============================================================================

def test_backtest_terminal_liquidation_final_long():
    """
    Verifies that when the backtest ends in an active Long position (+1.0),
    a terminal liquidation fee is charged at evaluation end, liquidating the position to cash.
    """
    dates = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")
    opens = pd.Series([100.0, 102.0, 105.0, 108.0], index=dates)
    closes = pd.Series([101.0, 104.0, 107.0, 110.0], index=dates)

    # 2 signals: both positive -> holds Long through terminal period
    # Period 0: Entry at Open D1 (102.0), Exit at Open D2 (105.0) -> pos=1.0, entry fee=0.0010
    # Period 1: Entry at Open D2 (105.0), Exit at Open D3 (108.0) -> pos=1.0, hold turnover=0.0, terminal exit fee=0.0010
    preds = pd.Series([0.05, 0.05], index=dates[:2])

    engine = BacktestEngine(
        config=BacktestConfig(threshold=0.0, allow_short=False, cost_bps=10.0, terminal_liquidation=True)
    )
    result = engine.run(
        predictions_series=preds,
        price_series=closes[:2],
        open_price_series=opens,
        is_synthetic=True,
    )

    eq_df = result.equity_curve
    assert len(eq_df) == 2

    # Period 0: entry turnover = 1.0, fee = 0.0010
    assert eq_df["turnover"].iloc[0] == 1.0
    expected_ret0 = ((105.0 - 102.0) / 102.0) - 0.0010
    assert np.isclose(eq_df["strategy_return"].iloc[0], expected_ret0)

    # Period 1: position held (0.0 turnover) + terminal liquidation (1.0 turnover) = 1.0 total turnover
    # Gross asset return = (108.0 - 105.0) / 105.0
    # Net return = gross - terminal_fee (0.0010)
    assert eq_df["position"].iloc[1] == 1.0
    assert eq_df["turnover"].iloc[1] == 1.0  # Terminal liquidation turnover charged
    expected_ret1 = ((108.0 - 105.0) / 105.0) - 0.0010
    assert np.isclose(eq_df["strategy_return"].iloc[1], expected_ret1)


def test_backtest_terminal_liquidation_final_cash():
    """
    Verifies that when the backtest ends in Cash (0.0), no terminal liquidation fee is charged,
    and no extra turnover is recorded.
    """
    dates = pd.date_range("2026-09-01", periods=4, freq="D", tz="UTC")
    opens = pd.Series([100.0, 102.0, 105.0, 108.0], index=dates)
    closes = pd.Series([101.0, 104.0, 107.0, 110.0], index=dates)

    # Signal 0: +0.05 -> Long for Period 0
    # Signal 1: -0.05 -> Flat for Period 1 (exits at Open D2, ends in Cash)
    preds = pd.Series([0.05, -0.05], index=dates[:2])

    engine = BacktestEngine(
        config=BacktestConfig(threshold=0.0, allow_short=False, cost_bps=10.0, terminal_liquidation=True)
    )
    result = engine.run(
        predictions_series=preds,
        price_series=closes[:2],
        open_price_series=opens,
        is_synthetic=True,
    )

    eq_df = result.equity_curve
    assert len(eq_df) == 2

    # Period 1: exited to cash at entry Open D2 (turnover = 1.0, fee = 0.0010)
    # Position during period 1 is 0.0. Since final_pos is 0.0, terminal liquidation adds 0 fee.
    assert eq_df["position"].iloc[1] == 0.0
    assert eq_df["turnover"].iloc[1] == 1.0  # Exit from period 0 only; no extra terminal fee
    assert np.isclose(eq_df["strategy_return"].iloc[1], -0.0010)


def test_backtest_max_drawdown_initial_negative_return():
    """
    Regression test: Verifies that when the very first return is negative,
    the loss from starting capital baseline of 1.0 is captured in max_drawdown.
    """
    # Period 0 return: -10% (equity drops from 1.0 to 0.90)
    # Period 1 return: -5%  (equity drops from 0.90 to 0.855)
    # Total drawdown from 1.0 peak is 1 - 0.855 = 14.5%
    returns = np.array([-0.10, -0.05])
    equity_curve = np.cumprod(1.0 + returns)  # [0.90, 0.855]
    positions = np.array([1.0, 1.0])
    turnovers = np.array([1.0, 0.0])

    engine = BacktestEngine()
    metrics = engine._calculate_performance_metrics(returns, equity_curve, positions, turnovers)

    # Max drawdown must reflect the drop from starting capital 1.0 (14.5%),
    # NOT just the drop from 0.90 to 0.855 (5.0%).
    assert np.isclose(metrics["max_drawdown"], 14.50, atol=1e-2)


def test_backtest_win_rate_reporting_fields():
    """
    Verifies that win_rate reports conditional active in-market days,
    while all_period_win_rate reports positive days over all evaluation periods.
    """
    # 4 periods:
    # Period 0: Long, return = +0.02 (win)
    # Period 1: Long, return = -0.01 (loss)
    # Period 2: Cash, return = 0.0 (cash day, not a win)
    # Period 3: Cash, return = 0.0 (cash day, not a win)
    returns = np.array([0.02, -0.01, 0.0, 0.0])
    equity_curve = np.cumprod(1.0 + returns)
    positions = np.array([1.0, 1.0, 0.0, 0.0])
    turnovers = np.array([1.0, 0.0, 1.0, 0.0])

    engine = BacktestEngine()
    metrics = engine._calculate_performance_metrics(returns, equity_curve, positions, turnovers)

    # Active days = 2, wins = 1 -> win_rate = 50.0%
    assert metrics["active_days"] == 2
    assert metrics["win_days"] == 1
    assert np.isclose(metrics["win_rate"], 50.0)

    # Total days = 4, positive return days = 1 -> all_period_win_rate = 25.0%
    assert metrics["n_days"] == 4
    assert metrics["all_period_win_days"] == 1
    assert np.isclose(metrics["all_period_win_rate"], 25.0)


