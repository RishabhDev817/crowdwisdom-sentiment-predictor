"""
Quantitative trading strategy backtest module.

Simulates strategy execution based strictly on causal prediction cutoff signals:
1. Signal Timing & Execution (Causal Next-Open Policy):
   - At prediction cutoff D (Close D at 23:59:59 UTC), the model calculates forward return forecast hat{r}_{D, 1}.
   - The trade executes at the next available execution price: Open_{D+1} (at D+1 00:00:00 UTC).
   - Position is held across the 24-hour continuous holding period until Open_{D+2}.
   - Realized asset return over the holding period: (Open_{D+2} - Open_{D+1}) / Open_{D+1}.
   - Note: Never assumes execution occurs at Close D when the signal uses Close D.
2. Position Mapping:
   - Long (+1) if hat{r}_{D, 1} > threshold.
   - Short (-1) if hat{r}_{D, 1} < -threshold and allow_short=True, else Flat (0).
   - Flat (0) if |hat{r}_{D, 1}| <= threshold or if prediction is NaN.
3. Transaction Costs & Turnover:
   - Rebalance occurs at Open_{D+1}: turnover = |pos_{D+1} - pos_D|.
   - Transaction fee deducted per unit turnover: cost_bps * 1e-4.
   - Net strategy return: R_{net} = pos * R_{asset} - cost * turnover.
4. Annualization Assumption:
   - Crypto trades 24/7/365, so daily returns are annualized using 365 periods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Strategy backtest configuration parameters."""
    threshold: float = 0.0          # Minimum predicted return to trigger entry
    allow_short: bool = False       # Long-only / Cash by default
    cost_bps: float = 10.0          # 10 bps (0.10%) transaction fee per unit turnover
    risk_free_rate: float = 0.0     # 0.0% annualized risk-free rate
    annualization_factor: int = 365 # 365 days for 24/7/365 crypto markets
    execution_policy: str = "next_open"  # 'next_open' (causal: signal at Close_D, execute Open_{D+1})
    terminal_liquidation: bool = True    # Liquidate open position at evaluation end


@dataclass
class BacktestResult:
    """Holds time series equity curves, trade logs, and quantitative performance metrics."""
    equity_curve: pd.DataFrame      # DataFrame with Date, position, asset_return, strategy_return, cumulative_equity
    metrics: Dict[str, Any]         # Strategy quantitative metrics
    benchmark_metrics: Dict[str, Any] # Buy-and-Hold baseline metrics
    config: BacktestConfig
    is_synthetic: bool = False


class BacktestEngine:
    """
    Executes backtests on out-of-sample forward return predictions.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config if config is not None else BacktestConfig()

    def run(
        self,
        predictions_series: Union[pd.Series, np.ndarray],
        price_series: Union[pd.Series, np.ndarray],
        open_price_series: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        is_synthetic: bool = False,
    ) -> BacktestResult:
        """
        Executes strategy backtest using 1-day ahead predicted returns.

        If open_price_series is provided with a DatetimeIndex (matching price data),
        the engine executes strictly under the causal Next-Open policy:
          - Prediction at Close D dictates position entered at Open D+1.
          - Position held from Open D+1 to Open D+2.
          - Asset return: (Open_{D+2} - Open_{D+1}) / Open_{D+1}.
          - Rebalance fee deducted at Open D+1 on turnover.

        Args:
            predictions_series: 1-day ahead predicted returns hat{r}_{t-1, 1}.
            price_series: Actual Close prices matching each observation.
            open_price_series: Optional actual Open prices for causal next-open execution.
            dates: Optional DatetimeIndex for alignment.
            is_synthetic: Whether data is synthetic.

        Returns:
            BacktestResult with complete metrics and equity curves.
        """
        # Causal Series execution mode with DatetimeIndex
        if (
            open_price_series is not None
            and isinstance(open_price_series, pd.Series)
            and isinstance(predictions_series, pd.Series)
            and isinstance(predictions_series.index, pd.DatetimeIndex)
        ):
            records = []
            prev_pos = 0.0
            cost_rate = self.config.cost_bps * 1e-4

            for sig_date, pred_val in predictions_series.items():
                if sig_date not in open_price_series.index:
                    continue
                loc = open_price_series.index.get_loc(sig_date)
                # Next day open (entry) and subsequent day open (exit/rebalance)
                if loc + 1 >= len(open_price_series) or loc + 2 >= len(open_price_series):
                    continue

                entry_date = open_price_series.index[loc + 1]
                exit_date = open_price_series.index[loc + 2]
                entry_open = float(open_price_series.iloc[loc + 1])
                exit_open = float(open_price_series.iloc[loc + 2])

                if entry_open <= 0 or exit_open <= 0:
                    continue

                asset_ret = (exit_open - entry_open) / entry_open

                # Position decision from signal at sig_date close
                if np.isnan(pred_val):
                    pos = 0.0
                elif pred_val > self.config.threshold:
                    pos = 1.0
                elif pred_val < -self.config.threshold and self.config.allow_short:
                    pos = -1.0
                else:
                    pos = 0.0

                turnover = abs(pos - prev_pos)
                prev_pos = pos
                fee = cost_rate * turnover
                strat_ret = pos * asset_ret - fee

                records.append({
                    "signal_date": sig_date,
                    "date": entry_date,
                    "exit_date": exit_date,
                    "entry_open": entry_open,
                    "exit_open": exit_open,
                    "prediction": pred_val,
                    "position": pos,
                    "turnover": turnover,
                    "asset_return": asset_ret,
                    "strategy_return": strat_ret,
                })

            if len(records) < 2:
                raise ValueError(
                    f"Insufficient causal observations: requires at least 2 executable periods, got {len(records)}."
                )

            # Terminal liquidation accounting:
            # If the strategy ends in an active position (pos != 0.0), charge the modeled
            # exit liquidation fee (cost_rate * abs(final_pos)) at terminal exit date.
            final_pos = records[-1]["position"]
            if self.config.terminal_liquidation and final_pos != 0.0:
                terminal_turnover = abs(final_pos)
                terminal_fee = cost_rate * terminal_turnover
                records[-1]["strategy_return"] -= terminal_fee
                records[-1]["turnover"] += terminal_turnover

            equity_df = pd.DataFrame(records).set_index("date")
            equity_df["strategy_equity"] = np.cumprod(1.0 + equity_df["strategy_return"].values)
            equity_df["benchmark_equity"] = np.cumprod(1.0 + equity_df["asset_return"].values)

            strat_metrics = self._calculate_performance_metrics(
                equity_df["strategy_return"].values,
                equity_df["strategy_equity"].values,
                equity_df["position"].values,
                equity_df["turnover"].values,
            )
            bench_metrics = self._calculate_performance_metrics(
                equity_df["asset_return"].values,
                equity_df["benchmark_equity"].values,
                np.ones(len(equity_df)),
                np.zeros(len(equity_df)),
            )

            return BacktestResult(
                equity_curve=equity_df,
                metrics=strat_metrics,
                benchmark_metrics=bench_metrics,
                config=self.config,
                is_synthetic=is_synthetic,
            )

        if len(predictions_series) != len(price_series):
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions_series)}) != prices ({len(price_series)})"
            )

        n = len(predictions_series)
        if n < 2:
            raise ValueError(f"Insufficient data for backtest: requires at least 2 observations, got {n}.")

        preds = np.asarray(predictions_series, dtype=float)
        prices = np.asarray(price_series, dtype=float)

        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Price series contains non-positive or non-finite values.")

        # Compute asset 1-day returns: if open_price_series is provided, use open-to-open, else close-to-close proxy
        asset_returns = np.zeros(n, dtype=float)
        if open_price_series is not None:
            opens = np.asarray(open_price_series, dtype=float)
            if len(opens) != n:
                raise ValueError(f"Length mismatch: open prices ({len(opens)}) != predictions ({n})")
            if not np.isfinite(opens).all() or (opens <= 0).any():
                raise ValueError("Open price series contains non-positive or non-finite values.")
            asset_returns[1:] = (opens[1:] - opens[:-1]) / opens[:-1]
        else:
            asset_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # Map predictions at t-1 to positions held during t
        # Prediction made at t-1 dictates position held during period t
        positions = np.zeros(n, dtype=float)

        for t in range(1, n):
            pred_t_minus_1 = preds[t - 1]
            if np.isnan(pred_t_minus_1):
                positions[t] = 0.0  # Flat on missing prediction
            elif pred_t_minus_1 > self.config.threshold:
                positions[t] = 1.0  # Long
            elif pred_t_minus_1 < -self.config.threshold and self.config.allow_short:
                positions[t] = -1.0 # Short
            else:
                positions[t] = 0.0  # Flat

        # Compute turnover and transaction costs
        cost_rate = self.config.cost_bps * 1e-4
        turnovers = np.zeros(n, dtype=float)
        turnovers[1:] = np.abs(positions[1:] - positions[:-1])

        # Net strategy return: pos_t * asset_return_t - cost * turnover_t
        strategy_returns = np.zeros(n, dtype=float)
        strategy_returns[1:] = positions[1:] * asset_returns[1:] - cost_rate * turnovers[1:]

        # Cumulative equity curves (starting at 1.0)
        # Note: Period 0 has return 0.0 (entry decision made at end of period 0)
        strat_equity = np.cumprod(1.0 + strategy_returns)
        bench_returns = asset_returns.copy()
        bench_equity = np.cumprod(1.0 + bench_returns)

        # Build equity curve DataFrame
        df_index = dates if dates is not None else pd.RangeIndex(n)
        equity_df = pd.DataFrame(
            {
                "price": prices,
                "prediction": preds,
                "position": positions,
                "turnover": turnovers,
                "asset_return": asset_returns,
                "strategy_return": strategy_returns,
                "strategy_equity": strat_equity,
                "benchmark_equity": bench_equity,
            },
            index=df_index,
        )

        # Calculate metrics over active evaluation window (t >= 1)
        active_strat_ret = strategy_returns[1:]
        active_bench_ret = bench_returns[1:]
        active_positions = positions[1:]

        strat_metrics = self._calculate_performance_metrics(
            active_strat_ret,
            strat_equity,
            active_positions,
            turnovers[1:],
        )
        bench_metrics = self._calculate_performance_metrics(
            active_bench_ret,
            bench_equity,
            np.ones(len(active_bench_ret)),
            np.zeros(len(active_bench_ret)),
        )

        return BacktestResult(
            equity_curve=equity_df,
            metrics=strat_metrics,
            benchmark_metrics=bench_metrics,
            config=self.config,
            is_synthetic=is_synthetic,
        )

    def _calculate_performance_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        positions: np.ndarray,
        turnovers: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates quantitative risk and return metrics."""
        n_periods = len(returns)
        if n_periods == 0:
            return {
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "max_drawdown": 0.0,
                "num_trades": 0,
                "exposure": 0.0,
                "win_rate": 0.0,
                "win_days": 0,
                "active_days": 0,
                "all_period_win_rate": 0.0,
                "all_period_win_days": 0,
                "n_days": 0,
            }

        # 1. Cumulative Return
        cum_ret = float(equity_curve[-1] - 1.0)

        # 2. Annualized Return (CAGR)
        # Using 365 days for 24/7 crypto
        ann_factor = self.config.annualization_factor
        if equity_curve[-1] > 0:
            ann_ret = float((equity_curve[-1]) ** (ann_factor / n_periods) - 1.0)
        else:
            ann_ret = -1.0

        # 3. Annualized Volatility
        daily_vol = float(np.std(returns, ddof=1)) if n_periods > 1 else 0.0
        ann_vol = float(daily_vol * np.sqrt(ann_factor))

        # 4. Sharpe Ratio
        rf = self.config.risk_free_rate
        if ann_vol > 1e-8:
            sharpe = float(round((ann_ret - rf) / ann_vol, 4))
        else:
            sharpe = np.nan

        # 5. Sortino Ratio (downside deviation using returns < 0)
        downside_returns = np.minimum(returns, 0.0)
        downside_sq = np.mean(downside_returns ** 2)
        downside_dev = float(np.sqrt(downside_sq) * np.sqrt(ann_factor))

        if downside_dev > 1e-8:
            sortino = float(round((ann_ret - rf) / downside_dev, 4))
        else:
            # If no downside deviation, return NaN if flat/loss, or positive indicator
            sortino = np.nan if ann_ret <= 0 else np.inf

        # 6. Maximum Drawdown (explicitly including initial capital baseline of 1.0)
        equity_with_initial = np.concatenate([[1.0], equity_curve])
        running_max = np.maximum.accumulate(equity_with_initial)
        drawdowns = 1.0 - (equity_with_initial / running_max)
        max_dd = float(round(np.max(drawdowns) * 100.0, 4))

        # 7. Trading and Exposure Statistics
        num_trades = int(np.sum(turnovers > 0))
        exposure = float(round(np.mean(positions != 0.0) * 100.0, 2))

        # Win rate: days with positive return while in position (active in-market days)
        in_pos = positions != 0.0
        n_in_pos = int(in_pos.sum())
        if n_in_pos > 0:
            win_days = int((returns[in_pos] > 0.0).sum())
            win_rate = float(round((win_days / n_in_pos) * 100.0, 2))
        else:
            win_days = 0
            win_rate = 0.0

        # All-period win rate: positive return days across all evaluation periods
        all_period_win_days = int((returns > 0.0).sum())
        all_period_win_rate = float(round((all_period_win_days / n_periods) * 100.0, 2))

        return {
            "cumulative_return": round(cum_ret * 100.0, 2),     # Percentage
            "annualized_return": round(ann_ret * 100.0, 2),     # Percentage
            "annualized_volatility": round(ann_vol * 100.0, 2), # Percentage
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino if not np.isinf(sortino) else "Inf (No Downside)",
            "max_drawdown": max_dd,                             # Percentage
            "num_trades": num_trades,
            "exposure": exposure,                               # Percentage
            "win_rate": win_rate,                               # Percentage (active in-market days)
            "win_days": win_days,
            "active_days": n_in_pos,
            "all_period_win_rate": all_period_win_rate,
            "all_period_win_days": all_period_win_days,
            "n_days": n_periods,
        }
