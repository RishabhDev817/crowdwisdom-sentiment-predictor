"""
Daily sentiment feature aggregation and price alignment module.

This module aggregates post-level sentiment scores into daily UTC calendar metrics
and aligns them with daily BTC price bars without look-ahead bias.

Availability and Timestamp Convention:
    1. Daily BTC Bar for Date D:
       Represents trading from D 00:00:00 UTC through D 23:59:59 UTC.
       The Close price for Date D is only finalized after D 23:59:59 UTC.
    2. Daily Sentiment Features for Date D:
       Aggregates all social posts timestamped between D 00:00:00 UTC and D 23:59:59 UTC.
       Therefore, the same-day sentiment features for Date D are known only at the end of Date D.
    3. Look-Ahead Bias Prevention:
       Same-day sentiment features for Date D MUST NOT be used to predict Date D prices
       or intra-day moves prior to Date D close.
       In downstream forecasting (Phase 3), day D sentiment features and day D close price
       become available at midnight UTC transitioning into D+1, and can only be used to forecast
       subsequent horizons (e.g. D+1 Close, D+1 Return, etc.).
       Posts from Date D+1 strictly never affect Date D feature aggregates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from src.processing.sentiment import score_posts

logger = logging.getLogger(__name__)

# Default engagement weights for interaction metrics
DEFAULT_ENGAGEMENT_WEIGHTS: Dict[str, float] = {
    "like_count": 1.0,
    "retweet_count": 2.0,
    "quote_count": 1.5,
    "reply_count": 1.0,
}

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

DAILY_FEATURE_COLUMNS = [
    "post_count",
    "positive_post_count",
    "negative_post_count",
    "neutral_post_count",
    "sentiment_mean",
    "subjectivity_mean",
    "sentiment_weighted",
    "sentiment_weighted_available",
    "has_posts",
]

NOWCAST_HORIZON: int = 0
FORECAST_HORIZONS: List[int] = [1, 2, 3, 4, 5]
DEFAULT_HORIZONS: List[int] = [1, 2, 3, 4, 5]
ALL_HORIZONS: List[int] = [0, 1, 2, 3, 4, 5]

PRICE_DERIVED_FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_7d",
    "volatility_7d",
    "high_low_ratio",
    "close_open_ratio",
]


def calculate_post_engagement(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Calculates engagement weight for each post row using available non-null metrics.

    Important:
        Missing metrics (None / NaN) are NOT converted to zero. If a post has no
        observed engagement metrics across all configured columns, its engagement
        weight evaluates to NaN (missing) rather than 0.0.

    Args:
        df: DataFrame containing social post engagement columns.
        weights: Dictionary mapping column names to numeric weights.
                 Defaults to DEFAULT_ENGAGEMENT_WEIGHTS.

    Returns:
        pd.Series of numeric engagement weights with NaN for posts lacking all metrics.
    """
    if weights is None:
        weights = DEFAULT_ENGAGEMENT_WEIGHTS

    available_cols = [c for c in weights if c in df.columns]
    if not available_cols:
        return pd.Series(np.nan, index=df.index, dtype=float)

    def _row_engagement(row: pd.Series) -> float:
        valid_contributions = []
        for col in available_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    num_val = float(val)
                    if num_val >= 0:
                        valid_contributions.append(weights[col] * num_val)
                except (ValueError, TypeError):
                    continue

        if not valid_contributions:
            return np.nan
        return float(sum(valid_contributions))

    return df[available_cols].apply(_row_engagement, axis=1)


def aggregate_daily_sentiment(
    posts_df: pd.DataFrame,
    text_column: str = "text",
    created_at_column: str = "created_at",
    deduplicate: bool = True,
    engagement_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Aggregates post-level sentiment scores into daily UTC calendar metrics.

    Features calculated per UTC day:
        - post_count: Total posts on that UTC day.
        - positive_post_count: Posts with sentiment_score > 0.05.
        - negative_post_count: Posts with sentiment_score < -0.05.
        - neutral_post_count: Posts with -0.05 <= sentiment_score <= 0.05.
        - sentiment_mean: Arithmetic mean of sentiment_score.
        - subjectivity_mean: Arithmetic mean of subjectivity_score.
        - sentiment_weighted: Engagement-weighted sentiment score.
        - sentiment_weighted_available: Boolean flag indicating if valid non-zero
          engagement was present to calculate the weighted score.
        - has_posts: Boolean flag indicating presence of posts (always True for aggregated days).

    Weighting & Zero-Denominator Behavior:
        If all posts on a day have missing engagement metrics or the total daily
        engagement sum is 0.0, the weighted score uses a neutral fallback of 0.0
        and sentiment_weighted_available is set to False.

    Args:
        posts_df: DataFrame of social posts.
        text_column: Name of post text column (default: 'text').
        created_at_column: Name of timestamp column (default: 'created_at').
        deduplicate: Whether to deterministically deduplicate by post_id (default: True).
        engagement_weights: Custom metric weights for engagement weighting.

    Returns:
        DataFrame indexed by UTC Date string ('YYYY-MM-DD') containing daily features.

    Raises:
        TypeError: If posts_df is not a pandas DataFrame.
        KeyError: If created_at_column or text_column is missing.
    """
    if not isinstance(posts_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(posts_df).__name__}")

    if posts_df.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS, index=pd.Index([], name="Date"))

    if created_at_column not in posts_df.columns:
        raise KeyError(f"Timestamp column '{created_at_column}' not found in DataFrame.")

    if text_column not in posts_df.columns:
        raise KeyError(f"Text column '{text_column}' not found in DataFrame.")

    df = posts_df.copy()

    # Deterministic deduplication by post_id preserving the richest record
    if deduplicate and "post_id" in df.columns:
        metric_cols = [
            "like_count",
            "retweet_count",
            "reply_count",
            "quote_count",
            "view_count",
            "author_followers",
        ]
        available_metrics = [c for c in metric_cols if c in df.columns]
        if available_metrics:
            df["_metric_richness"] = df[available_metrics].notna().sum(axis=1)
            df = df.sort_values(
                by=["post_id", "_metric_richness"],
                ascending=[True, False],
                kind="mergesort",
            )
            df = df.drop_duplicates(subset=["post_id"], keep="first").drop(columns=["_metric_richness"])
        else:
            df = df.drop_duplicates(subset=["post_id"], keep="first")

    # Score posts if not already scored
    if "sentiment_score" not in df.columns:
        df = score_posts(df, text_column=text_column)

    # Parse timestamps strictly to UTC
    try:
        utc_timestamps = pd.to_datetime(df[created_at_column], utc=True, errors="coerce", format="mixed")
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp column '{created_at_column}' as UTC: {e}") from e

    valid_mask = utc_timestamps.notna()
    if not valid_mask.all():
        logger.warning(
            f"Dropping {(~valid_mask).sum()} records with unparseable timestamps."
        )
        df = df[valid_mask].copy()
        utc_timestamps = utc_timestamps[valid_mask]

    if df.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS, index=pd.Index([], name="Date"))

    # Extract UTC calendar date (YYYY-MM-DD)
    df["_utc_date"] = utc_timestamps.dt.strftime("%Y-%m-%d")

    # Compute post-level engagement weight
    df["_engagement_weight"] = calculate_post_engagement(df, weights=engagement_weights)

    daily_records: List[Dict[str, Any]] = []
    grouped = df.groupby("_utc_date", sort=True)

    for date_str, group in grouped:
        p_count = len(group)

        # Polarity counts based on standard thresholds
        pos_count = int((group["sentiment_score"] > 0.05).sum())
        neg_count = int((group["sentiment_score"] < -0.05).sum())
        neu_count = int(p_count - pos_count - neg_count)

        mean_sent = round(float(group["sentiment_score"].mean()), 4)

        if "subjectivity_score" in group.columns and group["subjectivity_score"].notna().any():
            mean_subj = round(float(group["subjectivity_score"].mean()), 4)
        else:
            mean_subj = np.nan

        # Engagement-weighted sentiment calculation
        valid_eng = group[group["_engagement_weight"].notna() & group["sentiment_score"].notna()]
        total_eng = float(valid_eng["_engagement_weight"].sum()) if not valid_eng.empty else 0.0

        if total_eng > 0.0:
            weighted_val = (valid_eng["_engagement_weight"] * valid_eng["sentiment_score"]).sum() / total_eng
            sent_weighted = round(float(max(-1.0, min(1.0, weighted_val))), 4)
            weighted_avail = True
        else:
            # Documented neutral fallback when engagement is missing or total is 0.0
            sent_weighted = 0.0
            weighted_avail = False

        record = {
            "Date": date_str,
            "post_count": p_count,
            "positive_post_count": pos_count,
            "negative_post_count": neg_count,
            "neutral_post_count": neu_count,
            "sentiment_mean": mean_sent,
            "subjectivity_mean": mean_subj,
            "sentiment_weighted": sent_weighted,
            "sentiment_weighted_available": weighted_avail,
            "has_posts": True,
        }
        daily_records.append(record)

    agg_df = pd.DataFrame(daily_records).set_index("Date")
    return agg_df


def _extract_date_key_series(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    """
    Extracts a standardized 'YYYY-MM-DD' date string Series from DataFrame index or Date column.

    Returns:
        Tuple of (Series of YYYY-MM-DD strings, column name or None if from index).
    """
    if "Date" in df.columns:
        col = df["Date"]
        parsed = pd.to_datetime(col, utc=True)
        return parsed.dt.strftime("%Y-%m-%d"), "Date"
    elif isinstance(df.index, pd.DatetimeIndex):
        parsed = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        return pd.Series(parsed.strftime("%Y-%m-%d"), index=df.index), None
    else:
        # Try parsing string index
        try:
            parsed = pd.to_datetime(df.index, utc=True)
            return pd.Series(parsed.strftime("%Y-%m-%d"), index=df.index), None
        except Exception as e:
            raise ValueError(f"Unable to parse DataFrame date index or Date column: {e}") from e


def validate_calendar_continuity(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
) -> pd.DatetimeIndex:
    """
    Validates that a DataFrame has a unique, strictly ascending, and continuous
    daily UTC calendar (spaced exactly 1 calendar day apart).

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty, contains duplicate dates, is unsorted,
                    or contains missing dates / irregular day spacing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Price DataFrame is empty or None.")

    if date_column and date_column in df.columns:
        date_series = pd.to_datetime(df[date_column], utc=True)
    elif "Date" in df.columns:
        date_series = pd.to_datetime(df["Date"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        date_series = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    else:
        try:
            date_series = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            raise ValueError(f"Unable to parse DataFrame date index or Date column: {e}") from e

    # Normalize daily timestamps to UTC midnight as DatetimeIndex
    if isinstance(date_series, pd.Series):
        normalized = pd.DatetimeIndex(date_series.dt.normalize())
    elif isinstance(date_series, pd.DatetimeIndex):
        normalized = date_series.normalize()
    else:
        normalized = pd.DatetimeIndex(pd.to_datetime(date_series, utc=True)).normalize()

    # 1. Check for duplicate dates
    dup_mask = pd.Series(normalized, index=df.index).duplicated()
    if dup_mask.any():
        dups = pd.Series(normalized, index=df.index)[dup_mask].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"Duplicate dates found in price data: {dups}")

    # If only 1 row, continuity is trivially satisfied
    if len(normalized) <= 1:
        return pd.DatetimeIndex(normalized)

    # 2. Check strictly ascending order
    diffs = pd.Series(normalized, index=df.index).diff().dropna()
    if (diffs <= pd.Timedelta(0)).any():
        raise ValueError("Price data must be sorted chronologically in strictly ascending order.")

    # 3. Check calendar continuity: exactly 1 calendar day spacing
    expected_step = pd.Timedelta(days=1)
    gaps = diffs[diffs != expected_step]
    if not gaps.empty:
        gap_details = []
        for idx in gaps.index:
            loc = df.index.get_loc(idx)
            prev_d = normalized[loc - 1].strftime("%Y-%m-%d")
            curr_d = normalized[loc].strftime("%Y-%m-%d")
            gap_details.append(f"{prev_d} -> {curr_d}")
        raise ValueError(
            f"Calendar continuity violation: detected missing calendar days or irregular spacing in date series: {gap_details}. "
            "Consecutive daily observations (exactly 1 day apart) are required for row-based forecast targets."
        )

    return pd.DatetimeIndex(normalized)


def align_sentiment_with_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    text_column: str = "text",
    created_at_column: str = "created_at",
) -> pd.DataFrame:
    """
    Left-joins daily sentiment features onto the BTC price calendar.

    Preservation & Validation Guarantees:
        - Price dates are validated for uniqueness, strictly ascending chronological order,
          and exact 1-day calendar continuity.
        - Preserves all original price rows, price values, and column data types intact.
        - Does NOT forward-fill prices or invent missing price bars.
        - Does NOT interpolate missing prices.
        - Days with no posts have post_count = 0, counts = 0, has_posts = False,
          sentiment_weighted = 0.0 (fallback), and sentiment_mean = NaN.
          This cleanly distinguishes no-post days from days with actual neutral posts.

    Args:
        price_df: Validated OHLCV price DataFrame.
        sentiment_df: Either raw posts DataFrame or pre-aggregated daily sentiment DataFrame.
        text_column: Text column name if sentiment_df requires aggregation.
        created_at_column: Timestamp column name if sentiment_df requires aggregation.

    Returns:
        Aligned DataFrame with price columns and joined daily sentiment features.

    Raises:
        ValueError: If price_df contains duplicate dates, is unsorted, empty, or has calendar gaps.
    """
    if price_df is None or price_df.empty:
        raise ValueError("Price DataFrame is empty or None.")

    # Validate price date uniqueness, chronological order, and daily continuity
    validated_dates = validate_calendar_continuity(price_df)
    date_keys = pd.Series(validated_dates.strftime("%Y-%m-%d"), index=price_df.index)

    # Check if sentiment_df needs daily aggregation
    is_already_aggregated = "post_count" in sentiment_df.columns and "sentiment_mean" in sentiment_df.columns
    if not is_already_aggregated:
        daily_sent = aggregate_daily_sentiment(
            sentiment_df,
            text_column=text_column,
            created_at_column=created_at_column,
        )
    else:
        daily_sent = sentiment_df.copy()
        if "Date" in daily_sent.columns:
            daily_sent["Date"] = pd.to_datetime(daily_sent["Date"], utc=True).dt.strftime("%Y-%m-%d")
            daily_sent = daily_sent.set_index("Date")
        elif isinstance(daily_sent.index, pd.DatetimeIndex):
            tz_idx = daily_sent.index.tz_convert("UTC") if daily_sent.index.tz else daily_sent.index.tz_localize("UTC")
            daily_sent.index = tz_idx.strftime("%Y-%m-%d")
        else:
            daily_sent.index = daily_sent.index.astype(str)

    # Perform Left-Join on the price calendar
    result = price_df.copy()
    orig_index = price_df.index

    # Temporary join key
    result["_date_key"] = date_keys.values

    aligned = result.merge(
        daily_sent,
        how="left",
        left_on="_date_key",
        right_index=True,
    ).drop(columns=["_date_key"])

    # Restore original index
    aligned.index = orig_index

    # Fill days with no posts with explicit zero counts and false indicators
    no_post_mask = aligned["post_count"].isna()

    aligned.loc[no_post_mask, "post_count"] = 0
    aligned.loc[no_post_mask, "positive_post_count"] = 0
    aligned.loc[no_post_mask, "negative_post_count"] = 0
    aligned.loc[no_post_mask, "neutral_post_count"] = 0
    aligned.loc[no_post_mask, "has_posts"] = False
    aligned.loc[no_post_mask, "sentiment_weighted_available"] = False
    aligned.loc[no_post_mask, "sentiment_weighted"] = 0.0

    # Ensure integer counts
    int_cols = ["post_count", "positive_post_count", "negative_post_count", "neutral_post_count"]
    for c in int_cols:
        aligned[c] = aligned[c].astype(int)

    aligned["has_posts"] = aligned["has_posts"].astype(bool)
    aligned["sentiment_weighted_available"] = aligned["sentiment_weighted_available"].astype(bool)

    return aligned


def build_daily_features_pipeline(
    price_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    text_column: str = "text",
    created_at_column: str = "created_at",
    engagement_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline that scores posts, aggregates daily metrics, and aligns with prices.

    Args:
        price_df: Validated OHLCV price DataFrame.
        posts_df: Normalized X posts DataFrame.
        text_column: Name of text column in posts_df.
        created_at_column: Name of timestamp column in posts_df.
        engagement_weights: Optional custom engagement metric weights.

    Returns:
        Aligned DataFrame combining price bars with daily sentiment features.
    """
    daily_sentiment = aggregate_daily_sentiment(
        posts_df=posts_df,
        text_column=text_column,
        created_at_column=created_at_column,
        engagement_weights=engagement_weights,
    )

    aligned_features = align_sentiment_with_prices(
        price_df=price_df,
        sentiment_df=daily_sentiment,
        text_column=text_column,
        created_at_column=created_at_column,
    )

    return aligned_features


def add_forecast_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    target_column: str = "Close",
    include_nowcast: bool = False,
) -> pd.DataFrame:
    """
    Constructs multi-horizon forecasting targets across horizons n+0 through n+5.

    Continuity Validation:
        Validates that the input DataFrame has a strictly ascending, unique,
        and continuous daily calendar (spaced exactly 1 calendar day apart) before
        applying row-based shifts.

    Target Definitions & Conventions:
        - Future Horizons (h in 1..5):
            target_return_h{h} = Close.shift(-h) / Close - 1.0 (Forward return target for Ridge)
            target_{target_column}_n{h} = Close.shift(-h) (Future close price target for verification)
        - Nowcast Reference (h = 0, generated when 0 in horizons or include_nowcast=True):
            target_{target_column}_n0 = Close (Contemporaneous reference price, not a future forecast)
            target_return_h0 = 0.0 (Zero nowcast reference return)

    Temporal Leakage Prevention:
        Targets are explicitly generated by negative shifting (looking forward) and are
        strictly reserved as prediction targets; they must NEVER be included in feature
        matrices X during model training or inference.

    Args:
        df: DataFrame containing price and sentiment features, sorted chronologically.
        horizons: List of integer forecast horizons (default: FORECAST_HORIZONS = [1, 2, 3, 4, 5]).
        target_column: Column name to project forward (default: 'Close').
        include_nowcast: Whether to generate n+0 nowcast reference target (default: False).

    Returns:
        New DataFrame with target columns appended.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        KeyError: If target_column is missing from df.
        ValueError: If df is empty, contains non-positive prices, or fails calendar continuity.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    # Validate daily calendar continuity before applying row-based shifts
    validate_calendar_continuity(df)

    if (df[target_column] <= 0).any():
        raise ValueError(f"Target column '{target_column}' contains non-positive values.")

    if horizons is None:
        horizons = list(FORECAST_HORIZONS)
    else:
        horizons = list(horizons)

    if include_nowcast and 0 not in horizons:
        horizons = [0] + horizons

    res = df.copy()
    target_series = res[target_column]

    for h in horizons:
        if not isinstance(h, int) or h < 0:
            raise ValueError(f"Horizon must be a non-negative integer, got {h}")

        if h == 0:
            # Contemporaneous nowcast reference price (shift = 0)
            res[f"target_{target_column}_n0"] = target_series.copy()
            res["target_return_h0"] = pd.Series(0.0, index=res.index, dtype=float)
        else:
            # Future price target (shift = -h)
            res[f"target_{target_column}_n{h}"] = target_series.shift(-h)
            # Forward return target: Close.shift(-h) / Close - 1.0
            res[f"target_return_h{h}"] = (target_series.shift(-h) / target_series) - 1.0

    return res


def add_strategy_trade_target(
    df: pd.DataFrame,
    open_column: str = "Open",
    target_column_name: str = "target_trade_return",
) -> pd.DataFrame:
    """
    Constructs the separate trading strategy target for causal Next-Open execution:
        target_trade_return = Open.shift(-2) / Open.shift(-1) - 1.0

    Signal Timing & Execution Alignment:
        - Signal Date D (row t): Strategy generates trading signal after observing all
          features and price metrics finalized at Close[D] (23:59:59 UTC).
        - Execution Entry Date D+1 (row t+1): Position entered at Open[D+1] (00:00:00 UTC).
        - Execution Exit / Rebalance Date D+2 (row t+2): Position exited or rebalanced at Open[D+2].
        - Realized Trade Return: (Open[D+2] - Open[D+1]) / Open[D+1].

    Important Distinction:
        This target is NOT the same as the econometric close-to-close forecasting target
        (target_return_h1 = Close[D+1] / Close[D] - 1.0). The close-to-close target measures
        daily price drift between consecutive market closes, whereas target_trade_return measures
        the actual holding-period return earned by executing at the next open and exiting at the
        subsequent open.

    Leakage Prevention & Target Attachment:
        The target is attached to the signal date D (row t). Because row t's label relies on
        Open[t+1] and Open[t+2], this target must strictly NEVER be used as an input feature.
        In walk-forward cross-validation, observations whose label endpoint (t+2) meets or
        exceeds the test fold's start index must be purged from training.

    Validation Guarantees:
        - Input must be a valid pandas DataFrame.
        - Validates daily calendar continuity via validate_calendar_continuity().
        - Validates that the Open series is numeric and strictly positive (> 0).
        - Series must have at least 3 rows to provide at least 1 valid observation.
        - Exactly the last 2 rows evaluate to NaN because Open[t+1] and/or Open[t+2] are unobserved.

    Args:
        df: DataFrame containing price and feature columns.
        open_column: Name of Open price column (default: 'Open').
        target_column_name: Target column name to add (default: 'target_trade_return').

    Returns:
        New DataFrame with target_trade_return column appended.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty, has fewer than 3 observations, contains non-positive prices,
                    or fails daily calendar continuity.
        KeyError: If open_column is missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if len(df) < 3:
        raise ValueError(
            f"Insufficient history: DataFrame has {len(df)} rows, but computing "
            "strategy trade target (Open.shift(-2)/Open.shift(-1)-1) requires at least 3 rows."
        )
    if open_column not in df.columns:
        raise KeyError(f"Open price column '{open_column}' not found in DataFrame.")

    # Validate daily calendar continuity before applying row-based shifts
    validate_calendar_continuity(df)

    open_series = df[open_column]
    if not pd.api.types.is_numeric_dtype(open_series):
        raise TypeError(f"Open price column '{open_column}' must be numeric, got {open_series.dtype}.")

    if (open_series <= 0).any() or open_series.isna().any():
        raise ValueError(f"Open price column '{open_column}' contains non-positive, NaN, or non-finite values.")

    res = df.copy()
    res[target_column_name] = (res[open_column].shift(-2) / res[open_column].shift(-1)) - 1.0
    return res


def add_price_derived_features(
    df: pd.DataFrame,
    close_column: str = "Close",
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
) -> pd.DataFrame:
    """
    Computes backward-looking technical and return features available at prediction cutoff D.

    Features generated:
        - return_1d: 1-day percentage return (Close_t - Close_{t-1}) / Close_{t-1}
        - return_3d: 3-day percentage return (Close_t - Close_{t-3}) / Close_{t-3}
        - return_7d: 7-day percentage return (Close_t - Close_{t-7}) / Close_{t-7}
        - volatility_7d: 7-day rolling standard deviation of return_1d
        - high_low_ratio: High_t / Low_t
        - close_open_ratio: Close_t / Open_t

    Leakage Prevention:
        All transformations use only contemporaneous (row t) or historical (rows < t)
        data via backward shifting. Zero forward shifting is used.

    Args:
        df: DataFrame containing OHLC price data.
        close_column: Name of close price column (default: 'Close').
        open_column: Name of open price column (default: 'Open').
        high_column: Name of high price column (default: 'High').
        low_column: Name of low price column (default: 'Low').

    Returns:
        New DataFrame with price-derived features appended.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    res = df.copy()

    if close_column in res.columns:
        close_series = res[close_column]
        res["return_1d"] = close_series.pct_change(1)
        res["return_3d"] = close_series.pct_change(3)
        res["return_7d"] = close_series.pct_change(7)
        res["volatility_7d"] = res["return_1d"].rolling(window=7, min_periods=2).std()

    if high_column in res.columns and low_column in res.columns:
        res["high_low_ratio"] = res[high_column] / res[low_column]

    if close_column in res.columns and open_column in res.columns:
        res["close_open_ratio"] = res[close_column] / res[open_column]

    return res


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies and validates all permissible model input feature columns.

    Guarantees:
        - Excludes any future target columns (columns containing 'target_', 'forward_return', or 'trade_return').
        - Excludes raw price levels ('Open', 'High', 'Low', 'Close') to prevent non-stationary
          nominal price inputs and avoid contemporaneous price leakage for Ridge.
        - Excludes identifiers, URLs, raw text, and metadata ('post_id', 'url',
          'text', 'is_synthetic', 'Date', and hidden keys).
        - Returns only valid numeric price-derived and sentiment features.

    Args:
        df: DataFrame containing candidate feature columns.

    Returns:
        List of verified feature column names.
    """
    prohibited_substrings = [
        "target_",
        "forward_return",
        "trade_return",
        "post_id",
        "url",
        "text",
        "is_synthetic",
        "_utc_date",
        "_date_key",
        "_metric_richness",
        "_engagement_weight",
    ]
    prohibited_exact = {"Date", "Open", "High", "Low", "Close"}

    candidate_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str in prohibited_exact:
            continue
        if any(sub in col_str for sub in prohibited_substrings):
            continue
        # Verify column is numeric or boolean
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            candidate_cols.append(col_str)

    return candidate_cols


def prepare_modeling_dataset(
    aligned_df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    target_column: str = "Close",
    include_nowcast: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Assembles a complete modeling dataset with backward-looking features and future targets.

    Args:
        aligned_df: Aligned price and daily sentiment DataFrame.
        horizons: List of forecast horizons (default: FORECAST_HORIZONS = [1, 2, 3, 4, 5]).
        target_column: Target price column (default: 'Close').
        include_nowcast: Whether to generate n+0 nowcast reference target (default: False).

    Returns:
        Tuple of:
          - modeling_df: DataFrame with price-derived features and forecast targets.
          - feature_columns: Valid feature names for model input X (strictly stationary).
          - target_columns: Target forward return column names for model training.
    """
    if horizons is None:
        horizons = list(FORECAST_HORIZONS)

    df_with_features = add_price_derived_features(aligned_df)
    df_with_targets = add_forecast_targets(
        df_with_features,
        horizons=horizons,
        target_column=target_column,
        include_nowcast=include_nowcast,
    )

    feature_cols = get_feature_columns(df_with_targets)
    target_cols = [f"target_return_h{h}" for h in horizons if h >= 1]
    if include_nowcast or 0 in horizons:
        target_cols = ["target_return_h0"] + target_cols

    return df_with_targets, feature_cols, target_cols
