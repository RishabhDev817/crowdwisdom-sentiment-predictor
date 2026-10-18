"""
Unit tests for daily sentiment aggregation and price alignment in src/processing/features.py.

All tests operate completely offline without external APIs, model downloads,
or network connections. Test fixtures are strictly synthetic.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.processing.features import (
    DAILY_FEATURE_COLUMNS,
    DEFAULT_ENGAGEMENT_WEIGHTS,
    aggregate_daily_sentiment,
    align_sentiment_with_prices,
    build_daily_features_pipeline,
    calculate_post_engagement,
)
from src.processing.sentiment import load_synthetic_posts


@pytest.fixture
def sample_synthetic_posts() -> pd.DataFrame:
    """Creates a controlled synthetic posts DataFrame with diverse timestamps and metrics."""
    data = [
        # Day 1: 2026-10-14 (Bullish posts)
        {
            "post_id": "p_001",
            "created_at": "2026-10-14 02:00:00+00:00",
            "text": "Bitcoin is looking bullish! Strong buying momentum.",
            "like_count": 100,
            "retweet_count": 20,
            "reply_count": 5,
            "quote_count": 2,
            "is_synthetic": True,
        },
        {
            "post_id": "p_002",
            "created_at": "2026-10-14 23:59:59+00:00",  # 1 second before midnight
            "text": "Huge breakout rally underway! To the moon. #BTC",
            "like_count": 200,
            "retweet_count": 50,
            "reply_count": 10,
            "quote_count": 5,
            "is_synthetic": True,
        },
        # Day 2: 2026-10-15 (Midnight post + Crash post)
        {
            "post_id": "p_003",
            "created_at": "2026-10-15 00:00:00+00:00",  # Exact midnight
            "text": "Market opening flat, consolidation continues.",
            "like_count": 10,
            "retweet_count": 1,
            "reply_count": 0,
            "quote_count": 0,
            "is_synthetic": True,
        },
        {
            "post_id": "p_004",
            "created_at": "2026-10-15 15:30:00+00:00",
            "text": "Terrible crash! Massive panic dump and heavy losses. #rekt",
            "like_count": 500,
            "retweet_count": 150,
            "reply_count": 45,
            "quote_count": 20,
            "is_synthetic": True,
        },
        # Day 3: 2026-10-16 (Pure neutral posts)
        {
            "post_id": "p_005",
            "created_at": "2026-10-16 12:00:00+00:00",
            "text": "Bitcoin trading volume is $25B across major centralized exchanges.",
            "like_count": 30,
            "retweet_count": 2,
            "reply_count": 1,
            "quote_count": 0,
            "is_synthetic": True,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Creates a sample 5-day validated daily OHLCV price DataFrame."""
    dates = pd.date_range("2026-10-13", periods=5, freq="D", tz="UTC")
    data = {
        "Open": [62000.0, 62500.0, 64000.0, 61000.0, 61500.0],
        "High": [63000.0, 64500.0, 64200.0, 62000.0, 63000.0],
        "Low": [61500.0, 62000.0, 60500.0, 60000.0, 61200.0],
        "Close": [62500.0, 64000.0, 61000.0, 61500.0, 62800.0],
        "Volume": [1.5e9, 2.2e9, 3.1e9, 1.8e9, 1.4e9],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


# ==============================================================================
# 1. UTC Date Bucketing and Midnight Boundaries
# ==============================================================================

def test_utc_date_bucketing_midnight_boundary(sample_synthetic_posts):
    """
    Verifies that timestamps 1 second before midnight fall into Day 1,
    and timestamps at exact midnight fall into Day 2.
    """
    agg = aggregate_daily_sentiment(sample_synthetic_posts)

    assert "2026-10-14" in agg.index
    assert "2026-10-15" in agg.index
    assert "2026-10-16" in agg.index

    # Day 1: p_001 (02:00) and p_002 (23:59:59)
    assert agg.loc["2026-10-14", "post_count"] == 2

    # Day 2: p_003 (00:00:00) and p_004 (15:30:00)
    assert agg.loc["2026-10-15", "post_count"] == 2

    # Day 3: p_005 (12:00:00)
    assert agg.loc["2026-10-16", "post_count"] == 1


def test_utc_date_bucketing_different_timezones():
    """Verifies that non-UTC timestamps are properly converted to their UTC calendar day."""
    posts = [
        # 2026-10-14 20:00:00 in UTC-5 (EDT) is 2026-10-15 01:00:00 UTC -> Day 2026-10-15
        {
            "post_id": "tz_1",
            "created_at": "2026-10-14 20:00:00-05:00",
            "text": "Bullish Bitcoin rally!",
            "like_count": 10,
        },
        # 2026-10-15 03:00:00 in UTC+9 (JST) is 2026-10-14 18:00:00 UTC -> Day 2026-10-14
        {
            "post_id": "tz_2",
            "created_at": "2026-10-15 03:00:00+09:00",
            "text": "Bitcoin market dump!",
            "like_count": 15,
        },
    ]
    df = pd.DataFrame(posts)
    agg = aggregate_daily_sentiment(df)

    assert agg.loc["2026-10-14", "post_count"] == 1
    assert agg.loc["2026-10-15", "post_count"] == 1
    assert agg.loc["2026-10-14", "negative_post_count"] == 1
    assert agg.loc["2026-10-15", "positive_post_count"] == 1


# ==============================================================================
# 2. Daily Aggregates: Mean, Polarity Counts, and Weighted Scores
# ==============================================================================

def test_daily_sentiment_aggregates_polarity_and_means(sample_synthetic_posts):
    """Verifies calculation of daily post counts, polarity splits, and means."""
    agg = aggregate_daily_sentiment(sample_synthetic_posts)

    # Day 1: Both posts are bullish
    day1 = agg.loc["2026-10-14"]
    assert day1["post_count"] == 2
    assert day1["positive_post_count"] == 2
    assert day1["negative_post_count"] == 0
    assert day1["neutral_post_count"] == 0
    assert day1["sentiment_mean"] > 0.3
    assert day1["sentiment_weighted"] > 0.3
    assert bool(day1["sentiment_weighted_available"]) is True
    assert bool(day1["has_posts"]) is True

    # Day 2: One neutral post (p_003) and one heavy crash post (p_004)
    day2 = agg.loc["2026-10-15"]
    assert day2["post_count"] == 2
    assert day2["negative_post_count"] == 1
    assert day2["neutral_post_count"] == 1
    assert day2["positive_post_count"] == 0
    assert day2["sentiment_mean"] < 0.0
    # Weighted sentiment is heavily driven by p_004 (500 likes vs 10 likes)
    assert day2["sentiment_weighted"] < day2["sentiment_mean"]

    # Day 3: Pure neutral post
    day3 = agg.loc["2026-10-16"]
    assert day3["post_count"] == 1
    assert day3["neutral_post_count"] == 1
    assert day3["positive_post_count"] == 0
    assert day3["negative_post_count"] == 0
    assert -0.05 <= day3["sentiment_mean"] <= 0.05


# ==============================================================================
# 3. Missing Engagement Metrics and Zero-Denominator Fallback
# ==============================================================================

def test_calculate_post_engagement_preserves_nan():
    """Verifies that missing engagement metrics are not treated as real zeros."""
    data = [
        {"like_count": 10, "retweet_count": None},  # Partial metric -> weight 10.0
        {"like_count": None, "retweet_count": None},  # All missing -> NaN
        {"like_count": 0, "retweet_count": 0},  # Explicit zero -> 0.0
    ]
    df = pd.DataFrame(data)
    weights = {"like_count": 1.0, "retweet_count": 2.0}
    eng = calculate_post_engagement(df, weights=weights)

    assert eng.iloc[0] == 10.0
    assert pd.isna(eng.iloc[1])
    assert eng.iloc[2] == 0.0


def test_engagement_weighted_zero_denominator_fallback():
    """
    Verifies that when engagement is all missing or total engagement is 0,
    the weighted score falls back to neutral (0.0) and available flag is False.
    """
    # Case A: Engagement metrics are completely missing (all None)
    posts_missing_eng = pd.DataFrame([
        {
            "post_id": "m1",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Bullish Bitcoin rally!",
            "like_count": None,
            "retweet_count": None,
        },
        {
            "post_id": "m2",
            "created_at": "2026-10-14 12:00:00+00:00",
            "text": "Bitcoin ATH imminent!",
            "like_count": None,
            "retweet_count": None,
        },
    ])
    agg_missing = aggregate_daily_sentiment(posts_missing_eng)
    assert bool(agg_missing.loc["2026-10-14", "sentiment_weighted_available"]) is False
    assert agg_missing.loc["2026-10-14", "sentiment_weighted"] == 0.0
    assert agg_missing.loc["2026-10-14", "sentiment_mean"] > 0.0

    # Case B: Engagement metrics are explicitly 0
    posts_zero_eng = pd.DataFrame([
        {
            "post_id": "z1",
            "created_at": "2026-10-15 10:00:00+00:00",
            "text": "Bearish crash dump!",
            "like_count": 0,
            "retweet_count": 0,
        }
    ])
    agg_zero = aggregate_daily_sentiment(posts_zero_eng)
    assert bool(agg_zero.loc["2026-10-15", "sentiment_weighted_available"]) is False
    assert agg_zero.loc["2026-10-15", "sentiment_weighted"] == 0.0
    assert agg_zero.loc["2026-10-15", "sentiment_mean"] < 0.0


def test_engagement_weighted_nan_sentiment_does_not_dilute_denominator():
    """
    Regression test:
    Post A: sentiment_score = NaN, engagement weight = 14 (10 likes, 2 retweets)
    Post B: sentiment_score = -0.6, engagement weight = 5 (5 likes, 0 retweets)
    Verifies that Post A does not dilute the denominator:
      Expected sentiment_weighted: -0.6 (not -0.1579).
    """
    posts = pd.DataFrame([
        {
            "post_id": "post_a",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Bitcoin is bullish",
            "sentiment_score": np.nan,
            "like_count": 10,
            "retweet_count": 2,
            "reply_count": 0,
            "quote_count": 0,
        },
        {
            "post_id": "post_b",
            "created_at": "2026-10-14 14:00:00+00:00",
            "text": "Bitcoin is dumping",
            "sentiment_score": -0.6,
            "like_count": 5,
            "retweet_count": 0,
            "reply_count": 0,
            "quote_count": 0,
        },
    ])
    agg = aggregate_daily_sentiment(posts)

    # Denominator should be 5.0 (Post B only), NOT 19.0 (14 + 5)
    assert agg.loc["2026-10-14", "sentiment_weighted"] == -0.6
    assert bool(agg.loc["2026-10-14", "sentiment_weighted_available"]) is True
    assert agg.loc["2026-10-14", "sentiment_mean"] == -0.6


def test_engagement_weighted_all_posts_nan_sentiment_fallback():
    """
    Verifies that when all posts with engagement data have NaN sentiment scores,
    the weighted score safely falls back to neutral (0.0) with available flag False.
    """
    posts = pd.DataFrame([
        {
            "post_id": "nan_1",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Post 1",
            "sentiment_score": np.nan,
            "like_count": 50,
            "retweet_count": 10,
        },
        {
            "post_id": "nan_2",
            "created_at": "2026-10-14 15:00:00+00:00",
            "text": "Post 2",
            "sentiment_score": np.nan,
            "like_count": 100,
            "retweet_count": 20,
        },
    ])
    agg = aggregate_daily_sentiment(posts)

    assert bool(agg.loc["2026-10-14", "sentiment_weighted_available"]) is False
    assert agg.loc["2026-10-14", "sentiment_weighted"] == 0.0
    assert pd.isna(agg.loc["2026-10-14", "sentiment_mean"])


# ==============================================================================
# 4. Deterministic Deduplication
# ==============================================================================

def test_aggregate_daily_sentiment_deduplication():
    """Verifies that duplicate post IDs are deterministically resolved by highest metric richness."""
    duplicate_posts = pd.DataFrame([
        {
            "post_id": "dup_01",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Bitcoin is bullish",
            "like_count": 10,
            "retweet_count": None,  # Metric richness = 1
        },
        {
            "post_id": "dup_01",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Bitcoin is bullish",
            "like_count": 10,
            "retweet_count": 5,  # Metric richness = 2 (richer record)
        },
    ])
    agg = aggregate_daily_sentiment(duplicate_posts, deduplicate=True)
    assert agg.loc["2026-10-14", "post_count"] == 1


# ==============================================================================
# 5. Price Alignment & Row/Order Preservation
# ==============================================================================

def test_align_sentiment_with_prices_preservation(sample_price_df, sample_synthetic_posts):
    """
    Verifies that align_sentiment_with_prices:
      - Preserves all original price rows, order, and index.
      - Never forward-fills or interpolates prices.
      - Keeps original price columns and values exact.
    """
    aligned = align_sentiment_with_prices(sample_price_df, sample_synthetic_posts)

    # 1. Exact row count, order, and index preservation
    assert len(aligned) == len(sample_price_df)
    pd.testing.assert_index_equal(aligned.index, sample_price_df.index)

    # 2. Price columns and values bitwise preserved
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        pd.testing.assert_series_equal(aligned[col], sample_price_df[col])


def test_align_sentiment_distinguishes_no_post_from_neutral(sample_price_df, sample_synthetic_posts):
    """
    Verifies that days with no posts are explicitly distinguishable from
    days with actual neutral posts:
      - No-post day: post_count = 0, has_posts = False, sentiment_mean = NaN
      - Neutral post day: post_count > 0, has_posts = True, neutral_post_count > 0, sentiment_mean = 0.0
    """
    aligned = align_sentiment_with_prices(sample_price_df, sample_synthetic_posts)

    # Price dates: 2026-10-13 through 2026-10-17
    # 2026-10-13 has NO posts
    row_13 = aligned.loc["2026-10-13 00:00:00+00:00"]
    assert row_13["post_count"] == 0
    assert bool(row_13["has_posts"]) is False
    assert bool(row_13["sentiment_weighted_available"]) is False
    assert pd.isna(row_13["sentiment_mean"])
    assert row_13["sentiment_weighted"] == 0.0

    # 2026-10-16 has 1 ACTUAL NEUTRAL post (p_005)
    row_16 = aligned.loc["2026-10-16 00:00:00+00:00"]
    assert row_16["post_count"] == 1
    assert bool(row_16["has_posts"]) is True
    assert row_16["neutral_post_count"] == 1
    assert not pd.isna(row_16["sentiment_mean"])
    assert -0.05 <= row_16["sentiment_mean"] <= 0.05

    # 2026-10-17 has NO posts
    row_17 = aligned.loc["2026-10-17 00:00:00+00:00"]
    assert row_17["post_count"] == 0
    assert bool(row_17["has_posts"]) is False
    assert pd.isna(row_17["sentiment_mean"])


# ==============================================================================
# 6. Date Validation on Price Data
# ==============================================================================

def test_align_sentiment_rejects_duplicate_price_dates(sample_price_df, sample_synthetic_posts):
    """Verifies ValueError when price data contains duplicate dates."""
    dup_price_df = pd.concat([sample_price_df, sample_price_df.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate dates found in price data"):
        align_sentiment_with_prices(dup_price_df, sample_synthetic_posts)


def test_align_sentiment_rejects_unsorted_price_dates(sample_price_df, sample_synthetic_posts):
    """Verifies ValueError when price data is not sorted chronologically ascending."""
    unsorted_price_df = sample_price_df.iloc[::-1].copy()
    with pytest.raises(ValueError, match="must be sorted chronologically"):
        align_sentiment_with_prices(unsorted_price_df, sample_synthetic_posts)


def test_align_sentiment_rejects_empty_price_df(sample_synthetic_posts):
    """Verifies ValueError when price data is empty or None."""
    with pytest.raises(ValueError, match="empty or None"):
        align_sentiment_with_prices(pd.DataFrame(), sample_synthetic_posts)


def test_align_sentiment_rejects_missing_calendar_dates(sample_price_df, sample_synthetic_posts):
    """Verifies ValueError when price data has calendar gaps (missing days)."""
    # Drop middle date (2026-10-15) creating a gap between 2026-10-14 and 2026-10-16
    gapped_price_df = sample_price_df.drop(sample_price_df.index[2])
    with pytest.raises(ValueError, match="Calendar continuity violation"):
        align_sentiment_with_prices(gapped_price_df, sample_synthetic_posts)


# ==============================================================================
# 7. Look-Ahead Bias Prevention Tests
# ==============================================================================

def test_look_ahead_bias_prevention_later_posts_do_not_affect_earlier_aggregates():
    """
    Strict look-ahead bias test:
    Verifies that adding posts on Day 2 (or Day N+1) has zero impact on
    Day 1's aggregated feature values.
    """
    day1_posts = pd.DataFrame([
        {
            "post_id": "d1_01",
            "created_at": "2026-10-14 10:00:00+00:00",
            "text": "Bitcoin is slightly bullish.",
            "like_count": 50,
        },
        {
            "post_id": "d1_02",
            "created_at": "2026-10-14 22:30:00+00:00",
            "text": "Bitcoin holding support nicely.",
            "like_count": 25,
        },
    ])

    day1_aggregate = aggregate_daily_sentiment(day1_posts)

    # Now simulate future posts on Day 2 with extreme panic crash sentiment
    future_posts = pd.DataFrame([
        {
            "post_id": f"d2_{i}",
            "created_at": "2026-10-15 08:00:00+00:00",
            "text": "TERRIBLE CRASH! MASSIVE DUMP AND LIQUIDATION! #rekt",
            "like_count": 1000,
        }
        for i in range(50)
    ])

    combined_posts = pd.concat([day1_posts, future_posts], ignore_index=True)
    combined_aggregate = aggregate_daily_sentiment(combined_posts)

    # Verify Day 1 values in combined aggregate are bitwise identical to Day 1 alone
    for feature in DAILY_FEATURE_COLUMNS:
        assert combined_aggregate.loc["2026-10-14", feature] == day1_aggregate.loc["2026-10-14", feature], (
            f"Feature {feature} on Day 1 was altered by future Day 2 posts! Look-ahead leak detected."
        )


def test_look_ahead_bias_midnight_transition_isolation():
    """
    Verifies that a post at 2026-10-15 00:00:00+00:00 (exact start of Day 2)
    does not leak into Day 1 (2026-10-14).
    """
    posts = pd.DataFrame([
        {
            "post_id": "d1_last",
            "created_at": "2026-10-14 23:59:59+00:00",
            "text": "Bitcoin bullish close.",
            "like_count": 10,
        },
        {
            "post_id": "d2_first",
            "created_at": "2026-10-15 00:00:00+00:00",
            "text": "Catastrophic crash at midnight!",
            "like_count": 500,
        },
    ])
    agg = aggregate_daily_sentiment(posts)

    assert agg.loc["2026-10-14", "post_count"] == 1
    assert agg.loc["2026-10-14", "positive_post_count"] == 1
    assert agg.loc["2026-10-14", "negative_post_count"] == 0
    assert agg.loc["2026-10-14", "sentiment_mean"] > 0.0

    assert agg.loc["2026-10-15", "post_count"] == 1
    assert agg.loc["2026-10-15", "negative_post_count"] == 1
    assert agg.loc["2026-10-15", "positive_post_count"] == 0
    assert agg.loc["2026-10-15", "sentiment_mean"] < 0.0


# ==============================================================================
# 8. Empty and Malformed Input Safety
# ==============================================================================

def test_aggregate_daily_sentiment_empty_input():
    """Verifies that empty DataFrame returns empty aggregate with expected columns."""
    empty_df = pd.DataFrame(columns=["post_id", "text", "created_at"])
    agg = aggregate_daily_sentiment(empty_df)

    assert agg.empty
    assert list(agg.columns) == DAILY_FEATURE_COLUMNS


def test_aggregate_daily_sentiment_malformed_timestamps():
    """Verifies that malformed timestamps are safely rejected or handled without crashing."""
    malformed = pd.DataFrame([
        {
            "post_id": "bad_ts_1",
            "created_at": "not-a-valid-timestamp",
            "text": "Bitcoin is bullish",
        },
        {
            "post_id": "good_ts_2",
            "created_at": "2026-10-14 12:00:00+00:00",
            "text": "Bitcoin is bullish",
        },
    ])
    agg = aggregate_daily_sentiment(malformed)

    # Valid record preserved, invalid dropped
    assert len(agg) == 1
    assert "2026-10-14" in agg.index


# ==============================================================================
# 9. Synthetic Fixture End-to-End Pipeline Integration
# ==============================================================================

def test_build_daily_features_pipeline_integration(sample_price_df):
    """
    Tests end-to-end scoring, daily aggregation, and price alignment using
    the synthetic fixture in tests/fixtures/synthetic_x_posts.json.
    """
    fixture_df, meta = load_synthetic_posts("tests/fixtures/synthetic_x_posts.json")
    assert meta["is_synthetic"] is True

    # Run full feature pipeline
    aligned = build_daily_features_pipeline(
        price_df=sample_price_df,
        posts_df=fixture_df,
    )

    # Validations
    assert len(aligned) == len(sample_price_df)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in aligned.columns
    for feat in DAILY_FEATURE_COLUMNS:
        assert feat in aligned.columns

    assert not aligned["post_count"].isna().any()
    assert (aligned["post_count"] >= 0).all()
