"""
Unit tests for deterministic offline sentiment processing in src/processing/sentiment.py.

All tests operate completely offline without external APIs, model downloads,
or network connections. Test fixtures are strictly synthetic and verified.
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import pytest

from src.processing.sentiment import (
    classify_sentiment,
    compute_sentiment_score,
    compute_subjectivity_score,
    load_synthetic_posts,
    score_posts,
    score_text,
)


@pytest.fixture
def synthetic_fixture_path() -> Path:
    """Returns path to the synthetic X posts fixture file."""
    return Path("tests/fixtures/synthetic_x_posts.json")


@pytest.fixture
def sample_normalized_posts_df() -> pd.DataFrame:
    """Constructs a small DataFrame following the normalized X-post schema."""
    data = [
        {
            "post_id": "test_001",
            "created_at": "2026-10-14 12:30:00+00:00",
            "text": "Bitcoin is breaking out! Very bullish rally towards ATH. #BTC",
            "url": "https://twitter.com/user1/status/test_001",
            "like_count": 100,
            "retweet_count": 25,
            "reply_count": 5,
            "quote_count": 2,
            "view_count": 3000,
            "author_followers": 15000,
        },
        {
            "post_id": "test_002",
            "created_at": "2026-10-15 08:00:00+00:00",
            "text": "Terrible market dump. Major crash and catastrophic losses for long traders.",
            "url": "https://twitter.com/user2/status/test_002",
            "like_count": 50,
            "retweet_count": 10,
            "reply_count": 8,
            "quote_count": 1,
            "view_count": None,  # Missing optional metric
            "author_followers": None,  # Missing optional metric
        },
        {
            "post_id": "test_003",
            "created_at": "2026-10-16 00:00:00+00:00",
            "text": "Bitcoin trading volume is $25B over the last 24 hours.",
            "url": "https://twitter.com/user3/status/test_003",
            "like_count": 12,
            "retweet_count": 1,
            "reply_count": 0,
            "quote_count": 0,
            "view_count": 800,
            "author_followers": 2200,
        },
        {
            "post_id": "test_004",
            "created_at": "2026-10-17 09:15:00+00:00",
            "text": "",  # Empty text edge case
            "url": "https://twitter.com/user4/status/test_004",
            "like_count": 0,
            "retweet_count": 0,
            "reply_count": 0,
            "quote_count": 0,
            "view_count": 100,
            "author_followers": 500,
        },
    ]
    return pd.DataFrame(data)


# ==============================================================================
# 1. Positivity, Negativity, and Neutrality Tests
# ==============================================================================

def test_positive_sentiment_scoring():
    """Verifies that positive crypto posts receive a positive score."""
    pos_texts = [
        "Bitcoin is looking very bullish! Massive rally underway. #BTC",
        "BTC breaking out to new ATH, strong accumulation and buying momentum!",
        "To the moon! Great adoption and huge profits for BTC holders.",
    ]
    for text in pos_texts:
        score = compute_sentiment_score(text)
        assert score > 0.0, f"Expected positive score for '{text}', got {score}"
        label = classify_sentiment(score)
        assert label == "positive"


def test_negative_sentiment_scoring():
    """Verifies that negative crypto posts receive a negative score."""
    neg_texts = [
        "Massive Bitcoin crash! Panicking sellers and liquidation cascade.",
        "Bearish breakdown on BTC. Horrible losses and terrible dump. #rekt",
        "Scam project, complete rug pull and insolvent exchange.",
    ]
    for text in neg_texts:
        score = compute_sentiment_score(text)
        assert score < 0.0, f"Expected negative score for '{text}', got {score}"
        label = classify_sentiment(score)
        assert label == "negative"


def test_neutral_sentiment_scoring():
    """Verifies that factual, informational, or question posts yield neutral scores."""
    neutral_texts = [
        "Bitcoin 24-hour volume is $32B across centralized exchanges.",
        "The Federal Reserve meeting is scheduled for Wednesday at 2pm ET.",
        "BTC mining difficulty adjusted at block 840,000.",
    ]
    for text in neutral_texts:
        score = compute_sentiment_score(text)
        # Neutral threshold [-0.05, 0.05]
        assert -0.05 <= score <= 0.05, f"Expected neutral score near 0 for '{text}', got {score}"
        label = classify_sentiment(score)
        assert label == "neutral"


# ==============================================================================
# 2. Case Insensitivity & Mixed Case
# ==============================================================================

def test_mixed_case_invariance():
    """Verifies that upper, lower, and mixed case yield identical sentiment scores."""
    lower = "bitcoin is bullish and rallying to the moon"
    upper = "BITCOIN IS BULLISH AND RALLYING TO THE MOON"
    title = "Bitcoin Is Bullish And Rallying To The Moon"
    mixed = "BiTcOiN is BuLLiSh and RaLLyInG To ThE MoOn"

    s_lower = compute_sentiment_score(lower)
    s_upper = compute_sentiment_score(upper)
    s_title = compute_sentiment_score(title)
    s_mixed = compute_sentiment_score(mixed)

    assert s_lower == s_upper == s_title == s_mixed
    assert s_lower > 0.5


# ==============================================================================
# 3. Negation Handling
# ==============================================================================

def test_negation_inversion_positive_to_negative():
    """Verifies that negation words flip positive sentiment to negative."""
    pos_score = compute_sentiment_score("Bitcoin is bullish today")
    neg_score = compute_sentiment_score("Bitcoin is not bullish today")

    assert pos_score > 0.0
    assert neg_score < 0.0, f"Expected negated positive to be < 0, got {neg_score}"


def test_negation_inversion_negative_to_neutral_or_positive():
    """Verifies that negation of negative words inverts the negative valence."""
    crash_score = compute_sentiment_score("Bitcoin is crashing")
    not_crash_score = compute_sentiment_score("Bitcoin is not crashing")

    assert crash_score < 0.0
    assert not_crash_score >= 0.0, f"Expected 'not crashing' to be >= 0, got {not_crash_score}"


def test_negation_window_limit():
    """Verifies that negation does not affect words far outside its sliding window."""
    # Negation affects up to 3 tokens; 'rally' is 5 tokens away
    text = "Not bad at all, this is a huge rally"
    score = compute_sentiment_score(text)
    assert score > 0.0


def test_negation_regression_targeted():
    """
    Targeted regression tests for negation and non-negating idioms:
      - 'not only bullish' should remain positive.
      - 'not just bad' should remain negative.
      - 'not bullish' should remain negative.
      - 'not crashing' should remain positive.
      - "isn't bullish" should remain negative.
    """
    # 1. 'not only bullish'
    score_not_only = compute_sentiment_score("not only bullish")
    assert score_not_only > 0.0, f"Expected 'not only bullish' to be positive, got {score_not_only}"
    assert classify_sentiment(score_not_only) == "positive"

    # 2. 'not just bad'
    score_not_just = compute_sentiment_score("not just bad")
    assert score_not_just < 0.0, f"Expected 'not just bad' to be negative, got {score_not_just}"
    assert classify_sentiment(score_not_just) == "negative"

    # 3. 'not bullish'
    score_not_bullish = compute_sentiment_score("not bullish")
    assert score_not_bullish < 0.0, f"Expected 'not bullish' to be negative, got {score_not_bullish}"
    assert classify_sentiment(score_not_bullish) == "negative"

    # 4. 'not crashing'
    score_not_crashing = compute_sentiment_score("not crashing")
    assert score_not_crashing > 0.0, f"Expected 'not crashing' to be positive, got {score_not_crashing}"
    assert classify_sentiment(score_not_crashing) == "positive"

    # 5. "isn't bullish"
    score_isnt_bullish = compute_sentiment_score("isn't bullish")
    assert score_isnt_bullish < 0.0, f"Expected \"isn't bullish\" to be negative, got {score_isnt_bullish}"
    assert classify_sentiment(score_isnt_bullish) == "negative"


# ==============================================================================
# 4. Intensifiers and Diminishers
# ==============================================================================

def test_intensifier_boosts_magnitude():
    """Verifies that intensifiers ('extremely', 'very') increase score magnitude."""
    base_score = compute_sentiment_score("Bitcoin is bullish")
    boosted_score = compute_sentiment_score("Bitcoin is extremely bullish")

    assert boosted_score > base_score


def test_diminisher_reduces_magnitude():
    """Verifies that diminishers ('slightly') decrease score magnitude."""
    base_score = compute_sentiment_score("Bitcoin is bullish")
    diminished_score = compute_sentiment_score("Bitcoin is slightly bullish")

    assert 0.0 < diminished_score < base_score


# ==============================================================================
# 5. Missing, Empty, and Malformed Text Safety
# ==============================================================================

@pytest.mark.parametrize(
    "empty_input",
    [
        "",
        "   ",
        "\t\n  \r",
        None,
    ],
)
def test_missing_and_empty_text_safety(empty_input):
    """Verifies that missing, empty, or whitespace-only texts yield safe 0.0 values."""
    res = score_text(empty_input)
    assert res["sentiment_score"] == 0.0
    assert res["subjectivity_score"] == 0.0
    assert res["sentiment_label"] == "neutral"


# ==============================================================================
# 6. Score Bounds and Non-Linear Saturation
# ==============================================================================

def test_score_bounds_extreme_inputs():
    """Verifies that sentiment_score is strictly in [-1.0, 1.0] even with extreme inputs."""
    extreme_pos = "BULLISH MOON ATH RALLY PUMP BREAKOUT SURGE PROFIT WIN GAIN " * 20
    extreme_neg = "CRASH DUMP BEARISH REKT PANIC LIQUIDATION COLLAPSE SCAM BLEED " * 20

    pos_score = compute_sentiment_score(extreme_pos)
    neg_score = compute_sentiment_score(extreme_neg)

    assert -1.0 <= pos_score <= 1.0
    assert -1.0 <= neg_score <= 1.0
    assert pos_score > 0.95
    assert neg_score < -0.95


# ==============================================================================
# 7. Determinism
# ==============================================================================

def test_score_determinism():
    """Verifies that repeatedly scoring the same text produces identical output."""
    sample = "Bitcoin experiencing solid accumulation with strong support holding at $60,000."
    first_run = [score_text(sample) for _ in range(10)]

    for item in first_run:
        assert item == first_run[0]


# ==============================================================================
# 8. Subjectivity Heuristic Bounds and Definition
# ==============================================================================

def test_subjectivity_bounds_and_behavior():
    """Verifies that subjectivity_score is bounded in [0.0, 1.0] and behaves predictably."""
    factual = "Bitcoin block height is 850,000 and total supply is 19.7 million."
    opinionated = "I think Bitcoin is incredibly bullish and believe it will moon soon, imo!"

    subj_factual = compute_subjectivity_score(factual)
    subj_opinion = compute_subjectivity_score(opinionated)

    assert 0.0 <= subj_factual <= 1.0
    assert 0.0 <= subj_opinion <= 1.0
    assert subj_opinion > subj_factual
    assert subj_factual == 0.0  # Factual statement has no opinion markers


# ==============================================================================
# 9. DataFrame Scoring and Schema Preservation
# ==============================================================================

def test_score_posts_dataframe_preservation(sample_normalized_posts_df):
    """
    Verifies that score_posts keeps all original columns, rows, order,
    and null metrics intact while adding sentiment columns.
    """
    original_cols = list(sample_normalized_posts_df.columns)
    original_len = len(sample_normalized_posts_df)
    original_ids = list(sample_normalized_posts_df["post_id"])

    scored_df = score_posts(sample_normalized_posts_df)

    # 1. Row count and order preserved
    assert len(scored_df) == original_len
    assert list(scored_df["post_id"]) == original_ids

    # 2. All original columns are present and unchanged
    for col in original_cols:
        assert col in scored_df.columns
        # Engagement metric values preserved
        pd.testing.assert_series_equal(
            scored_df[col],
            sample_normalized_posts_df[col],
            check_names=True,
        )

    # 3. New sentiment columns exist and are typed properly
    assert "sentiment_score" in scored_df.columns
    assert "subjectivity_score" in scored_df.columns
    assert "sentiment_label" in scored_df.columns

    # 4. Scores fall within valid ranges
    assert (scored_df["sentiment_score"] >= -1.0).all()
    assert (scored_df["sentiment_score"] <= 1.0).all()
    assert (scored_df["subjectivity_score"] >= 0.0).all()
    assert (scored_df["subjectivity_score"] <= 1.0).all()

    # 5. Null metrics remain null, NOT converted to 0
    row_2 = scored_df[scored_df["post_id"] == "test_002"].iloc[0]
    assert pd.isna(row_2["view_count"])
    assert pd.isna(row_2["author_followers"])

    # 6. Empty text handled safely
    row_4 = scored_df[scored_df["post_id"] == "test_004"].iloc[0]
    assert row_4["sentiment_score"] == 0.0
    assert row_4["sentiment_label"] == "neutral"


def test_score_posts_empty_dataframe():
    """Verifies that an empty DataFrame is handled gracefully with expected columns."""
    empty_df = pd.DataFrame(columns=["post_id", "text"])
    scored_empty = score_posts(empty_df)

    assert scored_empty.empty
    assert "sentiment_score" in scored_empty.columns
    assert "subjectivity_score" in scored_empty.columns
    assert "sentiment_label" in scored_empty.columns


def test_score_posts_invalid_inputs():
    """Verifies that score_posts raises appropriate errors on invalid inputs."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        score_posts(["not", "a", "dataframe"])  # type: ignore

    df_missing_text = pd.DataFrame({"post_id": ["1", "2"]})
    with pytest.raises(KeyError, match="Required text column 'text' not found"):
        score_posts(df_missing_text)


# ==============================================================================
# 10. Synthetic Fixture Integration Tests
# ==============================================================================

def test_load_synthetic_fixture_metadata(synthetic_fixture_path):
    """Verifies that the synthetic fixture exists, is valid JSON, and has is_synthetic: true."""
    assert synthetic_fixture_path.is_file()
    df, meta = load_synthetic_posts(synthetic_fixture_path)

    assert meta.get("is_synthetic") is True
    assert len(df) > 0
    assert "text" in df.columns
    assert "post_id" in df.columns
    assert "created_at" in df.columns
    assert "is_synthetic" in df.columns
    assert df["is_synthetic"].all() is True or df["is_synthetic"].all() == 1


def test_score_all_synthetic_posts(synthetic_fixture_path):
    """Verifies end-to-end scoring across all records in the synthetic fixture."""
    df, meta = load_synthetic_posts(synthetic_fixture_path)
    scored = score_posts(df)

    assert len(scored) == len(df)
    assert not scored["sentiment_score"].isna().any()
    assert (scored["sentiment_score"] >= -1.0).all()
    assert (scored["sentiment_score"] <= 1.0).all()
    assert (scored["subjectivity_score"] >= 0.0).all()
    assert (scored["subjectivity_score"] <= 1.0).all()

    # Specific fixture row assertions
    # syn_post_001 is bullish
    row_001 = scored[scored["post_id"] == "syn_post_001"].iloc[0]
    assert row_001["sentiment_score"] > 0.5
    assert row_001["sentiment_label"] == "positive"

    # syn_post_002 is crash / panic
    row_002 = scored[scored["post_id"] == "syn_post_002"].iloc[0]
    assert row_002["sentiment_score"] < -0.5
    assert row_002["sentiment_label"] == "negative"

    # syn_post_003 is neutral close
    row_003 = scored[scored["post_id"] == "syn_post_003"].iloc[0]
    assert -0.05 <= row_003["sentiment_score"] <= 0.05
    assert row_003["sentiment_label"] == "neutral"

    # syn_post_006 is negated bullish
    row_006 = scored[scored["post_id"] == "syn_post_006"].iloc[0]
    assert row_006["sentiment_score"] < 0.0

    # syn_post_007 is negated crash
    row_007 = scored[scored["post_id"] == "syn_post_007"].iloc[0]
    assert row_007["sentiment_score"] >= 0.0

    # syn_post_009 is empty text
    row_009 = scored[scored["post_id"] == "syn_post_009"].iloc[0]
    assert row_009["sentiment_score"] == 0.0
    assert row_009["sentiment_label"] == "neutral"

    # syn_post_010 is whitespace-only text
    row_010 = scored[scored["post_id"] == "syn_post_010"].iloc[0]
    assert row_010["sentiment_score"] == 0.0
    assert row_010["sentiment_label"] == "neutral"


def test_reject_fixture_without_synthetic_flag(tmp_path):
    """Verifies that loader rejects fixtures lacking the is_synthetic: true flag."""
    bad_fixture = tmp_path / "fake_posts.json"
    bad_fixture.write_text(json.dumps({"_metadata": {"is_synthetic": False}, "posts": []}))

    with pytest.raises(ValueError, match="not flagged with 'is_synthetic: true'"):
        load_synthetic_posts(bad_fixture)
