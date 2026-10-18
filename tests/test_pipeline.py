"""
Offline tests for Phase 5.0 integrated pipeline orchestration.

Verifies:
- validate_and_load_x_posts strictly enforces data integrity:
  * Missing file returns (None, err_msg).
  * Empty file or empty dataset returns (None, err_msg).
  * Known dummy/placeholder files (e.g. 'BTC test') return (None, err_msg).
  * Synthetic fixtures return (None, err_msg) when allow_synthetic=False.
  * Synthetic fixtures return (DataFrame, None) when allow_synthetic=True.
  * Fewer than minimum genuine observations return (None, err_msg) when allow_synthetic=False.
  * Real collected posts pass validation with allow_synthetic=False.
- build_integrated_pipeline_dataset:
  * Preserves exact price calendar and historical daily rows.
  * Aggregates daily sentiment features from scored posts without look-ahead.
  * Quarantines features: no target, future price, raw OHLCV, or post metadata in feature set.
  * Handles days with missing sentiment cleanly.
- run_pipeline execution:
  * mode='price_only' runs the validated baseline.
  * mode='sentiment' halts with descriptive block_reason when data is missing/placeholder.
  * mode='auto' reports sentiment blocked when only placeholder posts exist.
  * mode='auto' completes both tracks when real posts are available.
  * mode='sentiment' with allow_synthetic=True completes end-to-end walk-forward backtest.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation.run_evaluation import (
    MIN_REAL_POSTS_THRESHOLD,
    build_integrated_pipeline_dataset,
    load_and_validate_btc_prices,
    run_pipeline,
    validate_and_load_x_posts,
)


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_fixture_path(repo_root):
    return repo_root / "tests" / "fixtures" / "synthetic_x_posts.json"


@pytest.fixture
def dummy_placeholder_csv_path(tmp_path):
    csv_file = tmp_path / "dummy_posts.csv"
    csv_file.write_text(
        "post_id,created_at,text,url,like_count,retweet_count,reply_count,quote_count,view_count,author_followers\n"
        "1,2026-10-18 00:00:00+00:00,BTC test,,,,,,,\n"
    )
    return csv_file


@pytest.fixture
def real_collected_csv_path(repo_root):
    return repo_root / "data" / "raw" / "x_posts_btc_normalized.csv"


@pytest.fixture
def price_path(repo_root):
    return repo_root / "data" / "raw" / "btc_usd_daily_raw.csv"


# ============================================================================
# 1. Validation & Loading of X Posts
# ============================================================================

def test_validate_and_load_x_posts_file_not_found(tmp_path):
    """Missing file should return None with descriptive error message."""
    missing = tmp_path / "non_existent.csv"
    df, err = validate_and_load_x_posts(missing)
    assert df is None
    assert "not found" in err


def test_validate_and_load_x_posts_empty_file(tmp_path):
    """Empty JSON or CSV file should return None with descriptive error."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    df, err = validate_and_load_x_posts(empty_csv)
    assert df is None
    assert "empty" in err.lower() or "failed" in err.lower()

    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]")
    df, err = validate_and_load_x_posts(empty_json)
    assert df is None
    assert "empty" in err.lower()


def test_validate_and_load_x_posts_rejects_placeholder_csv(dummy_placeholder_csv_path):
    """Dummy placeholder CSV must be detected and rejected."""
    df, err = validate_and_load_x_posts(dummy_placeholder_csv_path, allow_synthetic=False)
    assert df is None
    assert "insufficient" in err.lower() or "placeholder" in err.lower()


def test_validate_and_load_x_posts_rejects_synthetic_by_default(synthetic_fixture_path):
    """Synthetic fixtures must be rejected when allow_synthetic=False."""
    assert synthetic_fixture_path.exists()
    df, err = validate_and_load_x_posts(synthetic_fixture_path, allow_synthetic=False)
    assert df is None
    assert "synthetic" in err.lower()


def test_validate_and_load_x_posts_accepts_synthetic_when_allowed(synthetic_fixture_path):
    """Synthetic fixture must load successfully when allow_synthetic=True."""
    df, err = validate_and_load_x_posts(synthetic_fixture_path, allow_synthetic=True)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "text" in df.columns
    assert "created_at" in df.columns
    parsed = pd.to_datetime(df["created_at"], utc=True)
    assert not parsed.isna().all()


def test_validate_and_load_x_posts_accepts_real_collected_posts(real_collected_csv_path):
    """Genuinely collected posts must pass validation without synthetic flag."""
    if not real_collected_csv_path.exists():
        pytest.skip("data/raw/x_posts_btc_normalized.csv does not exist.")

    df, err = validate_and_load_x_posts(real_collected_csv_path, allow_synthetic=False)
    assert err is None
    assert df is not None
    assert len(df) >= MIN_REAL_POSTS_THRESHOLD
    assert "is_synthetic" not in df.columns or not df["is_synthetic"].any()


def test_validate_and_load_x_posts_low_sample_count(tmp_path):
    """Fewer than MIN_REAL_POSTS_THRESHOLD records must be rejected in real mode."""
    few_posts = tmp_path / "few_posts.json"
    data = [
        {"id": f"post_{i}", "text": f"Bitcoin analysis report number {i}", "created_at": "2026-03-01T12:00:00Z"}
        for i in range(3)
    ]
    few_posts.write_text(json.dumps(data))

    df, err = validate_and_load_x_posts(few_posts, allow_synthetic=False)
    assert df is None
    assert "insufficient" in err.lower()

    # But passes if allow_synthetic=True
    loaded, err2 = validate_and_load_x_posts(few_posts, allow_synthetic=True)
    assert err2 is None
    assert len(loaded) == 3


def test_validate_and_load_x_posts_missing_columns(tmp_path):
    """Posts missing text or created_at column must return error."""
    bad_posts = tmp_path / "bad_posts.json"
    data = [{"id": "1", "score": 5.0}]
    bad_posts.write_text(json.dumps(data))

    df, err = validate_and_load_x_posts(bad_posts, allow_synthetic=True)
    assert df is None
    assert "missing required columns" in err.lower()


# ============================================================================
# 2. Pipeline Dataset Builder & Feature Quarantine
# ============================================================================

def test_build_integrated_pipeline_dataset_preserves_prices(price_path, synthetic_fixture_path):
    """Dataset builder preserves exact historical BTC price series and length."""
    contiguous_prices = load_and_validate_btc_prices(price_path)

    posts_df, err = validate_and_load_x_posts(synthetic_fixture_path, allow_synthetic=True)
    assert err is None

    df, feat_cols = build_integrated_pipeline_dataset(
        price_df=contiguous_prices,
        posts_df=posts_df,
    )

    assert len(df) == len(contiguous_prices)
    np.testing.assert_array_equal(df["Close"].values, contiguous_prices["Close"].values)
    np.testing.assert_array_equal(df["Open"].values, contiguous_prices["Open"].values)


def test_build_integrated_pipeline_dataset_leakage_quarantine(price_path, synthetic_fixture_path):
    """Feature columns must never contain forward targets, raw prices, or post metadata."""
    contiguous_prices = load_and_validate_btc_prices(price_path)

    posts_df, err = validate_and_load_x_posts(synthetic_fixture_path, allow_synthetic=True)
    assert err is None

    df, feat_cols = build_integrated_pipeline_dataset(
        price_df=contiguous_prices,
        posts_df=posts_df,
    )

    forbidden_patterns = [
        "target", "forward", "trade_return",
        "Open", "High", "Low", "Close",
        "post_id", "url", "text", "is_synthetic", "author", "date"
    ]

    for col in feat_cols:
        for forbidden in forbidden_patterns:
            assert col != forbidden, f"Forbidden exact column '{col}' found in feature matrix!"
            if forbidden in ["target", "forward", "trade_return"]:
                assert not col.startswith(forbidden), f"Leakage column '{col}' found in features!"

    assert any(col.startswith("return_") for col in feat_cols)
    assert any(col.startswith("sentiment_") or col == "post_count" for col in feat_cols)


def test_build_integrated_pipeline_dataset_missing_sentiment_convention(price_path, synthetic_fixture_path):
    """Days without posts must receive post_count=0 and well-defined sentiment features."""
    contiguous_prices = load_and_validate_btc_prices(price_path)

    posts_df, err = validate_and_load_x_posts(synthetic_fixture_path, allow_synthetic=True)
    assert err is None

    df, feat_cols = build_integrated_pipeline_dataset(
        price_df=contiguous_prices,
        posts_df=posts_df,
    )

    zero_post_days = df[df["post_count"] == 0]
    assert len(zero_post_days) > 0, "Expected days with 0 posts in synthetic dataset."

    for col in feat_cols:
        assert col in df.columns


# ============================================================================
# 3. Pipeline End-to-End Execution & Guardrails
# ============================================================================

def test_run_pipeline_price_only(price_path):
    """mode='price_only' must execute and return valid baseline metrics without social data."""
    summary = run_pipeline(
        price_csv_path=str(price_path),
        posts_csv_path=None,
        mode="price_only",
    )
    assert "price_only_backtest" in summary
    assert summary["price_only_backtest"] is not None
    metrics = summary["price_only_backtest"].metrics
    assert metrics["n_days"] == 75
    assert metrics["num_trades"] > 0
    assert "win_rate" in metrics
    assert "all_period_win_rate" in metrics


def test_run_pipeline_sentiment_blocked_on_dummy_data(price_path, dummy_placeholder_csv_path):
    """mode='sentiment' must set sentiment_status to BLOCKED when given dummy placeholder data."""
    summary = run_pipeline(
        price_csv_path=str(price_path),
        posts_csv_path=str(dummy_placeholder_csv_path),
        mode="sentiment",
        allow_synthetic=False,
    )
    assert summary["sentiment_status"] == "BLOCKED"
    assert "placeholder" in summary["block_reason"].lower() or "insufficient" in summary["block_reason"].lower()


def test_run_pipeline_auto_mode_blocks_sentiment_on_dummy_data(price_path, dummy_placeholder_csv_path):
    """mode='auto' should complete baseline and report sentiment blocked on dummy data."""
    summary = run_pipeline(
        price_csv_path=str(price_path),
        posts_csv_path=str(dummy_placeholder_csv_path),
        mode="auto",
        allow_synthetic=False,
    )
    assert summary["price_only_backtest"] is not None
    assert summary["sentiment_backtest"] is None
    assert summary["sentiment_status"] == "BLOCKED"
    assert summary["block_reason"] is not None


def test_run_pipeline_auto_mode_with_real_posts(price_path, real_collected_csv_path):
    """mode='auto' executes both baseline and sentiment tracks when real posts exist."""
    if not real_collected_csv_path.exists():
        pytest.skip("data/raw/x_posts_btc_normalized.csv does not exist.")

    summary = run_pipeline(
        price_csv_path=str(price_path),
        posts_csv_path=str(real_collected_csv_path),
        mode="auto",
        allow_synthetic=False,
    )
    assert summary["price_only_backtest"] is not None
    assert summary["sentiment_backtest"] is not None
    assert summary["sentiment_status"] == "COMPLETED"
    metrics = summary["sentiment_backtest"].metrics
    assert metrics["n_days"] == 75
    assert "cumulative_return" in metrics


def test_run_pipeline_sentiment_synthetic_offline(price_path, synthetic_fixture_path):
    """mode='sentiment' with allow_synthetic=True should complete walk-forward backtest."""
    summary = run_pipeline(
        price_csv_path=str(price_path),
        posts_csv_path=str(synthetic_fixture_path),
        mode="sentiment",
        allow_synthetic=True,
    )
    assert summary["sentiment_status"] == "COMPLETED"
    assert summary["sentiment_backtest"] is not None
    metrics = summary["sentiment_backtest"].metrics
    assert metrics["n_days"] == 75
    assert "cumulative_return" in metrics
    assert "annualized_return" in metrics
    assert summary["sentiment_result"] is not None
