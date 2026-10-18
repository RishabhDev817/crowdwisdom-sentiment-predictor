"""
Offline unit tests for src/ingestion/x_posts.py.

All tests use synthetic fixtures and mocks to verify that no network requests,
real API tokens, or external services are called during execution.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import os
import pandas as pd
import pytest

from src.ingestion.x_posts import (
    ApifyExecutionError,
    EmptyDatasetError,
    MissingCredentialsError,
    build_search_queries,
    execute_apify_actor,
    get_apify_token,
    ingest_x_posts_pipeline,
    load_config,
    normalize_x_posts,
    save_x_posts,
)


@pytest.fixture
def sample_raw_tweet_items():
    """Provides a list of typical raw items matching apidojo/tweet-scraper schema."""
    return [
        {
            "id": "1840000000000000001",
            "url": "https://twitter.com/crypto_analyst/status/1840000000000000001",
            "text": "Bitcoin is testing major resistance at $80,000! Bullish momentum building. #BTC",
            "createdAt": "Wed Oct 14 12:30:00 +0000 2026",
            "likeCount": 150,
            "retweetCount": 45,
            "replyCount": 12,
            "quoteCount": 4,
            "viewCount": 5400,
            "author": {
                "userName": "crypto_analyst",
                "name": "Crypto Analyst",
                "followers": 12500,
            },
        },
        {
            "id": "1840000000000000002",
            "url": "https://twitter.com/trader_bob/status/1840000000000000002",
            "text": "BTC consolidation looks healthy. Watching daily close carefully.",
            "createdAt": "2026-10-15T08:15:00.000Z",
            "likeCount": 20,
            "retweetCount": 3,
            "replyCount": 1,
            "quoteCount": 0,
            "viewCount": None,  # View count not exposed
            "author": {
                "userName": "trader_bob",
                "name": "Trader Bob",
                # followers omitted
            },
        },
    ]


# 1. Missing Token & Token Security
def test_get_apify_token_missing():
    """Verifies MissingCredentialsError when APIFY_API_TOKEN is unset."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(MissingCredentialsError, match="APIFY_API_TOKEN"):
            get_apify_token()


def test_get_apify_token_placeholder():
    """Verifies MissingCredentialsError when token is the .env.example placeholder."""
    with patch.dict(os.environ, {"APIFY_API_TOKEN": "your_apify_api_token_here"}, clear=True):
        with pytest.raises(MissingCredentialsError, match="placeholder"):
            get_apify_token()


def test_get_apify_token_success():
    """Verifies retrieval of a validly configured token."""
    secret_val = "apify_api_test_dummy_token_12345"
    with patch.dict(os.environ, {"APIFY_API_TOKEN": secret_val}, clear=True):
        token = get_apify_token()
        assert token == secret_val


def test_no_token_leakage_in_error_message():
    """Verifies that sensitive token contents are never leaked in error messages."""
    secret_token = "secret_top_confidential_token_999"
    with patch.dict(os.environ, {"APIFY_API_TOKEN": secret_token}):
        try:
            # Simulate a failure inside pipeline with invalid config
            ingest_x_posts_pipeline(config_path="non_existent_config.yaml")
        except Exception as e:
            assert secret_token not in str(e)


# 2. Normalization & UTC Timestamp Parsing
def test_normalize_x_posts_success(sample_raw_tweet_items):
    """Verifies complete schema normalization and UTC timestamp conversion."""
    df, rejected = normalize_x_posts(sample_raw_tweet_items)

    assert rejected == 0
    assert len(df) == 2
    assert list(df.columns) == [
        "post_id",
        "created_at",
        "text",
        "url",
        "like_count",
        "retweet_count",
        "reply_count",
        "quote_count",
        "view_count",
        "author_followers",
    ]

    first = df.iloc[0]
    assert first["post_id"] == "1840000000000000001"
    assert first["like_count"] == 150
    assert first["retweet_count"] == 45
    assert first["view_count"] == 5400
    assert first["author_followers"] == 12500
    assert "+00:00" in first["created_at"]


# 3. Missing Optional Metrics Remain Null
def test_missing_optional_metrics_preserved_as_null(sample_raw_tweet_items):
    """Verifies that missing views and followers stay None without fabricating 0."""
    df, _ = normalize_x_posts(sample_raw_tweet_items)
    second = df.iloc[1]

    # In second fixture item, viewCount was None and followers was omitted
    assert pd.isna(second["view_count"]) or second["view_count"] is None
    assert pd.isna(second["author_followers"]) or second["author_followers"] is None
    # Ensure they are NOT converted into zero
    assert second["view_count"] != 0
    assert second["author_followers"] != 0


# 4. Invalid Fields & Invalid Timestamps
def test_normalize_x_posts_rejection_rules():
    """Tests rejection of records missing ID, text, or valid timestamps."""
    bad_items = [
        # Missing ID and URL
        {"text": "Valid text without ID", "createdAt": "2026-10-15T08:00:00Z"},
        # Missing text
        {"id": "1001", "text": "", "createdAt": "2026-10-15T08:00:00Z"},
        # Whitespace-only text
        {"id": "1002", "text": "   ", "createdAt": "2026-10-15T08:00:00Z"},
        # Invalid unparseable timestamp
        {"id": "1003", "text": "Valid text", "createdAt": "invalid-timestamp-string"},
        # Missing timestamp completely
        {"id": "1004", "text": "Valid text"},
        # Non-dict item
        "invalid string item",
    ]

    df, rejected = normalize_x_posts(bad_items)
    assert len(df) == 0
    assert rejected == len(bad_items)


def test_normalize_x_posts_url_fallback_id():
    """Verifies ID extraction fallback from URL when 'id' field is empty."""
    items = [
        {
            "url": "https://twitter.com/analyst/status/1999888777",
            "text": "Bitcoin rally continues!",
            "createdAt": "2026-10-15T08:00:00Z",
        }
    ]
    df, rejected = normalize_x_posts(items)
    assert rejected == 0
    assert len(df) == 1
    assert df.iloc[0]["post_id"] == "1999888777"


def test_normalize_x_posts_url_fallback_with_query_params():
    """Verifies ID extraction from URL with complex query parameters and trailing slashes."""
    items = [
        {
            "url": "https://x.com/crypto_trader/status/888777666555444333?s=20&t=xYz123/",
            "text": "BTC daily close above 80k!",
            "createdAt": "2026-10-15T09:00:00Z",
        }
    ]
    df, rejected = normalize_x_posts(items)
    assert rejected == 0
    assert len(df) == 1
    assert df.iloc[0]["post_id"] == "888777666555444333"


# 5. Deduplication
def test_deduplication_by_post_id(sample_raw_tweet_items):
    """Verifies that duplicate post IDs are removed deterministically."""
    # Duplicate first item
    duplicated_items = [sample_raw_tweet_items[0], sample_raw_tweet_items[0], sample_raw_tweet_items[1]]
    df, rejected = normalize_x_posts(duplicated_items)

    assert rejected == 0
    assert len(df) == 2
    assert len(df["post_id"].unique()) == 2


def test_deduplication_preserves_richest_record():
    """Verifies that between two duplicate post IDs, the record with more populated metrics is preserved."""
    sparse_item = {
        "id": "777000111",
        "url": "https://twitter.com/user/status/777000111",
        "text": "Bitcoin ETF inflows surge!",
        "createdAt": "2026-10-15T10:00:00Z",
        "likeCount": None,
        "viewCount": None,
        "retweetCount": None,
    }
    rich_item = {
        "id": "777000111",
        "url": "https://twitter.com/user/status/777000111",
        "text": "Bitcoin ETF inflows surge!",
        "createdAt": "2026-10-15T10:00:00Z",
        "likeCount": 120,
        "viewCount": 4500,
        "retweetCount": 35,
        "replyCount": 10,
        "author": {"followers": 8000},
    }

    # Order 1: sparse first
    df1, _ = normalize_x_posts([sparse_item, rich_item])
    assert len(df1) == 1
    assert df1.iloc[0]["like_count"] == 120
    assert df1.iloc[0]["view_count"] == 4500
    assert df1.iloc[0]["author_followers"] == 8000

    # Order 2: rich first
    df2, _ = normalize_x_posts([rich_item, sparse_item])
    assert len(df2) == 1
    assert df2.iloc[0]["like_count"] == 120
    assert df2.iloc[0]["view_count"] == 4500


# 6. Actor Failure Status
def test_execute_apify_actor_failed_status():
    """Verifies that actor run with status FAILED raises ApifyExecutionError."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_actor.call.return_value = {"id": "run_123", "status": "FAILED"}

    with pytest.raises(ApifyExecutionError, match="status: FAILED"):
        execute_apify_actor(mock_client, "apidojo/tweet-scraper", {})


def test_execute_apify_actor_aborted_status():
    """Verifies that actor run with status ABORTED raises ApifyExecutionError."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_actor.call.return_value = {"id": "run_456", "status": "ABORTED"}

    with pytest.raises(ApifyExecutionError, match="status: ABORTED"):
        execute_apify_actor(mock_client, "apidojo/tweet-scraper", {})


def test_execute_apify_actor_redacts_token_on_exception():
    """Verifies that if an underlying exception contains the API token, it is redacted."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor
    secret_token = "super_confidential_token_xyz999"
    mock_actor.call.side_effect = RuntimeError(f"Connection failed for Bearer {secret_token}")

    with pytest.raises(ApifyExecutionError) as exc_info:
        execute_apify_actor(
            client=mock_client,
            actor_id="apidojo/tweet-scraper",
            run_input={},
            token_to_redact=secret_token,
        )

    err_msg = str(exc_info.value)
    assert secret_token not in err_msg
    assert "[REDACTED]" in err_msg


# 7. Missing Dataset ID or Dataset Retrieval Errors
def test_execute_apify_actor_missing_dataset_id():
    """Verifies error when actor run succeeds but defaultDatasetId is absent."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_actor.call.return_value = {"id": "run_789", "status": "SUCCEEDED"}

    with pytest.raises(ApifyExecutionError, match="returned no defaultDatasetId"):
        execute_apify_actor(mock_client, "apidojo/tweet-scraper", {})


def test_execute_apify_actor_dataset_iteration_error():
    """Verifies error handling if dataset item streaming fails."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_dataset = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.return_value = mock_dataset

    mock_actor.call.return_value = {"id": "run_789", "status": "SUCCEEDED", "defaultDatasetId": "dataset_123"}
    mock_dataset.iterate_items.side_effect = RuntimeError("Dataset connection broken")

    with pytest.raises(ApifyExecutionError, match="Failed to retrieve items from dataset"):
        execute_apify_actor(mock_client, "apidojo/tweet-scraper", {})


# 8. Empty Dataset & All-Invalid Records
def test_ingest_x_posts_pipeline_empty_dataset():
    """Verifies EmptyDatasetError when Apify returns 0 items."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_dataset = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.return_value = mock_dataset

    mock_actor.call.return_value = {"id": "run_000", "status": "SUCCEEDED", "defaultDatasetId": "ds_000"}
    mock_dataset.iterate_items.return_value = iter([])

    with pytest.raises(EmptyDatasetError, match="returned 0 dataset items"):
        ingest_x_posts_pipeline(config_path="config/config.yaml", client=mock_client)


def test_ingest_x_posts_pipeline_all_invalid():
    """Verifies EmptyDatasetError when all items are rejected during normalization."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_dataset = MagicMock()
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.return_value = mock_dataset

    mock_actor.call.return_value = {"id": "run_001", "status": "SUCCEEDED", "defaultDatasetId": "ds_001"}
    # Items missing required fields
    mock_dataset.iterate_items.return_value = iter([{"invalid": "data"}, {"id": "", "text": ""}])

    with pytest.raises(EmptyDatasetError, match="rejected as invalid or unparseable"):
        ingest_x_posts_pipeline(config_path="config/config.yaml", client=mock_client)


# 9. JSON and CSV Persistence
def test_save_x_posts(sample_raw_tweet_items, tmp_path: Path):
    """Verifies that raw JSON and normalized CSV files are persisted accurately."""
    df, _ = normalize_x_posts(sample_raw_tweet_items)
    raw_json_path = tmp_path / "raw.json"
    normalized_csv_path = tmp_path / "normalized.csv"

    out_json, out_csv = save_x_posts(
        raw_items=sample_raw_tweet_items,
        normalized_df=df,
        raw_json_path=raw_json_path,
        normalized_csv_path=normalized_csv_path,
    )

    assert out_json.exists()
    assert out_csv.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        loaded_raw = json.load(f)
    assert len(loaded_raw) == 2

    loaded_csv = pd.read_csv(out_csv)
    assert len(loaded_csv) == 2
    assert "post_id" in loaded_csv.columns


# 10. Query Construction & Configuration
def test_build_search_queries():
    """Verifies query operator construction with since/until boundaries."""
    fixed_end = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    queries = build_search_queries(
        search_terms=["BTC", "Bitcoin", "#BTC"],
        lookback_days=180,
        language="en",
        end_date=fixed_end,
    )

    assert len(queries) == 3
    for q in queries:
        assert "lang:en" in q
        assert "since:2026-04-21" in q
        assert "until:2026-10-19" in q  # Exclusive until is next day


def test_build_search_queries_invalid_lookback():
    """Verifies ValueError when lookback is non-positive."""
    with pytest.raises(ValueError, match="lookback_days must be positive"):
        build_search_queries(search_terms=["BTC"], lookback_days=0)


def test_load_config_sentiment_section():
    """Verifies that config/config.yaml contains the required sentiment settings."""
    config = load_config("config/config.yaml")
    assert "sentiment" in config
    sentiment = config["sentiment"]
    assert sentiment["apify_actor_id"] == "apidojo/tweet-scraper"
    assert sentiment["search_terms"] == ["BTC", "Bitcoin", "#BTC"]
    assert sentiment["language"] == "en"
    assert sentiment["max_items"] == 500


# 11. Pipeline Orchestration Mocked End-to-End
def test_ingest_x_posts_pipeline_mocked_success(sample_raw_tweet_items, tmp_path: Path):
    """Verifies full ingestion pipeline flow with mocked client and custom output paths."""
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_dataset = MagicMock()

    mock_client.actor.return_value = mock_actor
    mock_client.dataset.return_value = mock_dataset

    mock_actor.call.return_value = {"id": "run_valid", "status": "SUCCEEDED", "defaultDatasetId": "ds_valid"}
    mock_dataset.iterate_items.return_value = iter(sample_raw_tweet_items)

    out_json = tmp_path / "custom_raw.json"
    out_csv = tmp_path / "custom_normalized.csv"

    df, saved_json, saved_csv = ingest_x_posts_pipeline(
        config_path="config/config.yaml",
        raw_json_override=out_json,
        normalized_csv_override=out_csv,
        client=mock_client,
    )

    assert len(df) == 2
    assert saved_json == out_json.resolve()
    assert saved_csv == out_csv.resolve()
    assert saved_json.exists()
    assert saved_csv.exists()
