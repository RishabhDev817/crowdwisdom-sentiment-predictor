"""
X (Twitter) post ingestion module via Apify for crowdwisdom-sentiment-predictor.

Connects to Apify, invokes a configured scraping actor (default: apidojo/tweet-scraper),
normalizes raw responses into a consistent tabular schema without fabricating metrics,
deduplicates records, and saves both raw provider JSON and cleaned CSV datasets.

Note:
    This module ingests and structures social text data. It does NOT generate
    sentiment polarity or subjectivity scores; sentiment modeling is handled
    in downstream processing stages.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import json
import logging
import os
import sys
import yaml
import dotenv
import pandas as pd
from apify_client import ApifyClient

# Configure logger
logger = logging.getLogger(__name__)

NORMALIZED_COLUMNS = [
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


class ApifyIngestionError(Exception):
    """Base exception for Apify ingestion errors."""
    pass


class MissingCredentialsError(ApifyIngestionError):
    """Raised when the Apify API token is missing or empty."""
    pass


class ApifyExecutionError(ApifyIngestionError):
    """Raised when an Apify actor run fails, is aborted, or returns an error."""
    pass


class EmptyDatasetError(ApifyIngestionError):
    """Raised when the actor yields no usable or valid post records."""
    pass


def get_apify_token() -> str:
    """
    Loads APIFY_API_TOKEN from the environment or .env file.

    Returns:
        The Apify API token.

    Raises:
        MissingCredentialsError: If APIFY_API_TOKEN is unset or empty.
    """
    dotenv.load_dotenv()
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not token or token == "your_apify_api_token_here":
        raise MissingCredentialsError(
            "APIFY_API_TOKEN environment variable is missing or contains placeholder. "
            "Please configure your valid Apify token in .env or your system environment."
        )
    return token


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration for the project.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If configuration file is missing.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def build_search_queries(
    search_terms: List[str],
    lookback_days: int = 180,
    language: Optional[str] = "en",
    end_date: Optional[datetime] = None,
) -> List[str]:
    """
    Constructs search query strings with language and date filtering operators.

    Twitter Advanced Search syntax:
        - `since:YYYY-MM-DD`: Inclusive lower boundary.
        - `until:YYYY-MM-DD`: Exclusive upper boundary (matches up to 00:00:00 UTC).

    Args:
        search_terms: List of base keywords or hashtags (e.g. ['BTC', 'Bitcoin']).
        lookback_days: Number of days of historical lookback.
        language: Language code filter (e.g. 'en'). If None, omitted.
        end_date: Upper boundary date (UTC). Defaults to current UTC time.

    Returns:
        List of formatted search query strings for the scraper actor.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be positive, got {lookback_days}")

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    else:
        end_date = end_date.astimezone(timezone.utc)

    start_date = end_date - timedelta(days=lookback_days)
    # Twitter 'until:' is exclusive at midnight, so we specify tomorrow to include today's posts
    fetch_until = end_date + timedelta(days=1)

    start_str = start_date.strftime("%Y-%m-%d")
    until_str = fetch_until.strftime("%Y-%m-%d")

    queries: List[str] = []
    for term in search_terms:
        clean_term = term.strip()
        if not clean_term:
            continue
        parts = [clean_term]
        if language:
            parts.append(f"lang:{language.strip()}")
        parts.append(f"since:{start_str}")
        parts.append(f"until:{until_str}")
        queries.append(" ".join(parts))

    return queries


def _parse_non_negative_int(val: Any) -> Optional[int]:
    """
    Safely parses non-negative integer values, returning None if missing or invalid.
    Never fabricates 0 from None or empty values.
    """
    if val is None or val == "":
        return None
    try:
        num = int(float(val))
        return num if num >= 0 else None
    except (ValueError, TypeError):
        return None


def _sanitize_token_message(msg: str, token: Optional[str] = None) -> str:
    """Removes token strings from error messages or logs if present."""
    if token and token in msg:
        return msg.replace(token, "[REDACTED]")
    return msg


def normalize_x_posts(raw_items: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, int]:
    """
    Normalizes raw actor output items into a standardized DataFrame schema.

    Schema:
        post_id (str), created_at (UTC ISO string), text (str), url (str),
        like_count (nullable int), retweet_count (nullable int),
        reply_count (nullable int), quote_count (nullable int),
        view_count (nullable int), author_followers (nullable int)

    Validation rules:
        - Must have a non-empty post ID (or valid fallback URL).
        - Must have non-empty text content.
        - Must have a parseable timestamp convertible to UTC.
        - Missing optional metrics remain None/NA; no zeroes are fabricated.

    Deduplication rules:
        - When duplicate post IDs exist across search queries, the record with
          the highest metric richness (most non-null engagement metrics) is deterministically preserved.

    Args:
        raw_items: List of raw dictionaries returned by Apify.

    Returns:
        Tuple of (normalized DataFrame, count of rejected invalid records).
    """
    normalized_records: List[Dict[str, Any]] = []
    rejected_count = 0

    for item in raw_items:
        if not isinstance(item, dict):
            rejected_count += 1
            continue

        # Post ID extraction
        raw_id = item.get("id") or item.get("id_str")
        post_id = str(raw_id).strip() if raw_id is not None else ""

        # URL extraction
        url = str(item.get("url") or item.get("twitterUrl") or "").strip()

        # Text extraction
        text = str(item.get("text") or item.get("full_text") or "").strip()

        # If ID is missing, try extracting from URL; if still missing, reject
        if not post_id and url:
            # Strip query parameters and trailing slashes: e.g. https://twitter.com/user/status/123456789?s=20
            clean_url = url.split("?")[0].rstrip("/")
            parts = [p for p in clean_url.split("/") if p]
            if len(parts) >= 2 and parts[-2] == "status" and parts[-1].isdigit():
                post_id = parts[-1]

        if not post_id or not text:
            rejected_count += 1
            continue

        # Timestamp parsing
        raw_created_at = item.get("createdAt") or item.get("created_at")
        if not raw_created_at:
            rejected_count += 1
            continue

        try:
            ts = pd.to_datetime(raw_created_at, utc=True)
            if pd.isna(ts):
                rejected_count += 1
                continue
            created_at_str = ts.strftime("%Y-%m-%d %H:%M:%S+00:00")
        except Exception:
            rejected_count += 1
            continue

        # Engagement metrics (defensive parsing preserving None)
        like_count = _parse_non_negative_int(item.get("likeCount") or item.get("favorite_count"))
        retweet_count = _parse_non_negative_int(item.get("retweetCount") or item.get("retweet_count"))
        reply_count = _parse_non_negative_int(item.get("replyCount") or item.get("reply_count"))
        quote_count = _parse_non_negative_int(item.get("quoteCount") or item.get("quote_count"))
        view_count = _parse_non_negative_int(item.get("viewCount") or item.get("views"))

        # Author followers
        author_obj = item.get("author") or item.get("user") or {}
        author_followers = None
        if isinstance(author_obj, dict):
            author_followers = _parse_non_negative_int(
                author_obj.get("followers") or author_obj.get("followers_count")
            )

        record = {
            "post_id": post_id,
            "created_at": created_at_str,
            "text": text,
            "url": url,
            "like_count": like_count,
            "retweet_count": retweet_count,
            "reply_count": reply_count,
            "quote_count": quote_count,
            "view_count": view_count,
            "author_followers": author_followers,
        }
        normalized_records.append(record)

    if not normalized_records:
        df = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    else:
        df = pd.DataFrame(normalized_records)
        # Deduplicate deterministically by post_id
        # Calculate metric richness to preserve the richest record among duplicates
        richness_cols = [
            "like_count",
            "retweet_count",
            "reply_count",
            "quote_count",
            "view_count",
            "author_followers",
        ]
        df["_metric_richness"] = df[richness_cols].notna().sum(axis=1)
        # Sort so highest metric richness comes first for each post_id
        df = df.sort_values(by=["post_id", "_metric_richness"], ascending=[True, False], kind="mergesort")
        df = df.drop_duplicates(subset=["post_id"], keep="first").drop(columns=["_metric_richness"])

        # Ensure chronological ordering
        df["_created_at_dt"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.sort_values(by="_created_at_dt", ascending=True, kind="mergesort").drop(columns=["_created_at_dt"])
        df = df.reset_index(drop=True)

    return df, rejected_count


def execute_apify_actor(
    client: ApifyClient,
    actor_id: str,
    run_input: Dict[str, Any],
    token_to_redact: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes an Apify actor and streams items from the resulting dataset.

    Args:
        client: Authenticated ApifyClient instance.
        actor_id: Identifier of the actor (e.g. 'apidojo/tweet-scraper').
        run_input: Input dictionary passed to the actor.
        token_to_redact: Optional token string to redact from any raised error messages.

    Returns:
        List of raw dictionaries from the output dataset.

    Raises:
        ApifyExecutionError: If actor run fails, aborts, or dataset is unreachable.
    """
    logger.info(f"Triggering Apify actor '{actor_id}'")
    try:
        actor_client = client.actor(actor_id)
        run = actor_client.call(run_input=run_input)
    except Exception as e:
        safe_msg = _sanitize_token_message(str(e), token_to_redact)
        raise ApifyExecutionError(f"Apify actor execution request failed: {safe_msg}") from None

    if not run or not isinstance(run, dict):
        raise ApifyExecutionError("Apify actor run returned no execution metadata.")

    status = run.get("status")
    if status != "SUCCEEDED":
        run_id = run.get("id", "unknown")
        raise ApifyExecutionError(
            f"Apify actor run {run_id} did not succeed (status: {status}). "
            "Please check your Apify console logs for run failure details."
        )

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        raise ApifyExecutionError("Apify actor run succeeded but returned no defaultDatasetId.")

    logger.info(f"Retrieving items from Apify dataset '{dataset_id}'")
    try:
        dataset_client = client.dataset(dataset_id)
        raw_items = list(dataset_client.iterate_items())
    except Exception as e:
        safe_msg = _sanitize_token_message(str(e), token_to_redact)
        raise ApifyExecutionError(f"Failed to retrieve items from dataset {dataset_id}: {safe_msg}") from None

    return raw_items


def save_x_posts(
    raw_items: List[Dict[str, Any]],
    normalized_df: pd.DataFrame,
    raw_json_path: Union[str, Path] = "data/raw/x_posts_btc_raw.json",
    normalized_csv_path: Union[str, Path] = "data/raw/x_posts_btc_normalized.csv",
) -> Tuple[Path, Path]:
    """
    Persists raw provider JSON and normalized unique records CSV to data/raw/.

    Args:
        raw_items: Unmodified items returned by Apify.
        normalized_df: Cleaned and deduplicated DataFrame.
        raw_json_path: Path for raw JSON output.
        normalized_csv_path: Path for normalized CSV output.

    Returns:
        Tuple of (resolved raw_json_path, resolved normalized_csv_path).
    """
    json_path = Path(raw_json_path)
    csv_path = Path(normalized_csv_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, indent=2, default=str)

    normalized_df.to_csv(csv_path, index=False)

    return json_path.resolve(), csv_path.resolve()


def ingest_x_posts_pipeline(
    config_path: Union[str, Path] = "config/config.yaml",
    actor_override: Optional[str] = None,
    max_items_override: Optional[int] = None,
    search_terms_override: Optional[Union[str, List[str]]] = None,
    raw_json_override: Optional[Union[str, Path]] = None,
    normalized_csv_override: Optional[Union[str, Path]] = None,
    client: Optional[ApifyClient] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Orchestrates configuration loading, Apify execution, normalization, and persistence.

    Args:
        config_path: Path to YAML configuration file.
        actor_override: Optional override for Apify actor ID.
        max_items_override: Optional override for max items.
        search_terms_override: Optional override for search term(s) (e.g. 'Bitcoin' or ['Bitcoin']).
        raw_json_override: Optional override for raw JSON path.
        normalized_csv_override: Optional override for normalized CSV path.
        client: Optional ApifyClient (useful for testing and dependency injection).

    Returns:
        Tuple of (normalized DataFrame, raw JSON path, normalized CSV path).

    Raises:
        MissingCredentialsError: If token is unset.
        ApifyExecutionError: If actor run fails.
        EmptyDatasetError: If dataset yields no usable post records.
    """
    config = load_config(config_path)
    asset_cfg = config.get("asset", {})
    sentiment_cfg = config.get("sentiment", {})
    paths_cfg = config.get("paths", {})

    actor_id = actor_override or sentiment_cfg.get("apify_actor_id", "apidojo/tweet-scraper")
    if search_terms_override is not None:
        if isinstance(search_terms_override, str):
            search_terms = [search_terms_override]
        else:
            search_terms = list(search_terms_override)
    else:
        search_terms = sentiment_cfg.get("search_terms", ["Bitcoin"])
    language = sentiment_cfg.get("language", "en")
    max_items = max_items_override or sentiment_cfg.get("max_items", 10)
    sort_order = sentiment_cfg.get("sort", "Latest")

    # Reuse asset lookback days by default
    lookback_days = sentiment_cfg.get("lookback_days") or asset_cfg.get("lookback_days", 180)

    raw_dir = Path(paths_cfg.get("raw_data_dir", "data/raw"))
    json_path = raw_json_override or raw_dir / sentiment_cfg.get(
        "raw_json_filename", "x_posts_btc_raw.json"
    )
    csv_path = normalized_csv_override or raw_dir / sentiment_cfg.get(
        "normalized_csv_filename", "x_posts_btc_normalized.csv"
    )

    queries = build_search_queries(
        search_terms=search_terms,
        lookback_days=lookback_days,
        language=language,
    )

    run_input = {
        "searchTerms": queries,
        "sort": sort_order,
        "maxItems": max_items,
    }

    token = None
    if client is None:
        token = get_apify_token()
        client = ApifyClient(token=token)

    raw_items = execute_apify_actor(
        client=client,
        actor_id=actor_id,
        run_input=run_input,
        token_to_redact=token,
    )

    if not raw_items:
        raise EmptyDatasetError("Apify actor run completed but returned 0 dataset items.")

    df, rejected_count = normalize_x_posts(raw_items)

    if df.empty:
        raise EmptyDatasetError(
            f"All {len(raw_items)} items returned by Apify were rejected as invalid or unparseable."
        )

    saved_json, saved_csv = save_x_posts(
        raw_items=raw_items,
        normalized_df=df,
        raw_json_path=json_path,
        normalized_csv_path=csv_path,
    )

    return df, saved_json, saved_csv


def main() -> None:
    """CLI entry point for X posts ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest historical X (Twitter) posts for BTC sentiment using Apify."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--actor-id",
        type=str,
        default=None,
        help="Apify actor ID override (default: from config)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum items override (default: from config)",
    )
    parser.add_argument(
        "--search-term",
        type=str,
        default=None,
        help="Single search term override (default: from config, e.g. 'Bitcoin')",
    )
    parser.add_argument(
        "--raw-json",
        type=str,
        default=None,
        help="Raw JSON output path override",
    )
    parser.add_argument(
        "--normalized-csv",
        type=str,
        default=None,
        help="Normalized CSV output path override",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        df, out_json, out_csv = ingest_x_posts_pipeline(
            config_path=args.config,
            actor_override=args.actor_id,
            max_items_override=args.max_items,
            search_terms_override=args.search_term,
            raw_json_override=args.raw_json,
            normalized_csv_override=args.normalized_csv,
        )
        print("\n=== X (Twitter) Post Ingestion Summary ===")
        print(f"Actor ID:            {args.actor_id or 'apidojo/tweet-scraper'}")
        print(f"Unique Records:      {len(df)}")
        if not df.empty and "created_at" in df.columns:
            print(f"Date Range:          {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"Raw JSON Output:     {out_json}")
        print(f"Normalized CSV:      {out_csv}")
        print("==========================================\n")
    except Exception as e:
        print(f"Error during X post ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
