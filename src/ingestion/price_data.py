"""
Historical price data ingestion module for crowdwisdom-sentiment-predictor.

Fetches historical daily OHLCV bars for crypto assets (BTC-USD) from Yahoo Finance,
applies structural and mathematical validation, normalizes timestamps,
and persists raw data to disk without fabricating or imputing missing prices.

Timestamp Convention:
    Yahoo Finance daily candles for 24/7 crypto assets (such as BTC-USD) represent
    a 24-hour trading session spanning 00:00:00 UTC through 23:59:59 UTC.
    Candles are indexed by their calendar date (YYYY-MM-DD) normalized to UTC midnight.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import argparse
import logging
import sys
import yaml
import pandas as pd
import yfinance as yf

# Configure logger
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class PriceIngestionError(Exception):
    """Base exception for price data ingestion failures."""
    pass


class PriceValidationError(PriceIngestionError):
    """Raised when price data fails structural or mathematical validation."""
    pass


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML project configuration.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def validate_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates structural integrity, positive prices, and geometric OHLC relationships.

    Validation rules:
        1. DataFrame must not be empty.
        2. Required columns ('Open', 'High', 'Low', 'Close', 'Volume') must be present.
        3. Index or timestamps must not contain duplicates.
        4. No missing (NaN / None) values in required columns.
        5. OHLC prices must be strictly positive (> 0).
        6. Volume must be non-negative (>= 0).
        7. Geometric validity: High >= Low, High >= Open, High >= Close,
           Low <= Open, Low <= Close.
        8. Records must be sorted chronologically.

    Args:
        df: Input DataFrame containing OHLCV records.

    Returns:
        Validated and chronologically sorted DataFrame with normalized index.

    Raises:
        PriceValidationError: If any validation rule is violated.
    """
    if df is None or df.empty:
        raise PriceValidationError("Downloaded price data is empty or None.")

    # Flatten MultiIndex columns if produced by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Verify required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise PriceValidationError(f"Missing required price columns: {missing_cols}")

    # Work on a copy of the required columns
    validated = df[REQUIRED_COLUMNS].copy()

    # Normalize index to DatetimeIndex
    if not isinstance(validated.index, pd.DatetimeIndex):
        try:
            validated.index = pd.to_datetime(validated.index, utc=True)
        except Exception as e:
            raise PriceValidationError(f"Unable to parse index into DatetimeIndex: {e}") from e
    else:
        if validated.index.tz is None:
            validated.index = validated.index.tz_localize("UTC")
        else:
            validated.index = validated.index.tz_convert("UTC")

    # Normalize daily timestamps to UTC midnight
    validated.index = validated.index.normalize()
    validated.index.name = "Date"

    # Check for duplicate timestamps
    if validated.index.duplicated().any():
        dup_dates = validated.index[validated.index.duplicated()].tolist()
        raise PriceValidationError(f"Duplicate timestamps detected in price data: {dup_dates}")

    # Check for NaN / missing values
    if validated.isna().any().any():
        nan_counts = validated.isna().sum().to_dict()
        raise PriceValidationError(f"Price data contains missing/NaN values: {nan_counts}")

    # Validate strictly positive prices
    for col in ["Open", "High", "Low", "Close"]:
        if (validated[col] <= 0).any():
            invalid_rows = validated[validated[col] <= 0]
            raise PriceValidationError(
                f"Non-positive price found in column '{col}': {invalid_rows[col].to_dict()}"
            )

    # Validate non-negative volume
    if (validated["Volume"] < 0).any():
        invalid_vol = validated[validated["Volume"] < 0]
        raise PriceValidationError(
            f"Negative volume found: {invalid_vol['Volume'].to_dict()}"
        )

    # Validate geometric OHLC relationships
    invalid_high_low = validated[validated["High"] < validated["Low"]]
    if not invalid_high_low.empty:
        raise PriceValidationError(
            f"Invalid High < Low relationship at: {invalid_high_low.index.tolist()}"
        )

    invalid_high_open = validated[validated["High"] < validated["Open"]]
    if not invalid_high_open.empty:
        raise PriceValidationError(
            f"Invalid High < Open relationship at: {invalid_high_open.index.tolist()}"
        )

    invalid_high_close = validated[validated["High"] < validated["Close"]]
    if not invalid_high_close.empty:
        raise PriceValidationError(
            f"Invalid High < Close relationship at: {invalid_high_close.index.tolist()}"
        )

    invalid_low_open = validated[validated["Low"] > validated["Open"]]
    if not invalid_low_open.empty:
        raise PriceValidationError(
            f"Invalid Low > Open relationship at: {invalid_low_open.index.tolist()}"
        )

    invalid_low_close = validated[validated["Low"] > validated["Close"]]
    if not invalid_low_close.empty:
        raise PriceValidationError(
            f"Invalid Low > Close relationship at: {invalid_low_close.index.tolist()}"
        )

    # Sort chronologically
    validated = validated.sort_index(ascending=True)

    return validated


def fetch_historical_price_data(
    symbol: str = "BTC-USD",
    lookback_days: int = 180,
    interval: str = "1d",
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetches historical asset prices using yfinance with explicit date boundaries.

    Args:
        symbol: Ticker symbol (e.g. 'BTC-USD').
        lookback_days: Number of calendar days of history to retrieve.
        interval: Bar interval (e.g. '1d').
        end_date: Upper boundary date (UTC). Defaults to current UTC time.

    Returns:
        Validated DataFrame of OHLCV bars.

    Raises:
        PriceIngestionError: If network download fails or API returns unreadable data.
        PriceValidationError: If returned data violates validation rules.
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

    # In yfinance, the `end` parameter for daily data is exclusive of the end date.
    # To include the most recent completed daily candle, we specify end + 1 day.
    fetch_end = end_date + timedelta(days=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = fetch_end.strftime("%Y-%m-%d")

    logger.info(
        f"Fetching {symbol} ({interval}) from {start_str} to {end_str} (~{lookback_days} days lookback)"
    )

    try:
        raw_df = yf.download(
            tickers=symbol,
            start=start_str,
            end=end_str,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        raise PriceIngestionError(f"Failed to download price data from yfinance for {symbol}: {e}") from e

    if raw_df is None or raw_df.empty:
        raise PriceIngestionError(
            f"yfinance returned an empty dataset for {symbol} between {start_str} and {end_str}."
        )

    # Validate and normalize
    validated_df = validate_ohlcv_dataframe(raw_df)
    return validated_df


def save_price_data(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "data/raw/btc_usd_daily_raw.csv",
) -> Path:
    """
    Persists validated price DataFrame to a CSV file.

    Args:
        df: Validated DataFrame to persist.
        output_path: Target CSV file path.

    Returns:
        Resolved Path to the saved CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, date_format="%Y-%m-%d")
    return path.resolve()


def ingest_price_pipeline(
    config_path: Union[str, Path] = "config/config.yaml",
    symbol_override: Optional[str] = None,
    lookback_override: Optional[int] = None,
    output_override: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    Orchestrates configuration loading, price fetching, validation, and storage.

    Args:
        config_path: Path to config.yaml.
        symbol_override: Optional override for asset symbol.
        lookback_override: Optional override for lookback days.
        output_override: Optional override for output CSV path.

    Returns:
        Tuple of (validated DataFrame, resolved output Path).
    """
    config = load_config(config_path)
    asset_cfg = config.get("asset", {})
    paths_cfg = config.get("paths", {})

    symbol = symbol_override or asset_cfg.get("symbol", "BTC-USD")
    lookback_days = lookback_override or asset_cfg.get("lookback_days", 180)
    interval = asset_cfg.get("interval", "1d")

    raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
    safe_symbol = symbol.lower().replace("-", "_")
    default_filename = f"{safe_symbol}_daily_raw.csv"
    output_path = output_override or Path(raw_dir) / default_filename

    df = fetch_historical_price_data(
        symbol=symbol,
        lookback_days=lookback_days,
        interval=interval,
    )

    saved_path = save_price_data(df, output_path=output_path)
    return df, saved_path


def main() -> None:
    """CLI entry point for historical price ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest historical daily OHLCV prices for BTC-USD."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Asset symbol override (default: from config)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Historical lookback days override (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path override (default: data/raw/<symbol>_daily_raw.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        df, out_path = ingest_price_pipeline(
            config_path=args.config,
            symbol_override=args.symbol,
            lookback_override=args.lookback_days,
            output_override=args.output,
        )
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
        print("\n=== Price Ingestion Summary ===")
        print(f"Asset Symbol:    {args.symbol or 'BTC-USD'}")
        print(f"Date Range:      {start_date} to {end_date}")
        print(f"Row Count:       {len(df)}")
        print(f"Columns:         {list(df.columns)}")
        print(f"Output File:     {out_path}")
        print("===============================\n")
    except Exception as e:
        print(f"Error during price ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
