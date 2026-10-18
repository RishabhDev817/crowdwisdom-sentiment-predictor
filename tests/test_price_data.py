"""
Unit tests for src/ingestion/price_data.py.

All tests use synthetic test fixtures and mocks to guarantee that no real network
or external API calls are made during test execution.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.ingestion.price_data import (
    PriceIngestionError,
    PriceValidationError,
    fetch_historical_price_data,
    ingest_price_pipeline,
    load_config,
    save_price_data,
    validate_ohlcv_dataframe,
)


@pytest.fixture
def sample_valid_ohlcv_df() -> pd.DataFrame:
    """Creates a small, geometrically valid OHLCV DataFrame."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    data = {
        "Open": [50000.0, 50500.0, 51000.0, 49500.0, 50200.0],
        "High": [51000.0, 51500.0, 51200.0, 50800.0, 51000.0],
        "Low": [49000.0, 50000.0, 49200.0, 49000.0, 49800.0],
        "Close": [50500.0, 51000.0, 49500.0, 50200.0, 50800.0],
        "Volume": [15000.0, 22000.0, 18500.0, 31000.0, 14200.0],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def test_validate_ohlcv_dataframe_valid(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that a valid OHLCV DataFrame passes validation cleanly."""
    validated = validate_ohlcv_dataframe(sample_valid_ohlcv_df)
    assert len(validated) == 5
    assert list(validated.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert validated.index.is_monotonic_increasing
    assert str(validated.index.tz) == "UTC"


def test_validate_ohlcv_dataframe_empty():
    """Verifies that empty or None DataFrames raise PriceValidationError."""
    with pytest.raises(PriceValidationError, match="empty or None"):
        validate_ohlcv_dataframe(pd.DataFrame())

    with pytest.raises(PriceValidationError, match="empty or None"):
        validate_ohlcv_dataframe(None)  # type: ignore


def test_validate_ohlcv_dataframe_missing_columns(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that missing required columns raise PriceValidationError."""
    incomplete_df = sample_valid_ohlcv_df.drop(columns=["Volume"])
    with pytest.raises(PriceValidationError, match="Missing required price columns"):
        validate_ohlcv_dataframe(incomplete_df)


def test_validate_ohlcv_dataframe_duplicates(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that duplicate timestamps raise PriceValidationError."""
    duplicate_row = sample_valid_ohlcv_df.iloc[[0]]
    df_with_dups = pd.concat([sample_valid_ohlcv_df, duplicate_row])
    with pytest.raises(PriceValidationError, match="Duplicate timestamps detected"):
        validate_ohlcv_dataframe(df_with_dups)


def test_validate_ohlcv_dataframe_nan_values(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that missing/NaN values raise PriceValidationError."""
    df_with_nan = sample_valid_ohlcv_df.copy()
    df_with_nan.iloc[1, df_with_nan.columns.get_loc("Close")] = np.nan
    with pytest.raises(PriceValidationError, match="missing/NaN"):
        validate_ohlcv_dataframe(df_with_nan)


@pytest.mark.parametrize(
    "col,bad_value",
    [
        ("Open", -10.0),
        ("Open", 0.0),
        ("High", -5.0),
        ("High", 0.0),
        ("Low", -100.0),
        ("Low", 0.0),
        ("Close", -50.0),
        ("Close", 0.0),
    ],
)
def test_validate_ohlcv_dataframe_non_positive_price(
    sample_valid_ohlcv_df: pd.DataFrame, col: str, bad_value: float
):
    """Verifies that zero or negative prices trigger PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    df_invalid.iloc[0, df_invalid.columns.get_loc(col)] = bad_value
    with pytest.raises(PriceValidationError, match="Non-positive price found"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_negative_volume(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that negative volume triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    df_invalid.iloc[0, df_invalid.columns.get_loc("Volume")] = -1.0
    with pytest.raises(PriceValidationError, match="Negative volume found"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_invalid_high_low(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that High < Low triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    # Set High lower than Low
    df_invalid.iloc[0, df_invalid.columns.get_loc("High")] = 48000.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Low")] = 49000.0
    with pytest.raises(PriceValidationError, match="Invalid High < Low"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_invalid_high_open(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that High < Open triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    df_invalid.iloc[0, df_invalid.columns.get_loc("High")] = 49999.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Open")] = 50000.0
    with pytest.raises(PriceValidationError, match="Invalid High < Open"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_invalid_high_close(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that High < Close triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    df_invalid.iloc[0, df_invalid.columns.get_loc("High")] = 50400.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Close")] = 50500.0
    with pytest.raises(PriceValidationError, match="Invalid High < Close"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_invalid_low_open(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that Low > Open triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    df_invalid.iloc[0, df_invalid.columns.get_loc("Low")] = 50001.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Open")] = 50000.0
    with pytest.raises(PriceValidationError, match="Invalid Low > Open"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_invalid_low_close(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that Low > Close triggers PriceValidationError."""
    df_invalid = sample_valid_ohlcv_df.copy()
    # Ensure High >= Open >= Low > Close
    df_invalid.iloc[0, df_invalid.columns.get_loc("High")] = 52000.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Open")] = 51000.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Low")] = 50500.0
    df_invalid.iloc[0, df_invalid.columns.get_loc("Close")] = 50000.0
    with pytest.raises(PriceValidationError, match="Invalid Low > Close"):
        validate_ohlcv_dataframe(df_invalid)


def test_validate_ohlcv_dataframe_multiindex_columns(sample_valid_ohlcv_df: pd.DataFrame):
    """Verifies that yfinance-style MultiIndex columns are flattened properly."""
    multi_df = sample_valid_ohlcv_df.copy()
    multi_df.columns = pd.MultiIndex.from_tuples(
        [(c, "BTC-USD") for c in multi_df.columns], names=["Price", "Ticker"]
    )
    validated = validate_ohlcv_dataframe(multi_df)
    assert list(validated.columns) == ["Open", "High", "Low", "Close", "Volume"]


@patch("yfinance.download")
def test_fetch_historical_price_data_mocked_success(
    mock_yf_download: MagicMock, sample_valid_ohlcv_df: pd.DataFrame
):
    """Verifies fetch_historical_price_data with a mocked successful yfinance return."""
    mock_yf_download.return_value = sample_valid_ohlcv_df

    df = fetch_historical_price_data(
        symbol="BTC-USD",
        lookback_days=180,
        interval="1d",
        end_date=datetime(2026, 1, 6, tzinfo=timezone.utc),
    )

    assert len(df) == 5
    assert mock_yf_download.called
    call_kwargs = mock_yf_download.call_args[1]
    assert call_kwargs["tickers"] == "BTC-USD"
    assert call_kwargs["interval"] == "1d"


@patch("yfinance.download")
def test_fetch_historical_price_data_mocked_empty(mock_yf_download: MagicMock):
    """Verifies that empty yfinance response raises PriceIngestionError."""
    mock_yf_download.return_value = pd.DataFrame()

    with pytest.raises(PriceIngestionError, match="returned an empty dataset"):
        fetch_historical_price_data(symbol="BTC-USD", lookback_days=30)


@patch("yfinance.download")
def test_fetch_historical_price_data_mocked_exception(mock_yf_download: MagicMock):
    """Verifies that yfinance network/download exceptions are wrapped in PriceIngestionError."""
    mock_yf_download.side_effect = ConnectionError("Network unreachable")

    with pytest.raises(PriceIngestionError, match="Failed to download price data"):
        fetch_historical_price_data(symbol="BTC-USD", lookback_days=30)


def test_fetch_historical_price_data_invalid_lookback():
    """Verifies that non-positive lookback_days raises ValueError."""
    with pytest.raises(ValueError, match="lookback_days must be positive"):
        fetch_historical_price_data(lookback_days=0)


def test_save_price_data(sample_valid_ohlcv_df: pd.DataFrame, tmp_path: Path):
    """Verifies CSV persistence and formatting."""
    target_csv = tmp_path / "test_btc.csv"
    saved = save_price_data(sample_valid_ohlcv_df, output_path=target_csv)

    assert saved.exists()
    read_back = pd.read_csv(saved, index_col="Date")
    assert len(read_back) == len(sample_valid_ohlcv_df)
    assert list(read_back.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_config_valid():
    """Verifies reading project configuration."""
    config = load_config("config/config.yaml")
    assert "asset" in config
    assert config["asset"]["symbol"] == "BTC-USD"
    assert config["asset"]["lookback_days"] == 180


def test_load_config_missing(tmp_path: Path):
    """Verifies that a missing config file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


@patch("src.ingestion.price_data.fetch_historical_price_data")
def test_ingest_price_pipeline_mocked(
    mock_fetch: MagicMock, sample_valid_ohlcv_df: pd.DataFrame, tmp_path: Path
):
    """Verifies the complete orchestration pipeline with mocked data fetching."""
    mock_fetch.return_value = sample_valid_ohlcv_df
    target_csv = tmp_path / "output_btc.csv"

    df, saved_path = ingest_price_pipeline(
        config_path="config/config.yaml",
        output_override=target_csv,
    )

    assert len(df) == 5
    assert saved_path == target_csv.resolve()
    assert saved_path.exists()
