 
#test_main.py
 
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import Mock, patch
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import google.auth.credentials
from main import (
    CONFIG, get_bigquery_client, get_date_range, download_data,
    is_stationary, decompose_series, fit_sarima_model,
    forecast, prepare_forecast_dataframe
)
 
# Fixtures
@pytest.fixture
def mock_bigquery_client():
    mock_client = Mock(spec=bigquery.Client)
    return mock_client
 
@pytest.fixture
def mock_credentials():
    mock_creds = Mock(spec=google.auth.credentials.Credentials)
    return mock_creds
 
@pytest.fixture
def sample_time_series():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    df = pd.DataFrame({
        'date': dates,
        'open_price': np.random.rand(len(dates)),
        'high_price': np.random.rand(len(dates)),
        'low_price': np.random.rand(len(dates)),
        'close_price': np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.1, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    return df
 
@pytest.fixture
def fitted_sarima_model(sample_time_series):
    model = fit_sarima_model(sample_time_series, 'close_price')
    return model
 
# BigQuery Client Tests
def test_get_bigquery_client_success(mock_credentials):
    with patch('google.oauth2.service_account.Credentials.from_service_account_file', return_value=mock_credentials):
        client = get_bigquery_client(CONFIG)
        assert isinstance(client, bigquery.Client)
 
def test_get_bigquery_client_invalid_credentials():
    with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
        mock_creds.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_bigquery_client(CONFIG)
 
# Date Range Tests
def test_get_date_range_success(mock_bigquery_client):
    expected_min_date = date(2020, 1, 1)
    expected_max_date = date(2023, 12, 31)
   
    # Create a proper mock row object
    class MockRow:
        def __init__(self, min_date, max_date):
            self.min_date = min_date
            self.max_date = max_date
   
    mock_row = MockRow(expected_min_date, expected_max_date)
   
    # Create a proper mock result that is both iterable and iterator
    class MockResult:
        def __init__(self):
            self._data = [mock_row]
            self._index = 0
 
        def __iter__(self):
            return self
 
        def __next__(self):
            if self._index >= len(self._data):
                raise StopIteration
            result = self._data[self._index]
            self._index += 1
            return result
   
    mock_result = MockResult()
    mock_bigquery_client.query.return_value.result.return_value = mock_result
   
    with patch('main.get_bigquery_client', return_value=mock_bigquery_client):
        min_date, max_date = get_date_range(CONFIG)
        assert min_date == expected_min_date
        assert max_date == expected_max_date
 
def test_get_date_range_empty_database(mock_bigquery_client):
    # Create a proper mock result that is both iterable and iterator (empty)
    class MockEmptyResult:
        def __init__(self):
            self._data = []
            self._index = 0
 
        def __iter__(self):
            return self
 
        def __next__(self):
            raise StopIteration
   
    mock_result = MockEmptyResult()
    mock_bigquery_client.query.return_value.result.return_value = mock_result
   
    with patch('main.get_bigquery_client', return_value=mock_bigquery_client):
        with pytest.raises(StopIteration):
            get_date_range(CONFIG)
 
# Data Download Tests
def test_download_data_success(mock_bigquery_client):
    mock_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'open_price': np.random.rand(10),
        'high_price': np.random.rand(10),
        'low_price': np.random.rand(10),
        'close_price': np.random.rand(10),
        'volume': np.random.randint(1000, 10000, 10)
    })
   
    mock_bigquery_client.query.return_value.to_dataframe.return_value = mock_df
   
    with patch('main.get_bigquery_client', return_value=mock_bigquery_client):
        result = download_data(
            CONFIG,
            'AAPL',
            date(2023, 1, 1),
            date(2023, 1, 10)
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert all(col in result.columns for col in ['date', 'open_price', 'close_price'])
 
def test_download_data_no_data(mock_bigquery_client):
    # Create empty DataFrame with expected columns
    mock_df = pd.DataFrame(columns=['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
    mock_bigquery_client.query.return_value.to_dataframe.return_value = mock_df
   
    with patch('main.get_bigquery_client', return_value=mock_bigquery_client):
        result = download_data(
            CONFIG,
            'INVALID',
            date(2023, 1, 1),
            date(2023, 1, 10)
        )
        assert len(result) == 0
        assert all(col in result.columns for col in ['date', 'open_price', 'close_price'])
 
# Time Series Analysis Tests
def test_is_stationary_with_stationary_data():
    # Create stationary data
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 100))
    assert is_stationary(data) == True
 
def test_is_stationary_with_non_stationary_data():
    # Create non-stationary data (trend)
    np.random.seed(42)
    trend = np.linspace(0, 10, 100)
    data = pd.Series(trend + np.random.normal(0, 1, 100))
    assert is_stationary(data) == False
 
 
def test_decompose_series_invalid_period():
    # Test with series shorter than period
    short_series = pd.Series(np.random.rand(5))
    with pytest.raises(ValueError):
        decompose_series(short_series, period=12)
 
# SARIMA Model Tests
def test_fit_sarima_model_success(sample_time_series):
    model = fit_sarima_model(sample_time_series, 'close_price')
    assert model is not None
    assert hasattr(model, '_train_median')
    assert hasattr(model, '_train_iqr')
    assert hasattr(model, '_exog')
    assert hasattr(model, '_feature_cols')
    assert hasattr(model, '_seasonal_order')
 
def test_prepare_forecast_dataframe_empty():
    predictions = pd.Series([])
    start_date = date(2023, 1, 1)
    result = prepare_forecast_dataframe(predictions, start_date)
    assert len(result) == 0
 
 
 
 
 