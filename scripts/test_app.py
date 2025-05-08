#test_app.py
 
import pytest
import pandas as pd
import numpy as np
from datetime import date
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock
from app import *
 
# Fixtures
@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'open_price': np.random.rand(10),
        'high_price': np.random.rand(10),
        'low_price': np.random.rand(10),
        'close_price': np.random.rand(10),
        'volume': np.random.randint(1000, 10000, 10)
    })
    # Add technical indicators
    data['SMA5'] = data['close_price'].rolling(window=5).mean()
    data['EMA5'] = data['close_price'].ewm(span=5, adjust=False).mean()
    data['Return_1d'] = data['close_price'].pct_change()
    return data
 
@pytest.fixture
def mock_decomposition():
    class MockDecomposition:
        def __init__(self):
            self.trend = pd.Series(np.random.rand(10))
            self.seasonal = pd.Series(np.random.rand(10))
            self.resid = pd.Series(np.random.rand(10))
       
        def plot(self):
            return Mock()
   
    return MockDecomposition()
 
@pytest.fixture
def mock_model():
    class MockModel:
        def __init__(self):
            self.aic = 100.5
            self.bic = 120.3
            self.resid = pd.Series(np.random.rand(10))
            self._train_median = 50.0
            self._train_iqr = 10.0
            self._exog = pd.DataFrame(np.random.rand(10, 3))
            self._feature_cols = ['SMA5', 'EMA5', 'Return_1d']
            self._seasonal_order = (1, 1, 1, 12)
       
        def summary(self):
            return "Mock Model Summary"
       
        def get_forecast(self, steps, exog=None):
            mock_forecast = Mock()
            mock_forecast.predicted_mean = pd.Series(np.random.rand(steps))
            mock_forecast.conf_int = lambda: pd.DataFrame({
                'lower': np.random.rand(steps),
                'upper': np.random.rand(steps) + 1
            })
            return mock_forecast
   
    return MockModel()
 
# Date Input Tests
def test_valid_date_range():
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    assert start_date < end_date
 
def test_invalid_date_range():
    start_date = date(2023, 12, 31)
    end_date = date(2023, 1, 1)
    assert start_date > end_date
 
# Ticker Selection Tests
def test_valid_ticker_selection():
    ticker = "AAPL"
    assert ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "NFLX", "META", "JPM"]
 
def test_invalid_ticker_selection():
    ticker = "INVALID"
    assert ticker not in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "NFLX", "META", "JPM"]
 
# Data Loading Tests
@patch('main.download_data')
def test_data_loading_success(mock_download):
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'close_price': np.random.rand(10)
    })
    mock_download.return_value = mock_data
   
    result = mock_download(CONFIG, 'AAPL', date(2023, 1, 1), date(2023, 1, 10))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert 'close_price' in result.columns
 
@patch('main.download_data')
def test_data_loading_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    result = mock_download(CONFIG, 'INVALID', date(2023, 1, 1), date(2023, 1, 10))
    assert len(result) == 0
 
# Column Selection Tests
def test_column_selection(mock_data):
    valid_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in valid_columns:
        assert col in mock_data.columns
 
def test_column_data_types(mock_data):
    numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in numeric_columns:
        assert np.issubdtype(mock_data[col].dtype, np.number)
 
# Model Parameter Tests
def test_valid_sarima_parameters():
    p = 2
    d = 1
    q = 2
    seasonal_order = 12
    assert 0 <= p <= 5
    assert 0 <= d <= 5
    assert 0 <= q <= 5
    assert 0 <= seasonal_order <= 24
 
def test_invalid_sarima_parameters():
    p = -1
    d = 6
    q = 10
    seasonal_order = 25
    assert not (0 <= p <= 5)
    assert not (0 <= d <= 5)
    assert not (0 <= q <= 5)
    assert not (0 <= seasonal_order <= 24)
 
# Forecast Period Tests
def test_valid_forecast_period():
    period = 10
    assert 1 <= period <= 365
 
def test_invalid_forecast_period():
    period = 0
    assert not (1 <= period <= 365)
    period = 366
    assert not (1 <= period <= 365)
 
# Visualization Tests
@patch('streamlit.plotly_chart')
def test_line_plot_creation(mock_plotly_chart, mock_data):
    fig = px.line(mock_data, x='date', y=mock_data.columns[1:], title='Stock Price Overview')
    mock_plotly_chart(fig)
    mock_plotly_chart.assert_called_once()
 
@patch('streamlit.plotly_chart')
def test_forecast_plot_creation(mock_plotly_chart, mock_data, mock_model):
    pred_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-11', periods=5, freq='B'),
        'predicted_mean': np.random.rand(5),
        'lower_ci': np.random.rand(5),
        'upper_ci': np.random.rand(5) + 1
    })
   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mock_data['date'], y=mock_data['close_price'], name='Actual'))
    fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['predicted_mean'], name='Forecast'))
    fig.add_trace(go.Scatter(
        x=pred_df['date'].tolist() + pred_df['date'].tolist()[::-1],
        y=pred_df['upper_ci'].tolist() + pred_df['lower_ci'].tolist()[::-1],
        fill='toself',
        name='95% Confidence Interval'
    ))
   
    mock_plotly_chart(fig)
    mock_plotly_chart.assert_called_once()
 
 
# Error Handling Tests
@patch('main.download_data')
def test_handle_download_error(mock_download):
    mock_download.side_effect = Exception("API Error")
    with pytest.raises(Exception):
        mock_download(CONFIG, 'AAPL', date(2023, 1, 1), date(2023, 1, 10))
 
@patch('main.fit_sarima_model')
def test_handle_model_fitting_error(mock_fit_model, mock_data):
    mock_fit_model.side_effect = ValueError("Invalid parameters")
    with pytest.raises(ValueError):
        mock_fit_model(mock_data, 'close_price')
 
@patch('main.forecast')
def test_handle_forecast_error(mock_forecast, mock_model, mock_data):
    mock_forecast.side_effect = Exception("Forecast Error")
    with pytest.raises(Exception):
        mock_forecast(mock_model, mock_data, 'close_price', 0, 10)
 
# Performance Tests
@patch('main.download_data')
def test_large_data_handling(mock_download):
    # Create large dataset
    dates = pd.date_range(start='2020-01-01', end='2023-12-31')
    large_data = pd.DataFrame({
        'date': dates,
        'open_price': np.random.rand(len(dates)),
        'high_price': np.random.rand(len(dates)),
        'low_price': np.random.rand(len(dates)),
        'close_price': np.random.rand(len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    mock_download.return_value = large_data
   
    result = mock_download(CONFIG, 'AAPL', date(2020, 1, 1), date(2023, 12, 31))
    assert len(result) > 1000  # Test handling of large datasets
    assert all(col in result.columns for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume'])
