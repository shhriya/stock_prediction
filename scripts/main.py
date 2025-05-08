import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_market_calendars as mcal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
    'bigquery': {
        'credentials_path': os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "project_id": os.getenv("PROJECT_ID"),
        "dataset_id": os.getenv("DATASET_ID"),
        "table_id": os.getenv("BQ_TABLE"),
    },
    'plots': {
        'color_actual': 'blue',
        'xticks_interval': 10
    }
}
 
 
def get_bigquery_client(config):
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials = service_account.Credentials.from_service_account_file(
        config['bigquery']['credentials_path'],
        scopes=['https://www.googleapis.com/auth/bigquery']
    )
    return bigquery.Client(credentials=credentials, project=config['bigquery']['project_id'])
 
 
def get_date_range(config):
    client = get_bigquery_client(config)
    query = f"""
    SELECT MIN(Date) as min_date, MAX(Date) as max_date
    FROM `{config['bigquery']['project_id']}.{config['bigquery']['dataset_id']}.{config['bigquery']['table_id']}`
    """
    result = client.query(query).result()
    row = next(result)
    return row.min_date, row.max_date
 
 
def download_data(config, ticker, start_date, end_date):
    client = get_bigquery_client(config)
    query = f"""
SELECT
    DATE(Date) as date,
    Open as open_price,
    High as high_price,
    Low as low_price,
    Close as close_price
FROM `{config['bigquery']['project_id']}.{config['bigquery']['dataset_id']}.{config['bigquery']['table_id']}`
WHERE symbol = @ticker_symbol
AND DATE(Date) BETWEEN @start_date AND @end_date
ORDER BY date ASC
"""

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker_symbol", "STRING", ticker),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.strftime('%Y-%m-%d')),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.strftime('%Y-%m-%d')),
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df
 
 
def is_stationary(series):
    return adfuller(series)[1] < 0.05
 
 
def decompose_series(series, model='additive', period=5):
    return seasonal_decompose(series, model=model, period=period)
 
 
def add_technical_indicators(data, price_col):
    """Add enhanced technical indicators with improved trend detection"""
    df = data.copy()
   
    try:
        # Long-term trend indicators
        df['Return_1d'] = df[price_col].pct_change()
        df['Return_5d'] = df[price_col].pct_change(periods=5)
        df['Return_20d'] = df[price_col].pct_change(periods=20)
        df['Return_60d'] = df[price_col].pct_change(periods=60)  # Added longer-term return
       
        # Enhanced moving averages
        for window in [5, 10, 20, 50, 100, 200]:  # Added 200-day MA
            df[f'SMA{window}'] = df[price_col].rolling(window=window).mean()
            df[f'EMA{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
            # Add distance from MA as percentage
            df[f'DistanceFromSMA{window}'] = (df[price_col] - df[f'SMA{window}']) / df[f'SMA{window}'] * 100
       
        # Multi-timeframe trend strength
        df['TrendStrength_ST'] = df['DistanceFromSMA20'].rolling(window=5).mean()
        df['TrendStrength_MT'] = df['DistanceFromSMA50'].rolling(window=20).mean()
        df['TrendStrength_LT'] = df['DistanceFromSMA200'].rolling(window=50).mean()
       
        # Price momentum with acceleration
        df['Momentum_ST'] = df[price_col].diff(5)
        df['Momentum_MT'] = df[price_col].diff(20)
        df['Momentum_LT'] = df[price_col].diff(60)
       
        # Momentum change rate (acceleration)
        df['Acceleration_ST'] = df['Momentum_ST'].diff()
        df['Acceleration_MT'] = df['Momentum_MT'].diff()
       
        # Volatility measures with trend adjustment
        for window in [10, 20, 50]:
            # Trend-adjusted volatility
            returns = df['Return_1d']
            trend = returns.rolling(window=window).mean()
            df[f'TrendAdjVolatility_{window}d'] = (returns - trend).rolling(window=window).std() * np.sqrt(252)
           
            # Directional volatility
            up_vol = returns.where(returns > 0, 0).rolling(window=window).std() * np.sqrt(252)
            down_vol = returns.where(returns < 0, 0).rolling(window=window).std() * np.sqrt(252)
            df[f'UpVolatility_{window}d'] = up_vol
            df[f'DownVolatility_{window}d'] = down_vol
       
        # Price momentum
        df['Momentum_1m'] = df[price_col] / df[price_col].shift(20) - 1
        df['Momentum_3m'] = df[price_col] / df[price_col].shift(60) - 1
       
        # Rate of change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df[price_col].pct_change(period)
       
        # MACD
        exp1 = df[price_col].ewm(span=12, adjust=False).mean()
        exp2 = df[price_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
       
    except Exception as e:
        print(f"Error in add_technical_indicators: {str(e)}")
        return data
   
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
   
    # Forward fill then backward fill NaN values
    df = df.ffill().bfill()
   
    # Normalize numeric features only
    numeric_cols = [col for col in df.columns if col != price_col and col != 'date'
                   and np.issubdtype(df[col].dtype, np.number)]
   
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
   
    # Handle missing values
    df = df.ffill().bfill()
   
    # Clip extreme values to 3 standard deviations for numeric columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 3*std, mean + 3*std)
   
    return df
 
def normalize_series(series, method='zscore'):
    """Normalize the series using various methods
   
    Args:
        series: pandas Series to normalize
        method: 'zscore' for standard normalization, 'minmax' for 0-1 scaling,
                'robust' for median/IQR scaling
    """
    if method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series - min_val
        return (series - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = series.median()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return series - median
        return (series - median) / iqr
    else:  # zscore
        std = series.std()
        if std == 0:
            return series - series.mean()
        return (series - series.mean()) / std
 
def grid_search_sarima(data, exog=None):
    """Find optimal SARIMA parameters through grid search with cross-validation"""
    best_rmse = float('inf')
    best_params = None
   
    # Optimized parameter grid for accuracy
    param_grid = {
        'p': [2],       # AR(2) for better trend capture
        'd': [1],       # First difference for stationarity
        'q': [2],       # MA(2) for error correction
        'P': [1],       # Seasonal AR for periodic patterns
        'D': [1],       # Seasonal differencing
        'Q': [1],       # Seasonal MA
        's': [5]        # Weekly seasonality
    }
   
    # Model configuration for better accuracy
    model_config = {
        'enforce_stationarity': True,
        'enforce_invertibility': True,
        'concentrate_scale': True,
        'disp': False,
        'maxiter': 500,
        'method': 'lbfgs',
        'optim_score': 'harvey',
        'simple_differencing': True
    }
   
    # Validate input data
    if data.isnull().any():
        data = data.fillna(method='ffill').fillna(method='bfill')
   
    # Remove any remaining NaN or inf values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
   
    # Use last 15% of data for validation
    val_size = int(len(data) * 0.15)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
   
    if exog is not None:
        train_exog = exog[:-val_size]
        val_exog = exog[-val_size:]
   
    # Simplified validation - use single train/test split
    test_size = int(len(data) * 0.2)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
   
    print("Starting parameter search with single validation split...")
   
    # Single set of parameters
    p = param_grid['p'][0]
    d = param_grid['d'][0]
    q = param_grid['q'][0]
    P = param_grid['P'][0]
    D = param_grid['D'][0]
    Q = param_grid['Q'][0]
    s = param_grid['s'][0]
   
    try:
        # Fit model with enhanced configuration
        model = sm.tsa.statespace.SARIMAX(
            train_data,
            exog=train_exog if exog is not None else None,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            **model_config
        )
       
        # Fit with optimized settings
        results = model.fit(
            start_params=None,
            method='lbfgs',        # Faster optimization
            maxiter=50,           # Limit iterations
            optim_score='harvey', # Faster score computation
            low_memory=True,      # Memory optimization
            **model_config
        )
        predictions = results.get_forecast(steps=len(test_data))
        pred_mean = predictions.predicted_mean
       
        # Calculate RMSE
        rmse = np.sqrt(((test_data - pred_mean) ** 2).mean())
        best_rmse = rmse
        best_params = (p, d, q, P, D, Q, s)
        print(f"Model performance - RMSE: {rmse:.2f}")
   
    except Exception as e:
        print(f"Error fitting model: {str(e)}")
        # Use default parameters
        best_params = (1, 1, 0, 0, 0, 0, 5)
        best_rmse = float('inf')
   
    print(f"Best parameters found: SARIMA{best_params} with score: {best_rmse:.2f}")
    return best_params
 
def fit_sarima_model(data, target_col, p=None, d=None, q=None, P=None, D=None, Q=None, s=None):
    print(f"Total data points: {len(data)}")
   
    # Add enhanced technical indicators
    data_with_features = add_technical_indicators(data.copy(), target_col)
   
    # Use 80% of the data for training
    train_size = int(len(data) * 0.8)
   
    # Calculate trend and volatility features
    price = data_with_features[target_col]
   
    # Decompose trend using different windows
    windows = [5, 10, 20, 50]
    for window in windows:
        # Trend
        data_with_features[f'Trend_{window}d'] = price.rolling(window=window).mean()
        # Trend direction
        data_with_features[f'Trend_Direction_{window}d'] = (data_with_features[f'Trend_{window}d'].diff() > 0).astype(int)
        # Distance from trend
        data_with_features[f'Trend_Distance_{window}d'] = (price - data_with_features[f'Trend_{window}d']) / data_with_features[f'Trend_{window}d']
        # Volatility
        data_with_features[f'Volatility_{window}d'] = price.rolling(window=window).std() / price.rolling(window=window).mean()
   
    # Momentum features
    for period in [5, 10, 20]:
        # ROC (Rate of Change)
        data_with_features[f'ROC_{period}d'] = price.pct_change(periods=period)
        # Acceleration
        data_with_features[f'Acceleration_{period}d'] = data_with_features[f'ROC_{period}d'].diff()
   
    # Select key features
    feature_cols = [
        'Trend_5d', 'Trend_20d', 'Trend_50d',           # Multi-timeframe trends
        'Trend_Direction_5d', 'Trend_Direction_20d',     # Trend directions
        'Trend_Distance_5d', 'Trend_Distance_20d',      # Price-trend relationships
        'Volatility_5d', 'Volatility_20d',              # Volatility measures
        'ROC_5d', 'ROC_20d',                           # Price momentum
        'Acceleration_5d', 'Acceleration_20d'           # Momentum changes
    ]
   
    # Clean and prepare features
    exog_data = data_with_features[feature_cols].copy()
   
    # Handle infinite values
    exog_data = exog_data.replace([np.inf, -np.inf], np.nan)
   
    # Forward fill then backward fill NaN values
    exog_data = exog_data.ffill().bfill()
   
    # Scale all features using z-score normalization
    for col in feature_cols:
        mean = exog_data[col].mean()
        std = exog_data[col].std()
        if std > 0:
            exog_data[col] = (exog_data[col] - mean) / std
        else:
            # If std is 0, just center the data
            exog_data[col] = exog_data[col] - mean
   
    # Remove any remaining NaN rows
    valid_idx = ~exog_data.isna().any(axis=1)
    exog_data = exog_data[valid_idx]
    train_target = data_with_features[target_col][valid_idx]
   
    # Normalize features with robust scaling
    normalized_exog = pd.DataFrame(index=exog_data.index)
    for col in feature_cols:
        series = exog_data[col]
        # Use robust scaling to handle outliers
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            normalized_exog[col] = (series - series.median())
        else:
            normalized_exog[col] = (series - series.median()) / iqr
   
    # Split into train/test
    train_target = train_target[:train_size]
    normalized_train = normalize_series(train_target, method='robust')
   
    # Store original scale parameters for denormalization
    median = train_target.median()
    q1, q3 = train_target.quantile([0.25, 0.75])
    iqr = q3 - q1
   
    print(f"Training on {len(train_target)} data points with {len(feature_cols)} features")
   
    # Default parameters if not provided (trend-focused model)
    if any(param is None for param in [p, d, q, P, D, Q, s]):
        # Use parameters optimized for trend following
        p = p or 3  # More AR terms to capture longer patterns
        d = d or 1  # First difference
        q = q or 2  # More MA terms for shock absorption
        P = P or 1  # Seasonal AR for cyclic patterns
        D = D or 1  # Seasonal difference
        Q = Q or 1  # Seasonal MA
        s = s or 10  # Two-week cycle
   
    print(f"Using SARIMA parameters: ({p},{d},{q})({P},{D},{Q}){s}")
   
    # Fit the model with seasonal components and exogenous variables
    model = sm.tsa.statespace.SARIMAX(
        normalized_train,
        exog=normalized_exog[:train_size],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
   
    # Fit with improved optimization settings
    fitted_model = model.fit(
        disp=False,
        method='lbfgs',  # More robust optimization method
        maxiter=50,      # Limit iterations
        optim_score='harvey',  # More stable score function
        optim_complex_step=True,  # More accurate derivatives
        optim_hessian='cs'  # Complex-step differentiation for Hessian
    )
    print("Model fitting completed")
   
    # Store enhanced data for later use
    fitted_model._train_median = median
    fitted_model._train_iqr = iqr
    fitted_model._exog = normalized_exog
    fitted_model._feature_cols = feature_cols
    fitted_model._seasonal_order = (P, D, Q, s)
   
    return fitted_model
 
 
def forecast(model, data, target_col, start, end):
    print(f"Forecasting from {start} to {end}")
    try:
        # Get the forecast using stored exogenous variables
        steps = end - start + 1
        forecast_obj = model.get_forecast(
            steps=steps,
            exog=model._exog[-steps:]
        )
        normalized_predictions = forecast_obj.predicted_mean
       
        # Get the last actual value for scaling
        last_actual = data[target_col].iloc[-1]
        first_pred = (normalized_predictions.iloc[0] * model._train_iqr) + model._train_median
        scale_factor = last_actual / first_pred
       
        # Scale predictions and confidence intervals
        predictions = ((normalized_predictions * model._train_iqr) + model._train_median) * scale_factor
       
        # Add confidence intervals with scaling
        conf_int = forecast_obj.conf_int()
        lower = ((conf_int.iloc[:, 0] * model._train_iqr) + model._train_median) * scale_factor
        upper = ((conf_int.iloc[:, 1] * model._train_iqr) + model._train_median) * scale_factor
       
        print(f"Generated {len(predictions)} predictions")
        return predictions, lower, upper
    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        return None, None, None
 
 
def prepare_forecast_dataframe(predictions, start_date):
    print(f"Preparing forecast DataFrame starting from {start_date}")
   
    if predictions is None or len(predictions) == 0:
        print("No predictions to process")
        return pd.DataFrame(columns=['date', 'predicted_mean'])
   
    print(f"Processing {len(predictions)} predictions")
   
    # Create initial date range
    date_range = pd.date_range(start=start_date, periods=len(predictions), freq='B')  # Use 'B' for business days
    print(f"Generated {len(date_range)} business days")
   
    # Create DataFrame with predictions
    df = pd.DataFrame({
        'date': date_range,
        'predicted_mean': predictions
    })
   
    print(f"Created DataFrame with {len(df)} rows")
    return df