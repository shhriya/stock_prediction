# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from google.cloud import bigquery
from google.oauth2 import service_account
 
CONFIG = {
      "bigquery": {
        "project_id": "stock-pricing-458605",
        "dataset_id": "stock_dataset",
        "table_id": "stock_data",
        "credentials_path": "credentials/google-credentials.json"
    },
    'plots': {
        'color_actual': 'blue',
        'xticks_interval': 10
    }
}
 
 
def get_bigquery_client(config):
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
        CAST(Date AS DATE) as date,
        Open as open_price,
        High as high_price,
        Low as low_price,
        Close as close_price,
    FROM `{config['bigquery']['project_id']}.{config['bigquery']['dataset_id']}.{config['bigquery']['table_id']}`
    WHERE symbol = @ticker_symbol
    AND CAST(Date AS DATE) BETWEEN DATE(@start_date) AND DATE(@end_date)
    ORDER BY date ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker_symbol", "STRING", ticker),
            bigquery.ScalarQueryParameter("start_date", "STRING", start_date.strftime('%Y-%m-%d')),
            bigquery.ScalarQueryParameter("end_date", "STRING", end_date.strftime('%Y-%m-%d')),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df
 
 
def is_stationary(series):
    return adfuller(series)[1] < 0.05
 
 
def decompose_series(series, model='additive', period=5):
    return seasonal_decompose(series, model=model, period=period)
 
 
def fit_sarima_model(series, p=1, d=1, q=2, seasonal_order=7):
    model = sm.tsa.statespace.SARIMAX(series, order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    return model.fit()
 
 
def forecast(model, start, end):
    predictions = model.get_prediction(start=start, end=end).predicted_mean
    return predictions
 
 
def prepare_forecast_dataframe(predictions, start_date):
    predictions.index = pd.date_range(start=start_date, periods=len(predictions), freq='D')
    df = pd.DataFrame(predictions)
    df.insert(0, "date", df.index)
    return df.reset_index(drop=True)
 