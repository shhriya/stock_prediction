import yfinance as yf
import datetime
import pandas as pd
import pytz
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google Application Credentials path from the environment variable
google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set Google Cloud credentials manually
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials

# Initialize BigQuery client
client = bigquery.Client()

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("BQ_TABLE")


# List of tickers
tickers = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    "NFLX", "META", "JPM", "AMZN", "BRK-B",
    "UNH", "V", "MA", "PG", "DIS",
    "ADBE", "INTC", "PFE", "CRM", "KO"
]

def fetch_stock_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data = data.reset_index()
        data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        }, inplace=True)
        data['symbol'] = symbol
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        india = pytz.timezone("Asia/Kolkata")
        data['inserted_at'] = datetime.datetime.now(india)
        data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'inserted_at']]
        print(f"Fetched {len(data)} records for {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def insert_into_bigquery(dataframe, project_id, dataset_id, table_id):
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        job_config.source_format = bigquery.SourceFormat.CSV
        job = client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
        job.result()
        print(f"Inserted {len(dataframe)} rows into {project_id}.{dataset_id}.{table_id}")
    except Exception as e:
        print(f"Error inserting data into BigQuery: {e}")

def main():
    start_date = '2023-05-05'
    end_date = '2025-05-05'
    all_dataframes = []
    for symbol in tickers:
        print(f"\nFetching stock data for {symbol} from {start_date} to {end_date}...")
        df = fetch_stock_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            all_dataframes.append(df)
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("\nInserting combined data into BigQuery...")
        insert_into_bigquery(combined_df, PROJECT_ID, DATASET_ID, TABLE_ID)
        print(f"\nTotal records processed: {len(combined_df)}")
    else:
        print("No data fetched for any ticker.")

if __name__ == "__main__":
    main()




# import requests
# import yfinance as yf
# import datetime
# import pandas as pd
# import pytz
# from google.cloud import bigquery
# import os
# # from dotenv import load_dotenv
 
# # Load environment variables from .env file
# # load_dotenv()
 
# # Get the Google Application Credentials path from the environment variable
# google_application_credentials = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../credentials.json")

# # Set Google Cloud credentials manually
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials

# # Initialize BigQuery client using the credentials path from the environment variable
# client = bigquery.Client.from_service_account_json(google_application_credentials)
 
# # Configuration
# PROJECT_ID = "stock-pricing-458605"
# DATASET_ID = "stock_dataset"
# TABLE_ID = "stock_data"
 
# # List of tickers
# tickers = [
#     "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
#     "NFLX", "META", "JPM", "AMZN", "BRK-B",
#     "UNH", "V", "MA", "PG", "DIS",
#     "ADBE", "INTC", "PFE", "CRM", "KO"
# ]
 
# def fetch_stock_data(symbol, start_date, end_date):
#     """Fetch historical stock data for a given symbol and date range."""
#     try:
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(start=start_date, end=end_date)
#         data = data.reset_index()
 
#         data.rename(columns={
#             'Date': 'date',
#             'Open': 'open',
#             'High': 'high',
#             'Low': 'low',
#             'Close': 'close'
#         }, inplace=True)
 
#         data['symbol'] = symbol
#         data['date'] = data['date'].dt.strftime('%Y-%m-%d')
#         india = pytz.timezone("Asia/Kolkata")
#         data['inserted_at'] = datetime.datetime.now(india)
 
#         data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'inserted_at']]
#         print(f"Fetched {len(data)} records for {symbol}")
#         return data
 
#     except Exception as e:
#         print(f"Error fetching data for {symbol}: {e}")
#         return None
 
# def insert_into_bigquery(dataframe, project_id, dataset_id, table_id):
#     """Insert dataframe into BigQuery table."""
#     try:
#         table_ref = client.dataset(dataset_id).table(table_id)
#         job_config = bigquery.LoadJobConfig()
#         job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
#         job_config.source_format = bigquery.SourceFormat.CSV
 
#         job = client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
#         job.result()
 
#         print(f"Inserted {len(dataframe)} rows into {project_id}.{dataset_id}.{table_id}")
 
#     except Exception as e:
#         print(f"Error inserting data into BigQuery: {e}")
 
# def main():
#     start_date = '2023-05-05'
#     end_date = '2025-05-05'
 
#     all_dataframes = []
 
#     for symbol in tickers:
#         print(f"\nFetching stock data for {symbol} from {start_date} to {end_date}...")
#         df = fetch_stock_data(symbol, start_date, end_date)
#         if df is not None and not df.empty:
#             all_dataframes.append(df)
 
#     if all_dataframes:
#         combined_df = pd.concat(all_dataframes, ignore_index=True)
#         print("\nInserting combined data into BigQuery...")
#         insert_into_bigquery(combined_df, PROJECT_ID, DATASET_ID, TABLE_ID)
#         print(f"\nTotal records processed: {len(combined_df)}")
#     else:
#         print("No data fetched for any ticker.")
 
# if __name__ == "__main__":
#     main()