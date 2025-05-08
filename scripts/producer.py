import yfinance as yf
from kafka import KafkaProducer
import json
from datetime import datetime
import pytz
import time
 
# Setup timezone
IST = pytz.timezone('Asia/Kolkata')
 
# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
 
# List of 20 tickers
tickers = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    "NFLX", "META", "JPM", "AMZN", "BRK-B",
    "UNH", "V", "MA", "PG", "DIS",
    "ADBE", "INTC", "PFE", "CRM", "KO"
]
 
# Function to check if market is closed
def is_market_closed():
    now = datetime.now(IST)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now > market_close
 
# Optional: Uncomment to wait until market closes
# if not is_market_closed():
#     print("Market still open. Please run this after 3:30 PM IST.")
#     exit()
 
# Fetch latest data for each ticker and send to Kafka
time.sleep(20)
for ticker_symbol in tickers:
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d", interval="1m").tail(1)
        if not data.empty:
            row = data.iloc[0]
            record = {
                "symbol": ticker_symbol,
                "timestamp": data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"])
            }
            print("Sending to Kafka:", record)
            producer.send("stock-data", value=record)
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
    time.sleep(5)
 
# Close the producer
producer.flush()
producer.close()
print("All data sent. Producer exiting.")