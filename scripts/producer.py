
import yfinance as yf
from kafka import KafkaProducer
import json
from datetime import datetime
import pytz
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Setup timezone
IST = pytz.timezone('Asia/Kolkata')
 
# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
topic_name = os.getenv("KAFKA_TOPIC", "stock-data")
tickers = os.getenv("TICKERS", "").split(",")
sleep_interval = int(os.getenv("SLEEP_INTERVAL", 5))
 
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

def main_produce():
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
                producer.send(topic_name, value=record)
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol}: {e}")
        time.sleep(sleep_interval)

    producer.flush()
    producer.close()
    print("All data sent. Producer exiting.")

if __name__ == "__main__":
    main_produce()


    
 
# Close the producer
producer.flush()
producer.close()
print("All data sent. Producer exiting.")