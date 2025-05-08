from kafka import KafkaConsumer, TopicPartition
import json
from google.cloud import bigtable
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid
import time
import logging
 
 
 
 
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# Load environment variables
load_dotenv()
 
# Configuration
project_id = os.getenv("PROJECT_ID")
instance_id = os.getenv("INSTANCE_ID")
table_id = os.getenv("TABLE_ID")
 
# Set credentials for Google Cloud client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
 
# Column family names
STOCK_CF = "cf1"  # Using the existing column family
 
def get_row_key(data):
    """
    Generate a row key for Bigtable based on stock data.
    Format: symbol#date#uuid
    """
    # Extract data from the message
    symbol = data.get('symbol', 'UNKNOWN')
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
   
    # Create a unique identifier for the row
    unique_id = str(uuid.uuid4())[:8]
   
    # Create a row key that ensures good distribution and logical grouping
    row_key = f"{symbol}#{date}#{unique_id}".encode()
   
    return row_key
 
def write_to_bigtable(data):
    """Write stock data to Bigtable using the existing cf1 column family."""
    # Create a connection to the Bigtable instance and table
    client = bigtable.Client(project=project_id, admin=False)
    instance = client.instance(instance_id)
    table = instance.table(table_id)
   
    try:
        # Generate row key
        row_key = get_row_key(data)
       
        # Get the row (or create it if it doesn't exist)
        row = table.direct_row(row_key)
       
        # Add data to the row
        for key, value in data.items():
            # Skip null values
            if value is None:
                continue
           
            # Convert all values to string for storage
            if isinstance(value, (int, float, bool)):
                value = str(value)
           
            # Add cell to the column family
            row.set_cell(
                STOCK_CF,
                key.encode(),
                value.encode(),
            )
       
        # Add inserted_at timestamp
        inserted_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row.set_cell(
            STOCK_CF,
            b'inserted_at',
            inserted_at.encode(),
        )
       
        # Write the row to Bigtable
        row.commit()
        logger.info(f"Successfully wrote row {row_key.decode()} to Bigtable")
        return True
       
    except Exception as e:
        logger.error(f"Error writing to Bigtable: {e}")
        return False
 
def process_message(message):
    """Process a message from Kafka and write it to Bigtable."""
    try:
        # Log received message
        logger.info(f"Received message: {message}")
       
        # Add current timestamp to track when the message was received
        message['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
       
        # Write message to Bigtable
        success = write_to_bigtable(message)
       
        if success:
            logger.info("Message successfully processed and stored in Bigtable")
        else:
            logger.error("Failed to store message in Bigtable")
           
    except Exception as e:
        logger.error(f"Error processing message: {e}")
 
def main():
    MAX_MESSAGES = 20
    message_count = 0
    """Main function to consume Kafka messages and store in Bigtable."""
    # Define a unique consumer group ID based on timestamp to ensure fresh data
    unique_group_id = f'stock-group-{int(time.time())}'
   
    try:
        # Connect to Kafka Consumer with unique group ID
        consumer = KafkaConsumer(
            bootstrap_servers='localhost:9092',
            auto_offset_reset='latest',  
            enable_auto_commit=True,
            group_id=unique_group_id,  # Using unique group ID for fresh start
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        )
       
        # Subscribe to the topic
        consumer.subscribe(['stock-data'])
       
        # Seek to the end of each partition to ensure only new messages are consumed
        # This is a crucial step to guarantee we only get fresh data
        for partition in consumer.assignment():
            consumer.seek_to_end(partition)
            logger.info(f"Positioned at the end of partition {partition}")
       
        logger.info(f"Kafka consumer started with group ID: {unique_group_id}")
        logger.info("Waiting for fresh messages...")
       
        # Process messages
        while True:
            # Poll for messages
            message_batch = consumer.poll(timeout_ms=5000)
           
            if not message_batch:
                logger.info("No new messages. Waiting...")
                if message_count>=MAX_MESSAGES:
                    break
                time.sleep(3)
                continue
               
            # Process any received messages
            for topic_partition, messages in message_batch.items():
                for msg in messages:
                    logger.info(f"Processing fresh message from offset {msg.offset}")
                    process_message(msg.value)
                    message_count += 1
           
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean shutdown
        if 'consumer' in locals():
            consumer.close()
            logger.info("Kafka consumer closed")
 
if __name__ == "__main__":
    main()

