import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

# Patch environment before importing producer
@patch.dict(os.environ, {
    'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
    'KAFKA_TOPIC': 'stock-data',
    'TICKERS': 'AAPL,GOOG',
    'SLEEP_INTERVAL': '1'
})
class TestProducer(unittest.TestCase):

    @patch('producer.create_producer')
    @patch('producer.yf.Ticker')
    @patch('producer.time.sleep', return_value=None)
    def test_fetch_and_send_stock_data(self, mock_sleep, MockTicker, mock_create_producer):
        import producer  # Must import after environment patch

        mock_producer = MagicMock()
        mock_create_producer.return_value = mock_producer

        mock_ticker = MagicMock()
        MockTicker.return_value = mock_ticker

        mock_data = pd.DataFrame([{
            'Open': 100.0, 'High': 102.0, 'Low': 98.0, 'Close': 101.0, 'Volume': 100000
        }], index=[pd.to_datetime('2025-05-06 15:30:00')])
        mock_ticker.history.return_value = mock_data

        producer.main_produce()

        self.assertEqual(mock_producer.send.call_count, 2)

    @patch('producer.create_producer')
    @patch('producer.yf.Ticker')
    @patch('producer.time.sleep', return_value=None)
    def test_error_handling_in_fetch(self, mock_sleep, MockTicker, mock_create_producer):
        import producer  # Must import after environment patch

        mock_create_producer.return_value = MagicMock()
        mock_ticker = MagicMock()
        MockTicker.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("API Error")

        with patch('builtins.print') as mock_print:
            producer.main_produce()
            mock_print.assert_any_call("Error fetching data for AAPL: API Error")

if __name__ == '__main__':
    unittest.main()









# import unittest
# from unittest.mock import patch, MagicMock
# import pandas as pd
# import producer  # Your actual Kafka-producing script
# import os

# class TestProducer(unittest.TestCase):

#     @patch('producer.yf.Ticker')
#     @patch('producer.KafkaProducer')
#     @patch('producer.time.sleep', return_value=None)
#     @patch.dict(os.environ, {
#         'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
#         'KAFKA_TOPIC': 'stock-data',
#         'TICKERS': 'AAPL,GOOG',
#         'SLEEP_INTERVAL': '5'
#     })
#     def test_fetch_and_send_stock_data(self, mock_sleep, MockKafkaProducer, MockTicker):
#         # Mock KafkaProducer
#         mock_producer = MagicMock()
#         MockKafkaProducer.return_value = mock_producer

#         # Mock yfinance Ticker
#         mock_ticker = MagicMock()
#         MockTicker.return_value = mock_ticker

#         # Mock historical data
#         mock_data = pd.DataFrame([{
#             'Open': 100.0, 'High': 102.0, 'Low': 98.0, 'Close': 101.0, 'Volume': 100000
#         }], index=[pd.to_datetime('2025-05-06 15:30:00')])
#         mock_ticker.history.return_value = mock_data

#         # Run main producer logic
#         producer.main_produce()

#         # Assert that send is called twice (once for each ticker)
#         self.assertEqual(mock_producer.send.call_count, 2)

#     @patch('producer.yf.Ticker')
#     @patch('producer.time.sleep', return_value=None)
#     @patch.dict(os.environ, {
#         'TICKERS': 'AAPL',
#     })
#     def test_error_handling_in_fetch(self, mock_sleep, MockTicker):
#         mock_ticker = MagicMock()
#         MockTicker.return_value = mock_ticker
#         mock_ticker.history.side_effect = Exception("API Error")

#         with patch('builtins.print') as mock_print:
#             producer.main_produce()
#             mock_print.assert_any_call("Error fetching data for AAPL: API Error")

# if __name__ == '__main__':
#     unittest.main()


















# import unittest
# from unittest.mock import patch, MagicMock
# import pandas as pd
# from datetime import datetime
# import producer  # Import your producer module directly
# from unittest.mock import patch, MagicMock

# @patch("producer.KafkaProducer")
# def test_producer_init(mock_kafka_producer):
#     mock_kafka_producer.return_value = MagicMock()
# import producer  # Re-import inside test to trigger KafkaProducer with mock



# from unittest.mock import patch
# import pytest

# @patch("producer.KafkaProducer")
# def test_producer_init(mock_kafka_producer):
#     from producer import producer  # triggers creation using mocked KafkaProducer
#     mock_kafka_producer.assert_called_once()




# class TestProducer(unittest.TestCase):

#     @patch('producer.yf.Ticker')
#     @patch('producer.KafkaProducer')
#     @patch('producer.time.sleep', return_value=None)
#     @patch.dict('os.environ', {
#         'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
#         'KAFKA_TOPIC': 'stock-data',
#         'TICKERS': 'AAPL,GOOG',
#         'SLEEP_INTERVAL': '5'
#     })
#     def test_fetch_and_send_stock_data(self, mock_sleep, MockKafkaProducer, MockTicker):
#         # Setup mock KafkaProducer
#         mock_producer = MagicMock()
#         MockKafkaProducer.return_value = mock_producer

#         # Setup mock yfinance.Ticker
#         mock_ticker = MagicMock()
#         MockTicker.return_value = mock_ticker

#         # Create dummy DataFrame
#         mock_data = pd.DataFrame([{
#             'Open': 100.0, 'High': 102.0, 'Low': 98.0, 'Close': 101.0, 'Volume': 100000
#         }], index=[pd.to_datetime('2025-05-06 15:30:00')])
#         mock_ticker.history.return_value = mock_data

#         # Run producer logic
#         producer.main_produce()

#         # Verify Kafka send called correctly
#         self.assertEqual(mock_producer.send.call_count, 2)  # Since we mock 2 tickers

#     @patch('producer.yf.Ticker')
#     @patch('producer.time.sleep', return_value=None)
#     @patch.dict('os.environ', {
#         'TICKERS': 'AAPL',
#     })
#     def test_error_handling_in_fetch(self, mock_sleep, MockTicker):
#         mock_ticker = MagicMock()
#         MockTicker.return_value = mock_ticker
#         mock_ticker.history.side_effect = Exception("API Error")

#         with patch('builtins.print') as mock_print:
#             producer.main_produce()
#             mock_print.assert_any_call("Error fetching data for AAPL: API Error")

# if __name__ == '__main__':
#     unittest.main()
