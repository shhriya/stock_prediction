import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from consumer import process_message, write_to_bigtable, get_row_key

class TestConsumer(unittest.TestCase):

    @patch('consumer.bigtable.Client')
    def test_process_message_success(self, MockBigtableClient):
        mock_client = MagicMock()
        MockBigtableClient.return_value = mock_client
        mock_instance = MagicMock()
        mock_table = MagicMock()
        mock_row = MagicMock()

        mock_client.instance.return_value = mock_instance
        mock_instance.table.return_value = mock_table
        mock_table.direct_row.return_value = mock_row

        message = {
            "symbol": "GOOG",
            "date": "2025-05-08",
            "open": 1500.0,
            "high": 1510.0,
            "low": 1495.0,
            "close": 1505.0
        }

        process_message(message)

        mock_row.set_cell.assert_any_call('cf1', b'symbol', b'GOOG')
        mock_row.set_cell.assert_any_call('cf1', b'inserted_at', unittest.mock.ANY)

    def test_get_row_key(self):
        data = {"symbol": "AAPL", "date": "2025-05-08"}
        row_key = get_row_key(data)
        self.assertIn(b"AAPL", row_key)
        self.assertIn(b"2025-05-08", row_key)

    @patch('consumer.bigtable.Client')
    def test_write_to_bigtable_failure(self, MockBigtableClient):
        mock_client = MagicMock()
        MockBigtableClient.return_value = mock_client
        mock_instance = MagicMock()
        mock_table = MagicMock()
        mock_client.instance.return_value = mock_instance
        mock_instance.table.return_value = mock_table
        mock_table.direct_row.side_effect = Exception("BT failure")

        data = {
            "symbol": "AMZN",
            "date": "2025-05-08",
            "open": 1800.0,
            "high": 1810.0,
            "low": 1795.0,
            "close": 1805.0
        }

        result = write_to_bigtable(data)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
