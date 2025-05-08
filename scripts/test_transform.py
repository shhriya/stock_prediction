import unittest
from unittest.mock import MagicMock
from bigtable_to_bigquery import bigtable_to_bigquery

def mock_row_data():
    mock_row = MagicMock()
    mock_row.cells = {
        "cf1": {
            b"symbol": [MagicMock(value=b"GOOG")],
            b"open": [MagicMock(value=b"1500.0")],
            b"high": [MagicMock(value=b"1510.0")],
            b"low": [MagicMock(value=b"1495.0")],
            b"close": [MagicMock(value=b"1505.0")],
            b"inserted_at": [MagicMock(value=b"2025-05-08 15:30:00")]
        }
    }
    return mock_row

class TestBigtableToBigQuery(unittest.TestCase):
    def test_valid_row(self):
        mock_row = mock_row_data()
        result = bigtable_to_bigquery(mock_row)
        self.assertEqual(result["symbol"], "GOOG")
        self.assertEqual(result["open"], 1500.0)
        self.assertEqual(result["high"], 1510.0)
        self.assertEqual(result["low"], 1495.0)
        self.assertEqual(result["close"], 1505.0)
        self.assertEqual(result["inserted_at"], "2025-05-08 15:30:00")

    def test_missing_cell(self):
        mock_row = mock_row_data()
        del mock_row.cells["cf1"][b"symbol"]
        result = bigtable_to_bigquery(mock_row)
        self.assertNotIn("symbol", result)

    def test_invalid_float(self):
        mock_row = mock_row_data()
        mock_row.cells["cf1"][b"open"][0].value = b"not_a_float"
        result = bigtable_to_bigquery(mock_row)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
