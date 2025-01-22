import unittest
from unittest.mock import patch
from datetime import datetime

from alpaca_package.alpaca import (
    place_order,
    nearest_strike_price,
    get_third_friday,
    get_valid_expiration
)


class TestAlpacaFunctions(unittest.TestCase):

    @patch('strategy_analysis.alpaca.api')
    def test_place_order(self, mock_api):
        mock_api.submit_order.return_value = {'id': 'test_order_id'}
        result = place_order(
            symbol="TBT",
            strike_price=150.0,
            limit_price=1.5,
            expiration_date="2025-12-15",
            time_in_force="gtc",
            order_side="buy",
            order_type="limit"
        )
        self.assertIsNotNone(result)
        self.assertIn('id', result)
        self.assertEqual(result['id'], 'test_order_id')

    def test_nearest_strike_price(self):
        result = nearest_strike_price(152.75, 2.5)
        self.assertEqual(result, 152.5)
        result = nearest_strike_price(153.0, 2.5)
        self.assertEqual(result, 152.5)
        result = nearest_strike_price(154.0, 2.5)
        self.assertEqual(result, 155.0)

    def test_get_third_friday(self):
        result = get_third_friday(2023, 12)
        self.assertEqual(result, datetime(2023, 12, 15))
        result = get_third_friday(2024, 1)
        self.assertEqual(result, datetime(2024, 1, 19))

    @patch('strategy_analysis.alpaca.api')
    def test_get_valid_expiration(self, mock_api):
        mock_api.get_options_expirations.return_value = [
            '2023-12-01', '2023-12-08', '2023-12-15', '2023-12-22'
        ]
        target_date = datetime(2023, 12, 10)
        result = get_valid_expiration("TBT", target_date)
        self.assertEqual(result, datetime(2023, 12, 15))

        target_date = datetime(2023, 12, 16)
        result = get_valid_expiration("TBT", target_date)
        self.assertEqual(result, datetime(2023, 12, 22))

        target_date = datetime(2023, 12, 23)
        result = get_valid_expiration("TBT", target_date)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
