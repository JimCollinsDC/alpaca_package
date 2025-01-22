import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alpaca_trading.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from a .env file
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


def place_order(
    symbol: str,
    strike_price: float,
    limit_price: float,
    expiration_date: str,
    time_in_force: str,  # "gtc", "day", etc.
    order_side: str,  # "buy", "sell"
    order_type: str  # "market", "limit", etc.
) -> Optional[Dict]:
    """
    Place an option order for the given symbol at the specified price.

    Args:
        symbol (str): The underlying symbol (e.g., "TBT").
        strike_price (float): The strike price for the option order.
        limit_price (float): The limit price for the option order.
        expiration_date (str): The expiration date in YYYY-MM-DD format.
        time_in_force (str): The time-in-force for the order (e.g., "gtc", "day").
        order_side (str): The side of the order, either "buy" or "sell".
        order_type (str): The type of the order, e.g., "market", "limit".

    Returns:
        Optional[Dict]: A dictionary containing the order details if successful, or None on failure.
    """
    try:
        nearest_strike = round(strike_price)
        # Format strike price for OCC compliance
        strike_formatted = f"{int(nearest_strike * 1000):08d}"  
        option_symbol = (
            f"{symbol}{expiration_date[2:].replace('-', '')}P{strike_formatted}"
        )

        logging.info(
            f"Placing order: symbol={option_symbol},"
            f"limit_price={limit_price},"
            f"expiration_date={expiration_date}"
        )

        # Submit the order
        order = api.submit_order(
            symbol=option_symbol,
            qty=1,
            side=order_side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )

        logging.info(f"Order placed successfully: {order}")
        return order
    except Exception as e:
        logging.error(f"Failed to place order: {e}")
        return None


def nearest_strike_price(price: float, increment: float) -> float:
    """
    Calculate the nearest valid strike price based on a given increment.

    Args:
        price (float): The target price to adjust.
        increment (float): The increment used to determine valid strike prices.

    Returns:
        float: The nearest valid strike price.
    """
    result = round(price / increment) * increment
    logging.debug(
        f"Nearest strike price for price={price},"
        f"increment={increment}: {result}"
    )
    return result


def get_third_friday(year: int, month: int) -> datetime:
    """
    Calculate the third Friday of a given month and year.

    Args:
        year (int): The year for which to calculate the third Friday.
        month (int): The month for which to calculate the third Friday.

    Returns:
        datetime: The date of the third Friday in the specified month and year.
    """
    first_day = datetime(year, month, 1)
    # First Friday of the month
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    logging.debug(f"Third Friday for {year}-{month}: {third_friday}")
    return third_friday


def get_valid_expiration(symbol: str,
                         target_date: datetime) -> Optional[datetime]:
    """
    Retrieve the nearest valid expiration date for a given symbol.

    Args:
        symbol (str): The stock or ETF symbol.
        target_date (datetime): The desired expiration date.

    Returns:
        Optional[datetime]: The nearest valid expiration date, or None 
        if not found.
    """
    try:
        expirations = api.get_options_expirations(symbol)
        expiration_dates = sorted(datetime.strptime(date, '%Y-%m-%d') for date in expirations)

        for expiration in expiration_dates:
            if expiration >= target_date:
                logging.info(
                    f"Valid expiration found for {symbol}: {expiration}"
                    )
                return expiration

        logging.warning(
            f"No valid expiration date found for {symbol}"
            f"on or after {target_date}.")
        return None
    except Exception as e:
        logging.error(f"Error fetching expirations for {symbol}: {e}")
        return None
