import alpaca_trade_api as tradeapi
from datetime import date, datetime, timedelta
import os
import logging
from dotenv import load_dotenv
from typing import Union, List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field


"""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.option import OptionDataStream

from alpaca.data.requests import (
    OptionBarsRequest,
    OptionTradesRequest,
    OptionLatestQuoteRequest,
    OptionLatestTradeRequest,
    OptionSnapshotRequest,
    OptionChainRequest    
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    GetAssetsRequest,
    MarketOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import (
    AssetStatus,
    ExerciseStyle,
    OrderSide,
    OrderType,
    TimeInForce,
    QueryOrderStatus 
)
from alpaca.common.exceptions import APIError
"""

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
trade_client = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')



"""
# get list of options contracts for the given underlying symbol (e.g. SPY,AAPL)
# - get_option_contracts() is a new method to get list of options contracts
# - in this example, we get 2 options contracts for SPY,AAPL
# - you can continue to fetch options contracts by specifying page_token from next_page_token of response
underlying_symbols = ["SPY", "AAPL"]

# Define the Pydantic model
class ContractsRequest(BaseModel):
    underlying_symbols: List[str]
    status: AssetStatus = AssetStatus.ACTIVE
    expiration_date: Optional[date] = None
    expiration_date_gte: Optional[date] = None
    expiration_date_lte: Optional[date] = None
    root_symbol: Optional[str] = None
    type: Optional[ContractType] = None
    style: Optional[ContractStyle] = None
    strike_price_gte: Optional[float] = None
    strike_price_lte: Optional[float] = None
    limit: int = 2
    page_token: Optional[str] = None

    
# Create a ContractsRequest object
request = ContractsRequest(
    underlying_symbols=["AAPL", "GOOG"],
    status=AssetStatus.ACTIVE,
    expiration_date=date(2025, 1, 31),
    limit=10
)


# Create the ContractsRequest object dynamically
args = {
    "underlying_symbols": ["AAPL", "GOOG"],
    "status": AssetStatus.ACTIVE,
    "expiration_date": date(2025, 1, 31),
    "limit": 10
}


# place buy put option order
# - we can place buy put option order same as buy stock/crypto order
req = MarketOrderRequest(
    symbol = high_open_interest_contract.symbol,
    qty = 1,
    side = OrderSide.BUY,
    type = OrderType.MARKET,
    time_in_force = TimeInForce.DAY,
)
res = trade_client.submit_order(req)
res

# get high open_interest contract
open_interest = 0
high_open_interest_contract = None
for contract in res.option_contracts:
    if (contract.open_interest is not None) and (int(contract.open_interest) > open_interest):
        open_interest = int(contract.open_interest)
        high_open_interest_contract = contract
high_open_interest_contract

# Create a ContractsRequest object dynamically
request = create_contracts_request(args)
# def create_contracts_request(args: Dict) -> ContractsRequest:
#     """
#     Create a ContractsRequest object dynamically from a dictionary of arguments.

#     :param args: Dictionary of arguments to populate the ContractsRequest object.
#     :return: A populated ContractsRequest object.
#     """
#    return ContractsRequest(**args)


# Enum definitions for AssetStatus, ContractType, and ContractStyle
class AssetStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ContractType(str, Enum):
    CALL = "call"
    PUT = "put"


class ContractStyle(str, Enum):
    AMERICAN = "american"
    EUROPEAN = "european"


# Pydantic model with getters and setters
class OptionContractsRequestModel(BaseModel):
    underlying_symbols: List[str] = []
    status: AssetStatus = AssetStatus.ACTIVE
    expiration_date: Optional[date] = None
    expiration_date_gte: Optional[date] = None
    expiration_date_lte: Optional[date] = None
    root_symbol: Optional[str] = None
    type: Optional[ContractType] = None
    style: Optional[ContractStyle] = None
    strike_price_gte: Optional[float] = None
    strike_price_lte: Optional[float] = None
    limit: int = 2
    page_token: Optional[str] = None

    # Setters
    def set_underlying_symbols(self, symbols: List[str]):
        self.underlying_symbols = symbols

    def set_status(self, status: AssetStatus):
        self.status = status

    def set_expiration_date(self, expiration_date: date):
        self.expiration_date = expiration_date

    def set_expiration_date_gte(self, expiration_date_gte: date):
        self.expiration_date_gte = expiration_date_gte

    def set_expiration_date_lte(self, expiration_date_lte: date):
        self.expiration_date_lte = expiration_date_lte

    def set_root_symbol(self, root_symbol: str):
        self.root_symbol = root_symbol

    def set_type(self, option_type: ContractType):
        self.type = option_type

    def set_style(self, option_style: ContractStyle):
        self.style = option_style

    def set_strike_price_gte(self, strike_price_gte: float):
        self.strike_price_gte = strike_price_gte

    def set_strike_price_lte(self, strike_price_lte: float):
        self.strike_price_lte = strike_price_lte

    def set_limit(self, limit: int):
        self.limit = limit

    def set_page_token(self, page_token: str):
        self.page_token = page_token

    # Getters
    def get_underlying_symbols(self) -> List[str]:
        return self.underlying_symbols

    def get_status(self) -> AssetStatus:
        return self.status

    def get_expiration_date(self) -> Optional[date]:
        return self.expiration_date

    def get_expiration_date_gte(self) -> Optional[date]:
        return self.expiration_date_gte

    def get_expiration_date_lte(self) -> Optional[date]:
        return self.expiration_date_lte

    def get_root_symbol(self) -> Optional[str]:
        return self.root_symbol

    def get_type(self) -> Optional[ContractType]:
        return self.type

    def get_style(self) -> Optional[ContractStyle]:
        return self.style

    def get_strike_price_gte(self) -> Optional[float]:
        return self.strike_price_gte

    def get_strike_price_lte(self) -> Optional[float]:
        return self.strike_price_lte

    def get_limit(self) -> int:
        return self.limit

    def get_page_token(self) -> Optional[str]:
        return self.page_token


"""
# Example usage
def main():
    # Create and set up the request object
    option_contracts_request = OptionContractsRequestModel()
    option_contracts_request.set_underlying_symbols(["AAPL", "TSLA"])
    option_contracts_request.set_status(AssetStatus.ACTIVE)
    option_contracts_request.set_expiration_date_gte(date(2025, 1, 1))
    option_contracts_request.set_expiration_date_lte(date(2025, 12, 31))
    option_contracts_request.set_type(ContractType.CALL)
    option_contracts_request.set_style(ContractStyle.AMERICAN)
    option_contracts_request.set_limit(10)

    # Convert the Pydantic model into a dictionary for the trade_client
    request_data = option_contracts_request.dict()

    # Send the request to the trade_client
    response = trade_client.get_option_contracts(**request_data)

    # Print the response
    print(response)

    # Retrieve values using getters
    print("Underlying Symbols:", option_contracts_request.get_underlying_symbols())
    print("Status:", option_contracts_request.get_status())
    print("Expiration Date >=:", option_contracts_request.get_expiration_date_gte())
    print("Expiration Date <=:", option_contracts_request.get_expiration_date_lte())
    print("Option Type:", option_contracts_request.get_type())
    print("Option Style:", option_contracts_request.get_style())
    print("Strike Price >=:", option_contracts_request.get_strike_price_gte())
    print("Strike Price <=:", option_contracts_request.get_strike_price_lte())
    print("Limit:", option_contracts_request.get_limit())
"""

# Define enums for SupportedCurrencies, Sort, and TimeFrame
class SupportedCurrencies(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"  # Add more as needed


class Sort(str, Enum):
    ASCENDING = "asc"
    DESCENDING = "desc"


class TimeFrame(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


# Define the Pydantic model for OptionBarsRequest
class OptionBarsRequestModel(BaseModel):
    symbol_or_symbols: Union[str, List[str]] = Field(..., description="Symbol or list of symbols")
    start: Optional[datetime] = Field(None, description="Start datetime for the data")
    end: Optional[datetime] = Field(None, description="End datetime for the data")
    limit: Optional[int] = Field(None, description="Limit for the number of results")
    currency: Optional[SupportedCurrencies] = Field(None, description="Currency for the data")
    sort: Optional[Sort] = Field(None, description="Sort order for the results")
    timeframe: TimeFrame = Field(..., description="Timeframe for the option bars")

    # Getters and setters for each attribute
    def set_symbol_or_symbols(self, symbol_or_symbols: Union[str, List[str]]):
        self.symbol_or_symbols = symbol_or_symbols

    def get_symbol_or_symbols(self) -> Union[str, List[str]]:
        return self.symbol_or_symbols

    def set_start(self, start: Optional[datetime]):
        self.start = start

    def get_start(self) -> Optional[datetime]:
        return self.start

    def set_end(self, end: Optional[datetime]):
        self.end = end

    def get_end(self) -> Optional[datetime]:
        return self.end

    def set_limit(self, limit: Optional[int]):
        self.limit = limit

    def get_limit(self) -> Optional[int]:
        return self.limit

    def set_currency(self, currency: Optional[SupportedCurrencies]):
        self.currency = currency

    def get_currency(self) -> Optional[SupportedCurrencies]:
        return self.currency

    def set_sort(self, sort: Optional[Sort]):
        self.sort = sort

    def get_sort(self) -> Optional[Sort]:
        return self.sort

    def set_timeframe(self, timeframe: TimeFrame):
        self.timeframe = timeframe

    def get_timeframe(self) -> TimeFrame:
        return self.timeframe
    
"""
option_historical_data_client = OptionHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
# Create and set up the request object
option_bars_request = OptionBarsRequestModel(
    symbol_or_symbols=["AAPL", "TSLA"], 
    timeframe=TimeFrame.DAY
)
option_bars_request.set_start(datetime(2025, 1, 1))
option_bars_request.set_end(datetime(2025, 1, 10))
option_bars_request.set_limit(100)
option_bars_request.set_currency(SupportedCurrencies.USD)
option_bars_request.set_sort(Sort.ASCENDING)

# Convert the request object into a dictionary
request_data = option_bars_request.dict()

# Send the request to the trade_client
response = option_historical_data_client.get_option_bars(**request_data)

# Print the response
print(response)
"""
# Reuse enums for SupportedCurrencies and Sort


class SupportedCurrencies(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"  # Add more as needed


class Sort(str, Enum):
    ASCENDING = "asc"
    DESCENDING = "desc"


# Define the Pydantic model for OptionTradesRequest
class OptionTradesRequestModel(BaseModel):
    symbol_or_symbols: Union[str, List[str]] = Field(..., description="Symbol or list of symbols")
    start: Optional[datetime] = Field(None, description="Start datetime for the trades")
    end: Optional[datetime] = Field(None, description="End datetime for the trades")
    limit: Optional[int] = Field(None, description="Limit for the number of results")
    currency: Optional[SupportedCurrencies] = Field(None, description="Currency for the trades")
    sort: Optional[Sort] = Field(None, description="Sort order for the results")

    # Getters and setters for each attribute
    def set_symbol_or_symbols(self, symbol_or_symbols: Union[str, List[str]]):
        self.symbol_or_symbols = symbol_or_symbols

    def get_symbol_or_symbols(self) -> Union[str, List[str]]:
        return self.symbol_or_symbols

    def set_start(self, start: Optional[datetime]):
        self.start = start

    def get_start(self) -> Optional[datetime]:
        return self.start

    def set_end(self, end: Optional[datetime]):
        self.end = end

    def get_end(self) -> Optional[datetime]:
        return self.end

    def set_limit(self, limit: Optional[int]):
        self.limit = limit

    def get_limit(self) -> Optional[int]:
        return self.limit

    def set_currency(self, currency: Optional[SupportedCurrencies]):
        self.currency = currency

    def get_currency(self) -> Optional[SupportedCurrencies]:
        return self.currency

    def set_sort(self, sort: Optional[Sort]):
        self.sort = sort

    def get_sort(self) -> Optional[Sort]:
        return self.sort
    
"""
option_trades_request = OptionTradesRequestModel(symbol_or_symbols=["AAPL", "TSLA"])
option_trades_request.set_start(datetime(2025, 1, 1))
option_trades_request.set_end(datetime(2025, 1, 10))
option_trades_request.set_limit(100)
option_trades_request.set_currency(SupportedCurrencies.USD)
option_trades_request.set_sort(Sort.DESCENDING)

# Convert the request object into a dictionary
request_data = option_trades_request.dict()

# Send the request to the trade_client
response = trade_client.get_option_trades(**request_data)

# Print the response
print(response)

"""

# Define the OptionsFeed enum
class OptionsFeed(str, Enum):
    SIP = "sip"  # Securities Information Processor
    OTC = "otc"  # Over-The-Counter feed


# Define the Pydantic model for OptionLatestQuoteRequest
class OptionLatestQuoteRequestModel(BaseModel):
    symbol_or_symbols: Union[str, List[str]] = Field(..., description="Symbol or list of symbols")
    feed: Optional[OptionsFeed] = Field(None, description="Feed type (e.g., SIP or OTC)")

    # Getters and setters for each attribute
    def set_symbol_or_symbols(self, symbol_or_symbols: Union[str, List[str]]):
        self.symbol_or_symbols = symbol_or_symbols

    def get_symbol_or_symbols(self) -> Union[str, List[str]]:
        return self.symbol_or_symbols

    def set_feed(self, feed: Optional[OptionsFeed]):
        self.feed = feed

    def get_feed(self) -> Optional[OptionsFeed]:
        return self.feed
    

"""
# Create and set up the request object
option_latest_quote_request = OptionLatestQuoteRequestModel(symbol_or_symbols=["AAPL", "TSLA"])
option_latest_quote_request.set_feed(OptionsFeed.SIP)

# Convert the request object into a dictionary
request_data = option_latest_quote_request.dict()

# Send the request to the trade_client
response = trade_client.get_option_latest_quote(**request_data)

# Print the response
print(response)

"""

# Define the OptionsFeed enum (reuse if already defined)
class OptionsFeed(str, Enum):
    SIP = "sip"  # Securities Information Processor
    OTC = "otc"  # Over-The-Counter feed


# Define the Pydantic model for OptionLatestTradeRequest
class OptionLatestTradeRequestModel(BaseModel):
    symbol_or_symbols: Union[str, List[str]] = Field(..., description="Symbol or list of symbols")
    feed: Optional[OptionsFeed] = Field(None, description="Feed type (e.g., SIP or OTC)")

    # Getters and setters for each attribute
    def set_symbol_or_symbols(self, symbol_or_symbols: Union[str, List[str]]):
        self.symbol_or_symbols = symbol_or_symbols

    def get_symbol_or_symbols(self) -> Union[str, List[str]]:
        return self.symbol_or_symbols

    def set_feed(self, feed: Optional[OptionsFeed]):
        self.feed = feed

    def get_feed(self) -> Optional[OptionsFeed]:
        return self.feed

"""
# Create and set up the request object
option_latest_trade_request = OptionLatestTradeRequestModel(symbol_or_symbols=["AAPL", "TSLA"])
option_latest_trade_request.set_feed(OptionsFeed.SIP)

# Convert the request object into a dictionary
request_data = option_latest_trade_request.dict()

# Send the request to the trade_client
response = trade_client.get_option_latest_trade(**request_data)

# Print the response
print(response)

"""
# Define the Pydantic model for OptionSnapshotRequest


class OptionSnapshotRequestModel(BaseModel):
    symbol_or_symbols: Union[str, List[str]] = Field(..., description="Symbol or list of symbols")
    feed: Optional[OptionsFeed] = Field(None, description="Feed type (e.g., SIP or OTC)")

    # Getters and setters for each attribute
    def set_symbol_or_symbols(self, symbol_or_symbols: Union[str, List[str]]):
        self.symbol_or_symbols = symbol_or_symbols

    def get_symbol_or_symbols(self) -> Union[str, List[str]]:
        return self.symbol_or_symbols

    def set_feed(self, feed: Optional[OptionsFeed]):
        self.feed = feed

    def get_feed(self) -> Optional[OptionsFeed]:
        return self.feed


"""
# Create and set up the request object
option_snapshot_request = OptionSnapshotRequestModel(symbol_or_symbols=["AAPL", "TSLA"])
option_snapshot_request.set_feed(OptionsFeed.OTC)

# Convert the request object into a dictionary
request_data = option_snapshot_request.dict()

# Send the request to the trade_client
response = trade_client.get_option_snapshot(**request_data)

# Print the response
print(response)
"""


# Define the OptionsFeed and ContractType enums (reuse if already defined)
class OptionsFeed(str, Enum):
    SIP = "sip"  # Securities Information Processor
    OTC = "otc"  # Over-The-Counter feed


class ContractType(str, Enum):
    CALL = "call"
    PUT = "put"


# Define the Pydantic model for OptionChainRequest
class OptionChainRequestModel(BaseModel):
    underlying_symbol: str = Field(..., description="Underlying symbol")
    feed: Optional[OptionsFeed] = Field(None, description="Feed type (e.g., SIP or OTC)")
    type: Optional[ContractType] = Field(None, description="Option type (CALL or PUT)")
    strike_price_gte: Optional[float] = Field(None, description="Minimum strike price")
    strike_price_lte: Optional[float] = Field(None, description="Maximum strike price")
    expiration_date: Optional[Union[date, str]] = Field(None, description="Specific expiration date")
    expiration_date_gte: Optional[Union[date, str]] = Field(None, description="Earliest expiration date")
    expiration_date_lte: Optional[Union[date, str]] = Field(None, description="Latest expiration date")
    root_symbol: Optional[str] = Field(None, description="Root symbol")
    updated_since: Optional[datetime] = Field(None, description="Updated since a specific datetime")

    # Getters and setters for each attribute
    def set_underlying_symbol(self, underlying_symbol: str):
        self.underlying_symbol = underlying_symbol

    def get_underlying_symbol(self) -> str:
        return self.underlying_symbol

    def set_feed(self, feed: Optional[OptionsFeed]):
        self.feed = feed

    def get_feed(self) -> Optional[OptionsFeed]:
        return self.feed

    def set_type(self, type: Optional[ContractType]):
        self.type = type

    def get_type(self) -> Optional[ContractType]:
        return self.type

    def set_strike_price_gte(self, strike_price_gte: Optional[float]):
        self.strike_price_gte = strike_price_gte

    def get_strike_price_gte(self) -> Optional[float]:
        return self.strike_price_gte

    def set_strike_price_lte(self, strike_price_lte: Optional[float]):
        self.strike_price_lte = strike_price_lte

    def get_strike_price_lte(self) -> Optional[float]:
        return self.strike_price_lte

    def set_expiration_date(self, expiration_date: Optional[Union[date, str]]):
        self.expiration_date = expiration_date

    def get_expiration_date(self) -> Optional[Union[date, str]]:
        return self.expiration_date

    def set_expiration_date_gte(self, expiration_date_gte: Optional[Union[date, str]]):
        self.expiration_date_gte = expiration_date_gte

    def get_expiration_date_gte(self) -> Optional[Union[date, str]]:
        return self.expiration_date_gte

    def set_expiration_date_lte(self, expiration_date_lte: Optional[Union[date, str]]):
        self.expiration_date_lte = expiration_date_lte

    def get_expiration_date_lte(self) -> Optional[Union[date, str]]:
        return self.expiration_date_lte

    def set_root_symbol(self, root_symbol: Optional[str]):
        self.root_symbol = root_symbol

    def get_root_symbol(self) -> Optional[str]:
        return self.root_symbol

    def set_updated_since(self, updated_since: Optional[datetime]):
        self.updated_since = updated_since

    def get_updated_since(self) -> Optional[datetime]:
        return self.updated_since

"""
# Create and set up the request object
option_chain_request = OptionChainRequestModel(underlying_symbol="AAPL")
option_chain_request.set_feed(OptionsFeed.SIP)
option_chain_request.set_type(ContractType.CALL)
option_chain_request.set_strike_price_gte(100.0)
option_chain_request.set_strike_price_lte(200.0)
option_chain_request.set_expiration_date_gte(date(2025, 1, 1))
option_chain_request.set_expiration_date_lte(date(2025, 12, 31))
option_chain_request.set_root_symbol("AAPL")
option_chain_request.set_updated_since(datetime(2025, 1, 1, 12, 0, 0))

# Convert the request object into a dictionary
request_data = option_chain_request.dict()

# Send the request to the trade_client
response = trade_client.get_option_chain(**request_data)

# Print the response
print(response)


"""





"""
# get list of options contracts for the given underlying symbol (e.g. SPY)

# setup option historical data client
option_historical_data_client = OptionHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
# get options historical bars by symbol
req = OptionBarsRequest(
    symbol_or_symbols = high_open_interest_contract.symbol,
    timeframe = TimeFrame(amount = 1, unit = TimeFrameUnit.Hour),   # specify timeframe
    start = now - timedelta(days = 5),                              # specify start datetime, default=the beginning of the current day.
    # end_date=None,                                                # specify end datetime, default=now
    limit = 2,                                                      # specify limit
)
# get options historical bars by symbol
option_historical_data_client.get_option_bars(req).df
# get options historical trades by symbol
req = OptionTradesRequest(
    symbol_or_symbols = high_open_interest_contract.symbol,
    start = now - timedelta(days = 5),                              # specify start datetime, default=the beginning of the current day.
    # end=None,                                                     # specify end datetime, default=now
    limit = 2,                                                      # specify limit
)
option_historical_data_client.get_option_trades(req).df

"""
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
