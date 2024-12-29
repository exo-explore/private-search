import yfinance as yf
import requests
from datetime import datetime
import pytz
from tabulate import tabulate
import time
from typing import Dict, Union, Tuple

def get_stock_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        price = None
        for attr in ['regularMarketPrice', 'currentPrice', 'previousClose']:
            price = stock.info.get(attr)
            if price is not None:
                break
        
        if price is None:
            hist = stock.history(period='1d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        
        return price if price is not None else None
    except Exception as e:
        return None

def get_crypto_price(coin_id: str) -> float:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[coin_id]['usd']
    except Exception:
        return None

def get_market_prices() -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Fetches current prices for major stocks and cryptocurrencies.
    
    Returns:
        Tuple containing:
        - Dictionary of stock prices {company_name: price}
        - Dictionary of crypto prices {crypto_name: price}
        - Current timestamp in EST
    """
    # Stock tickers
    stocks = {
        'Apple Inc.': 'AAPL',
        'NVIDIA Corporation': 'NVDA',
        'Microsoft Corporation': 'MSFT',
        'Amazon.com, Inc.': 'AMZN',
        'Alphabet Inc.': 'GOOGL',
        'Meta Platforms, Inc.': 'META',
        'Tesla, Inc.': 'TSLA'
    }

    # Crypto IDs (as used by CoinGecko)
    cryptos = {
        'Bitcoin (BTC)': 'bitcoin',
        'Ethereum (ETH)': 'ethereum',
        'Solana (SOL)': 'solana'
    }

    # Get current time in EST
    est = pytz.timezone('US/Eastern')
    current_time = datetime.now(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')

    # Fetch stock prices
    stock_prices = {}
    for name, ticker in stocks.items():
        price = get_stock_price(ticker)
        stock_prices[name] = price
        time.sleep(0.5)

    # Fetch crypto prices
    crypto_prices = {}
    for name, coin_id in cryptos.items():
        price = get_crypto_price(coin_id)
        crypto_prices[name] = price
        time.sleep(0.5)

    return stock_prices, crypto_prices, current_time

def format_prices(stock_prices: Dict[str, float], crypto_prices: Dict[str, float], timestamp: str) -> str:
    """
    Formats the prices into a readable string.
    
    Args:
        stock_prices: Dictionary of stock prices
        crypto_prices: Dictionary of crypto prices
        timestamp: Current timestamp
        
    Returns:
        Formatted string with all prices
    """
    stock_data = [
        [name, f"${price:,.2f}" if price is not None else "N/A"]
        for name, price in stock_prices.items()
    ]
    
    crypto_data = [
        [name, f"${price:,.2f}" if price is not None else "N/A"]
        for name, price in crypto_prices.items()
    ]
    
    output = []
    output.append(f"\nMarket Prices as of {timestamp}\n")
    output.append("Stocks:")
    output.append(tabulate(stock_data, headers=['Company', 'Price'], tablefmt='grid'))
    output.append("\nCryptocurrencies:")
    output.append(tabulate(crypto_data, headers=['Cryptocurrency', 'Price'], tablefmt='grid'))
    
    return "\n".join(output)

if __name__ == "__main__":
    # Example usage
    stock_prices, crypto_prices, timestamp = get_market_prices()
    
    # Print formatted results
    print(format_prices(stock_prices, crypto_prices, timestamp))
    
    # Example of using the raw data
    print("\nRaw data example:")
    print(f"Apple stock price: ${stock_prices['Apple Inc.']:,.2f}")
    print(f"Bitcoin price: ${crypto_prices['Bitcoin (BTC)']:,.2f}") 