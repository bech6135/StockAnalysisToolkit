import requests
import pandas as pd

def fetch_stock_data(symbol):
    API_KEY = 'Z3ZU4UQGVFEXQZ59'  # Use your dedicated access key
    BASE_URL = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df.index = pd.to_datetime(df.index)
    return df

# Example usage:
df = fetch_stock_data('AAPL')  # Fetch Apple Inc. stock data
print(df.head())  # Print the first few rows of the data
