from statistics import LinearRegression
import requests
import pandas as pd
import time
import os
import pickle
from datetime import datetime, timedelta
import logging
import plotly.graph_objs as go  # Import Plotly for interactive graphs
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicHermiteSpline
import numpy as np
from dotenv import load_dotenv
from datetime import datetime


# Set up logging to output detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cache_data(data, symbol, data_type):
    """Cache data with a timestamp for freshness validation."""
    filename = f"cache_{symbol}_{data_type}.pkl"
    timestamped_data = {'timestamp': datetime.now(), 'data': data}
    with open(filename, 'wb') as file:
        pickle.dump(timestamped_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Data for {symbol} cached successfully.")


def load_cached_data(symbol, data_type, freshness='1 day'):
    """Load cached data if available and not outdated."""
    filename = f"cache_{symbol}_{data_type}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            cached_data = pickle.load(file)
            freshness_days = int(freshness.split()[0])  # Assumes freshness format is like '1 day'
            if cached_data['timestamp'] + timedelta(days=freshness_days) > datetime.now():
                logging.info(f"Using cached data for {symbol}.")
                return cached_data['data']
    logging.info(f"No valid cached data found for {symbol}.")
    return None


def rate_limiter(last_call_time, max_calls_per_minute=5):
    """Calculate the necessary delay to maintain API rate limits."""
    time_since_last_call = datetime.now() - last_call_time
    required_delay = max(0, (60 / max_calls_per_minute) - time_since_last_call.total_seconds())
    if required_delay > 0:
        logging.info(f"Rate limiting in effect. Sleeping for {required_delay} seconds.")
        time.sleep(required_delay)
    return datetime.now()


def fetch_stock_data(symbol, api_key, data_type='daily', last_call_time=datetime.min, freshness='1 day'):
    """Fetch stock data considering caching and rate limiting."""
    cached_data = load_cached_data(symbol, data_type, freshness)
    if cached_data is not None and not cached_data.empty:
        return cached_data, last_call_time

    BASE_URL = "https://www.alphavantage.co/query"
    FUNCTION_MAP = {
        'intraday': 'TIME_SERIES_INTRADAY',
        'daily': 'TIME_SERIES_DAILY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY'
    }
    params = {
        "function": FUNCTION_MAP[data_type],
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": 'full'
    }
    if data_type == 'intraday':
        params['interval'] = '5min'

    last_call_time = rate_limiter(last_call_time)
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Error Message" in data:
        error_msg = f"Error fetching data for {symbol}: {data['Error Message']}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    key = next(iter(data.keys() - {'Meta Data'}), None)
    if not key:
        error_msg = f"No data key found for {symbol}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(f"Data for {symbol} under key '{key}': {data[key]}")

    try:
        df = pd.DataFrame.from_dict(data[key], orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    except Exception as e:
        logging.error(f"Error processing data for {symbol}: {e}")
        raise

    cache_data(df, symbol, data_type)
    return df, last_call_time


def fetch_data_for_multiple_stocks(symbols, api_key, data_type, freshness, last_call_time):
    """Fetch data for multiple stocks handling caching and rate limits."""
    all_data = {}
    for symbol in symbols:
        df, last_call_time = fetch_stock_data(symbol, api_key, data_type, last_call_time, freshness)
        if df is not None:
            all_data[symbol] = df
    return all_data, last_call_time


def convert_data_types(df):
    """
    Convert data columns to numeric, handling errors.
    """
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calculate_moving_averages(df, windows=[20, 50, 200]):
    """
    Calculate moving averages for specified window lengths.
    """
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands for stock prices.
    """
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper_BB'] = df['MA'] + (df['STD'] * num_std)
    df['Lower_BB'] = df['MA'] - (df['STD'] * num_std)
    return df


def predict_future_prices(df, feature_cols=['SMA_20', 'SMA_50', 'SMA_200', 'Upper_BB', 'Lower_BB']):
    """
    Predict future stock prices using linear regression based on calculated indicators.
    """
    df = df.dropna(subset=feature_cols + ['Close'])
    X = df[feature_cols]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error of prediction: {mse}")

    return model


def run_analysis(symbols, api_key):
    """
    Fetch and analyze stock data for given symbols, and perform predictions.
    """
    data = fetch_data_for_multiple_stocks(symbols, api_key, data_type='daily')

    analysis_results = {}
    for symbol, df in data.items():
        print(f"Analyzing data for {symbol}")
        df = convert_data_types(df)
        df = calculate_moving_averages(df)
        df = add_bollinger_bands(df)
        model = predict_future_prices(df)
        analysis_results[symbol] = {
            'dataframe': df,
            'model': model
        }
    return analysis_results


# Load the sentiment analysis model once outside the function
bert_sentiment = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


def fetch_news_headlines(symbol, news_api_key):
    """
    Fetch news headlines and their publication dates for a given stock symbol using NewsAPI.
    """
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}&language=en&sortBy=publishedAt"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch news for {symbol}: {response.text}")
        return []
    articles = response.json().get('articles', [])
    print(f"Fetched {len(articles)} articles for {symbol}")
    headlines = [{'title': article['title'], 'publishedAt': pd.to_datetime(article['publishedAt']).date()} for article in articles]
    return headlines


def analyze_sentiment(headlines):
    """
    Analyze sentiment of headlines using a pre-loaded BERT-based model.
    """
    if not headlines:
        print("No headlines to analyze.")
        return []
    try:
        for headline in headlines:
            result = bert_sentiment(headline['title'])
            headline['sentiment'] = result[0]['label']
            headline['score'] = result[0]['score']
        print(f"Sentiment analysis complete for {len(headlines)} headlines.")
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return []
    return headlines


def integrate_sentiment_data(df, headlines):
    """
    Integrate sentiment data into the stock DataFrame based on publication dates.
    """
    if not headlines:
        print("No sentiment data to integrate.")
        return df

    sentiment_df = pd.DataFrame(headlines)
    if sentiment_df.empty:
        print("Sentiment DataFrame is empty.")
        return df

    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt']).dt.date
    sentiment_df.set_index('publishedAt', inplace=True)
    df = df.join(sentiment_df['score'], how='left')
    df['score'].fillna(method='ffill', inplace=True)
    print("Sentiment scores successfully integrated.")

    return df


def perform_sentiment_analysis(symbol, news_api_key, df):
    """
    Fetch headlines, perform sentiment analysis, and append results to the DataFrame.
    """
    print(f"Starting sentiment analysis for {symbol}")
    headlines = fetch_news_headlines(symbol, news_api_key)
    headlines = analyze_sentiment(headlines)
    df = integrate_sentiment_data(df, headlines)
    return df


def preprocess_data(df, feature_columns):
    df = df.dropna(subset=feature_columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    y = df['Close'].values
    return X, y, scaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Training completed. MSE: {mse:.2f}, R2: {r2:.2f}")
    return model


def predict_future_prices_hermite(model, df, feature_columns, scaler, years=5, last_n=10):
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=years * 365, freq='D')[1:]
    future_data = pd.DataFrame(index=future_dates)

    last_values = df[feature_columns][-last_n:].values
    scaled_features = scaler.transform(last_values)
    y = model.predict(scaled_features)
    dydx = np.gradient(y)
    interpolator = CubicHermiteSpline(range(last_n), y, dydx)
    trend = interpolator(np.linspace(0, last_n - 1, num=len(future_dates)))

    future_data['Predicted Close'] = trend
    print(f"Future predictions saved to 'predicted_stock_prices_hermite.csv'.")
    return future_data


def save_plot(figure, file_path, title):
    """
    Saves the given Matplotlib figure to the specified file path and closes it.
    """
    figure.suptitle(title)
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.savefig(file_path)
    plt.close(figure)
    print(f"Plot saved to {file_path}")


def plot_combined(df, title):
    """
    Creates a combined static and interactive plot for stock prices and volumes.
    """
    file_path = f"combined_{title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)

    # Line graph for stock prices using Matplotlib
    ax1.plot(df.index, df['Close'], label=f'{title} Close Price', color='skyblue')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')

    # Bar graph for volume trends using Matplotlib
    ax2.bar(df.index, df['Volume'], color='lightblue', label=f'{title} Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right')

    save_plot(fig, file_path, f"Combined Trends for {title}")

    # Interactive version using Plotly
    trace1 = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price')
    trace2 = go.Bar(x=df.index, y=df['Volume'], name='Volume')
    layout = go.Layout(title=f"Interactive Combined Trends for {title}",
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price ($)'),
                       template='plotly_white')
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.write_html(f"interactive_{title}.html")
    print(f"Interactive plot saved to interactive_{title}.html")


def plot_sentiment_trends(df, title):
    """
    Creates a static and interactive plot for sentiment trends if sentiment data is available.
    """
    if 'score' in df.columns and not df['score'].isna().all():
        file_path = f"sentiment_trends_{title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['score'], color='purple', label=f'{title} Sentiment Score')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        save_plot(fig, file_path, f"Sentiment Trends for {title}")

        # Interactive version using Plotly
        trace = go.Scatter(x=df.index, y=df['score'], mode='lines', name='Sentiment Score')
        layout = go.Layout(title=f"Interactive Sentiment Trends for {title}",
                           xaxis=dict(title='Date'),
                           yaxis=dict(title='Sentiment Score'),
                           template='plotly_white')
        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(f"interactive_{title}.html")
        print(f"Interactive sentiment plot saved to interactive_{title}.html")
    else:
        print("No sentiment data available to plot.")


# main function
def main():
    load_dotenv()
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    news_api_key = os.getenv('NEWS_API_KEY')
    symbols = ['AAPL', 'MSFT', 'NVDA']
    last_call_time = datetime.min

    logging.info("Starting data load process...")
    data_type = 'daily'
    freshness = '1 day'
    stock_data, last_call_time = fetch_data_for_multiple_stocks(symbols, api_key, data_type, freshness, last_call_time)

    predictions = {}  # Dictionary to store predictions for each stock

    if not stock_data:
        logging.error("Failed to load data. Check API key and network connection.")
        return

    logging.info("Data loaded successfully.")
    for symbol, df_raw in stock_data.items():
        if df_raw is not None and not df_raw.empty:
            logging.info(f"Performing analysis and modeling for {symbol}...")
            df = preprocess_data(df_raw, ['Open', 'High', 'Low', 'Volume'])
            model = train_model(df[0], df[1])

            logging.info(
                f"Training model completed for {symbol}. Now predicting future prices using Hermite interpolation...")
            future_prices_df = predict_future_prices_hermite(model, df_raw, ['Open', 'High', 'Low', 'Volume'], df[2],
                                                             years=5)
            predictions[symbol] = future_prices_df['Predicted Close']

            logging.info(f"Visualizing data for {symbol}...")
            plot_combined(df_raw, symbol)
            plot_sentiment_trends(df_raw, symbol)
            logging.info(f"Visualization completed for {symbol}.")
        else:
            logging.warning(f"No data available for {symbol} to analyze.")

    # Combine predictions into one DataFrame and save to CSV
    combined_predictions = pd.DataFrame(predictions)
    combined_predictions.to_csv('combined_predicted_stock_prices.csv')
    logging.info("Combined future predictions saved to 'combined_predicted_stock_prices.csv'.")


if __name__ == "__main__":
    main()
