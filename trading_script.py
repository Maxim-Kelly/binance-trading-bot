import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime as dt
import ccxt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker, start_date, end_date, interval='1d'):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)
    logging.info("Data fetched successfully!")
    return data

def create_features(data):
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(10).apply(lambda x: (x[x > 0].mean() / -x[x < 0].mean()), raw=False)))
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['BB_Upper'] = data['Close'].rolling(window=10).mean() + (data['Close'].rolling(window=10).std() * 2)
    data['BB_Lower'] = data['Close'].rolling(window=10).mean() - (data['Close'].rolling(window=10).std() * 2)
    data.dropna(inplace=True)
    logging.info("Features created successfully!")
    return data

def prepare_data(data):
    X = data[['RSI', 'MA20', 'MA50', 'BB_Upper', 'BB_Lower']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    y.dropna(inplace=True)
    X = X.iloc[:len(y)]
    logging.info("Data prepared successfully!")
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model trained successfully! Accuracy: {accuracy}")
    return model, accuracy

def execute_trade(exchange, symbol, trade_type, amount):
    try:
        if trade_type == 'buy':
            order = exchange.create_market_buy_order(symbol, amount)
        elif trade_type == 'sell':
            order = exchange.create_market_sell_order(symbol, amount)
        logging.info(f"Trade executed successfully: {order}")
    except Exception as e:
        logging.error(f"Error executing trade: {e}")

def run_trading_bot():
    start_date = "2024-01-01"
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    ticker = 'BTC-USD'

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    if data.empty:
        return

    # Create features
    data = create_features(data)
    if data.empty:
        return

    # Prepare data
    X, y = prepare_data(data)
    if X.empty or y.empty:
        return

    # Train model
    model, accuracy = train_model(X, y)
    if model is None:
        return

    # Set up Binance Testnet
    exchange = ccxt.binance({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_API_SECRET',
        'test': True,  # Use the testnet
        'urls': {
            'api': {
                'public': 'https://testnet.binance.vision/api/v3',
                'private': 'https://testnet.binance.vision/sapi/v1',
            }
        }
    })
    symbol = 'BTC/FDUSD'

    # Fetch current balance
    try:
        balance = exchange.fetch_balance()
        btc_holding = balance['total']['BTC']
        fdusd_holding = balance['total']['FDUSD']
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return

    # Generate predictions
    data['Signal'] = model.predict(data[['RSI', 'MA20', 'MA50', 'BB_Upper', 'BB_Lower']])
    data['Position'] = data['Signal'].diff()

    for i in range(1, len(data)):
        if data['Position'].iloc[i] == 1:  # Buy signal
            if fdusd_holding > 0:
                amount = fdusd_holding / data['Close'].iloc[i]
                execute_trade(exchange, symbol, 'buy', amount)
        elif data['Position'].iloc[i] == -1:  # Sell signal
            if btc_holding > 0:
                execute_trade(exchange, symbol, 'sell', btc_holding)

if __name__ == "__main__":
    run_trading_bot()
