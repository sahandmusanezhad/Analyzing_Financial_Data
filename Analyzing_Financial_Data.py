import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress

ticker = 'AAPL'
data = yf.download(ticker, start="2020-01-01", end="2023-01-01", auto_adjust=False)

print(data.head())

data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Plot the Closing price and moving averages
plt.figure(figsize = (12,6))
plt.plot(data['Close'], label = 'Close Prices')
plt.plot(data['MA20'], label = '20-Day MA')
plt.plot(data['MA50'], label = '50-Day MA')
plt.title(f'{ticker} - Close Price and Moving Averages')
plt.legend()
plt.show()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

# Calculate RSI
data['RSI'] = calculate_rsi(data)

# Plot RSI
plt.figure(figsize = (12,6))
plt.plot(data['RSI'], label = 'RSI')
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.title(f'{ticker} - Relative Strength Index (RSI)')
plt.legend()
plt.show()

# Define the trading strategy
def trading_strategy(data):
    data['Signal'] = 0

    # buy Signal: When 20-day MA crosses above 50-day MA and RSI < 30
    data.loc[(data['MA20'] > data['MA50']) & (data['RSI'] < 30), 'Signal'] = 1

    # Sell signal: When 20-day MA crosses below 20-day and RSI > 70
    data.loc[(data['MA20'] < data['MA50']) & (data['RSI'] > 70), 'Signal'] = -1

    data['Position'] = data['Signal'].shift()

    return data

# Apply the strategy
data = trading_strategy(data)

# Plot buy and sell signals
plt.figure(figsize = (12,6))
plt.plot(data['Close'], label = 'Close Price')
plt.plot(data[data['Position'] == 1].index,
         data['Close'][data['Position'] == 1],
         '^', markersize = 10, color = 'g', label='Buy Signal')
plt.plot(data[data['Position'] == -1].index,
         data['Close'][data['Position'] == -1],
         'v', markersize = 10, color = 'r', label='Sell Signal')
plt.title(f'{ticker} - Buy and Sell Signals')
plt.legend()
plt.show()


# Calculate returns
data['Returns'] = data['Close'].pct_change()

# Calculate strategy returns
data['Strategy_Returns'] = data['Returns'].shift()

# Plot cumulative returns
plt.figure(figsize = (12,6))
data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()
data['Cumulative_Market_Returns'] = (1 + data['Returns']).cumprod()

plt.plot(data['Cumulative_Strategy_Returns'], label = 'Strategy Returns')
plt.plot(data['Cumulative_Market_Returns'], label = 'Market Returns')
plt.title(f'{ticker} - Cumulative Strategy vs Market Returns')
plt.legend()
plt.show()