import os
import requests
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, StringVar
from ta import momentum, volatility
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import hashlib
import requests
import hmac
from scipy.stats import linregress

base_url = "https://fapi.binance.com"

# Set your API credentials
API_KEY = ""
API_SECRET = ""

delta = 99
historical_orders = []
PNL_timestamps = []
historic_PNL = []
last_price = 0
market_sentiment = ""
last_order_time = None

# Define scatter plot variables
scatter_points = {'buy': [], 'sell': []}
scatter_colors = {'buy': '#8A93B2', 'sell': '#464A76'}
scatter_markers = {'buy': '^', 'sell': 'v'}
scatter_labels = {'buy': 'Buy', 'sell': 'Sell'}


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Trading parameters
symbol = 'ETHUSDT'
interval = '12h'
limit = 90
position_size = 4

def generate_signature(data, secret):
    return hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()

# Function to fetch Kline data
def fetch_kline_data(symbol, interval, limit=100):
    global accumulated_pnl
    base_url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])

    historic_PNL.append(accumulated_pnl)
    PNL_timestamps.append(datetime.utcnow())
    

    return df

# Functions to calculate technical indicators
def calculate_rsi(data, window=7):
    data['rsi'] = momentum.RSIIndicator(data['close'], window=window).rsi()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()

def calculate_stoch_rsi(data, window=14):
    rsi = data['rsi']
    data['stoch_rsi'] = momentum.StochasticOscillator(rsi, rsi, rsi, window=window).stoch()

def calculate_sma(data, window=5):
    data['sma5'] = data['close'].rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    indicator_bb = volatility.BollingerBands(close=data["close"], window=window, window_dev=num_std)
    data['bollinger_mavg'] = indicator_bb.bollinger_mavg()
    data['bollinger_hband'] = indicator_bb.bollinger_hband()
    data['bollinger_lband'] = indicator_bb.bollinger_lband()

def get_entry_price(symbol):
    endpoint = '/fapi/v2/positionRisk'
    url = base_url + endpoint
    
    timestamp = str(int(time.time() * 1000))
    params = f'timestamp={timestamp}'
    
    signature = generate_signature(params, API_SECRET)
    params += f'&signature={signature}'
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        positions = response.json()
        
        for position in positions:
            if position['symbol'] == symbol:
                print(f"Entry Price for {symbol}: {position['entryPrice']}")
                return position['entryPrice']
        
        print(f"No position found for {symbol}")
        
    else:
        print(f"Failed to get position information: {response.content}")

# Function to get account balance

def get_position_risk(symbol):
    endpoint = '/fapi/v2/positionRisk'
    url = base_url + endpoint

    timestamp = str(int(time.time() * 1000))
    params = f'symbol={symbol}&timestamp=' + timestamp
    signature = generate_signature(params, API_SECRET)
    params += '&signature=' + signature

    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        position_risks = response.json()
        
        for position in position_risks:
            if position['symbol'] == symbol:
                print(f"Symbol: {position['symbol']}")
                print(f"Unrealized PnL: {position['unRealizedProfit']}")
                print("---")
                return
        print(f"No position data found for {symbol}")
    else:
        print("Failed to retrieve position risk:", response.content)

def get_position_risk(symbol, pnl_var):
    endpoint = '/fapi/v2/positionRisk'
    url = base_url + endpoint

    timestamp = str(int(time.time() * 1000))
    params = f'symbol={symbol}&timestamp=' + timestamp
    signature = generate_signature(params, API_SECRET)
    params += '&signature=' + signature

    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        position_risks = response.json()

        for position in position_risks:
            if position['symbol'] == symbol:
                pnl = float(position['unRealizedProfit'])
                pnl_var.set(pnl)
                return

        pnl_var.set("No position data found")
    else:
        pnl_var.set("Failed to retrieve position risk")


def place_order(symbol, quantity, side, order_type, price=None):
    endpoint = '/fapi/v1/order'
    url = base_url + endpoint

    timestamp = str(int(time.time() * 1000))

    params = f'symbol={symbol}&side={side}&type={order_type}&quantity={quantity}&timestamp={timestamp}'
    if price:
        params += f'&price={price}'

    signature = generate_signature(params, API_SECRET)
    params += f'&signature={signature}'

    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    response = requests.post(url, headers=headers, params=params)

    if response.status_code == 200:
        print(f"Successfully closed position for {symbol}. Response: {response.json()}")
    else:
        print(f"Failed to close position for {symbol}. Response: {response.content}")

# Create the main GUI window
root = tk.Tk()
root.title(f"{symbol} Financial Data Visualization")
root.configure(bg='#111111')

pnl_var = StringVar()
pnl_var.set("Loading...")

# Initialize a Text widget for the log viewer
log_viewer_label = tk.Label(root, text="Log Viewer", font=("Helvetica", 16), fg="white", bg="#111111")
log_viewer_label.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="w")

scrollbar = tk.Scrollbar(root)
scrollbar.grid(row=4, column=2, sticky="ns")

log_viewer = tk.Text(root, height=10, width=80, bg="#111111", fg="white", wrap=tk.WORD, yscrollcommand=scrollbar.set)
log_viewer.grid(row=4, column=0, columnspan=2, padx=20, pady=10)
log_viewer.insert(tk.END, "Log Viewer Initialized\n")
log_viewer.config(state=tk.DISABLED)

scrollbar.config(command=log_viewer.yview)

accumulated_pnl_label = tk.Label(root, text="Accumulated PNL: $0.00", font=("Helvetica", 16), fg="#8A93B2", bg="#111111")
accumulated_pnl_label.grid(row=11, column=0, padx=20, pady=10, sticky="w")

position_label = tk.Label(root, text="No current position.", font=("Helvetica", 12), fg="white", bg="#111111")
position_label.grid(row=5, column=0, padx=20, pady=10, sticky="w")

trading_params_label = tk.Label(root, text=f"Trading Symbol: {symbol} | Interval: {interval} | Data Limit: {limit} | Deltat: ${delta} | Position size: {position_size} ETH", font=("Helvetica", 12), fg="white", bg="#111111")
trading_params_label.grid(row=10, column=0, padx=20, pady=10, sticky="w")

latest_price_label = tk.Label(root, text="test", font=("Helvetica", 12), fg="white", bg="#111111")
latest_price_label.grid(row=6, column=0, padx=20, pady=10, sticky="w")

conditions_label = tk.Label(root, text="Latest Price: -", font=("Helvetica", 12), fg="white", bg="#111111")
conditions_label.grid(row=9, column=0, padx=20, pady=10, sticky="w")

sentiment_label = tk.Label(root, text="Sentiment is ", font=("Helvetica", 12), fg="white", bg="#111111")
sentiment_label.grid(row=8, column=0, padx=20, pady=10, sticky="w")


# Functions to manage trading logic
position = 0
entry_price = 0
margin = 0
accumulated_pnl = 0
order_type = ""
value_area = ""

def open_long_position(price):
    global position, entry_price, margin, order_type, last_order_time

    current_time = datetime.utcnow()

    # Check if 15 minutes have passed since the last order
    if last_order_time is not None and current_time - last_order_time < timedelta(minutes=720):
        logging.info("Cannot open a new position within 12 hours of the last one.")
        return

    position = position_size
    entry_price = price
    margin = 0
    order_type = "Long"
    historical_orders.append((current_time, 'BUY', price))
    
    # Update last_order_time
    last_order_time = current_time
    
    position_label.config(text=f"Position Size: {position_size} ETH\nEntry Price: {entry_price:.4f} ETH")
    place_order('ETHUSDT', position, 'BUY', 'MARKET')
    logging.info("Opened long position.")

def open_short_position(price):
    global position, entry_price, margin, order_type, last_order_time

    current_time = datetime.utcnow()

    # Check if 15 minutes have passed since the last order
    if last_order_time is not None and current_time - last_order_time < timedelta(minutes=720):
        logging.info("Cannot open a new position within 12 hours of the last one.")
        return

    position = position_size
    entry_price = price
    margin = 0
    order_type = "Short"
    historical_orders.append((current_time, 'SELL', price))
    
    # Update last_order_time
    last_order_time = current_time
    
    position_label.config(text=f"Position Size: {position_size} ETH\nEntry Price: {entry_price:.4f} ETH")
    place_order('ETHUSDT', position, 'SELL', 'MARKET')
    logging.info("Opened short position.")

def calculate_margin_long(price):
    global position, entry_price, margin, symbol
    margin = (price - entry_price) * position
    position_label.config(text=f"Position Size: {position} ETH\nEntry Price: {entry_price:.4f} ETH\nMargin: ${margin}")

def get_overall_trend(numbers):
    if len(numbers) < 2:
        return "Insufficient data for analysis"

    # Compute overall trend using linear regression
    x = np.arange(len(numbers))
    slope, _, _, _, _ = linregress(x, numbers)

    if slope > 0:
        return "Bullish"
    elif slope < 0:
        return "Bearish"
    else:
        return "Sideways"

def calculate_margin_short(price):
    global position, entry_price, margin, symbol
    margin = (entry_price - price) * position
    print(margin)
    position_label.config(text=f"Position Size: {position} ETH\nEntry Price: {entry_price:.4f} ETH\nMargin: ${margin:.4f}")

def check_market_sentiment():

    daily_kline = fetch_kline_data(symbol, '12h', 90)
    calculate_rsi(daily_kline)

    trend_data = [daily_kline['rsi'].iloc[80], daily_kline['rsi'].iloc[81], daily_kline['rsi'].iloc[82], daily_kline['rsi'].iloc[83], daily_kline['rsi'].iloc[84], daily_kline['rsi'].iloc[85], daily_kline['rsi'].iloc[86], daily_kline['rsi'].iloc[87], daily_kline['rsi'].iloc[88], daily_kline['rsi'].iloc[89]]

    market_sentiment = get_overall_trend(trend_data)

    if daily_kline['rsi'].iloc[-1] > 70:
        
        if daily_kline['rsi'].iloc[88] > daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Overbought! Dumping!", fg="#8A93B2")
        elif daily_kline['rsi'].iloc[88] < daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Overbought! Pumping!", fg="#8A93B2")
        else:
            latest_price_label.config(text=f"Overbought! Sideways...", fg="#8A93B2")

    elif daily_kline['rsi'].iloc[-1] < 30:
        
        if daily_kline['rsi'].iloc[88] > daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Over Sold! Dumping!", fg="#464A76")
        elif daily_kline['rsi'].iloc[88] < daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Over Sold! Pumping!", fg="#464A76")
        else:
            latest_price_label.config(text=f"Over Sold! Sideways...", fg="#464A76")

    else:

        if daily_kline['rsi'].iloc[88] > daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Dumping!", fg="#464A76")
        elif daily_kline['rsi'].iloc[88] < daily_kline['rsi'].iloc[-1]:
            latest_price_label.config(text=f"Pumping!", fg="#8A93B2")
        else:
            latest_price_label.config(text=f"Sideways...", fg="white")


    if market_sentiment == "Bearish":
        sentiment_label.config(text=f"Price trend is downward", fg="red")

    if market_sentiment == "Bullish":
        sentiment_label.config(text=f"Price trend is downward", fg="green")
    
def close_long_position(price, reason):
    global position, entry_price, margin, accumulated_pnl, order_type
    accumulated_pnl += margin
    historical_orders.append((datetime.utcnow(), 'SELL', price))
    margin = 0
    place_order('ETHUSDT', position, 'SELL', 'MARKET')
    position = 0
    entry_price = 0
    order_type = ""
    position_label.config(text=f"Position Closed. Reason: {reason}")
    logging.info("Closed long position.")

def close_short_position(price, reason):
    global position, entry_price, margin, accumulated_pnl, order_type
    accumulated_pnl += margin
    historical_orders.append((datetime.utcnow(), 'BUY', price))
    margin = 0
    place_order('ETHUSDT', position, 'BUY', 'MARKET')
    position = 0
    entry_price = 0
    order_type = ""
    position_label.config(text=f"Position Closed. Reason: {reason}")
    logging.info("Closed short position.")

# Plotting
frame = ttk.Frame(root)
frame.grid(row=0, column=0, rowspan=4, padx=20, pady=10)
fig = Figure(figsize=(9 * 1.25, 13), dpi=100, facecolor='#111111')  # Set dark grey background

ax1 = fig.add_subplot(511, facecolor='#111111')
ax2 = fig.add_subplot(512, sharex=ax1, facecolor='#111111')
ax3 = fig.add_subplot(513, sharex=ax1, facecolor='#111111')
ax4 = fig.add_subplot(514, sharex=ax1, facecolor='#111111')
ax5 = fig.add_subplot(515, sharex=ax1, facecolor='#111111')

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()
canvas.draw()

def update_kline_data():
    global position, entry_price, margin, accumulated_pnl, order_type, scatter_points, last_price
    kline_data = fetch_kline_data(symbol, interval, limit)
    earliest_kline_time = kline_data.index.min()
    latest_kline_time = kline_data.index.max()
    get_position_risk(symbol, pnl_var)
    calculate_rsi(kline_data)
    calculate_macd(kline_data)
    calculate_stoch_rsi(kline_data)
    calculate_sma(kline_data)
    calculate_bollinger_bands(kline_data)
    check_market_sentiment()

    latest_ETH_price = kline_data['close'].iloc[-1]
    buy_volume = kline_data['taker_buy_base_asset_volume'].astype(float)
    sell_volume = kline_data['volume'].astype(float) - buy_volume
    
    buy_times = [order_time for order_time, side, price in historical_orders if side == 'BUY']
    buy_prices = [price for order_time, side, price in historical_orders if side == 'BUY']
    sell_times = [order_time for order_time, side, price in historical_orders if side == 'SELL']
    sell_prices = [price for order_time, side, price in historical_orders if side == 'SELL']
    filtered_buy_times = [order_time for order_time, side, price in historical_orders if side == 'BUY' and earliest_kline_time <= order_time <= latest_kline_time]
    filtered_buy_prices = [price for order_time, side, price in historical_orders if side == 'BUY' and earliest_kline_time <= order_time <= latest_kline_time]
    filtered_sell_times = [order_time for order_time, side, price in historical_orders if side == 'SELL' and earliest_kline_time <= order_time <= latest_kline_time]
    filtered_sell_prices = [price for order_time, side, price in historical_orders if side == 'SELL' and earliest_kline_time <= order_time <= latest_kline_time]
    filtered_PNL_timestamps = [time for time in PNL_timestamps if earliest_kline_time <= time <= latest_kline_time]
    filtered_historic_PNL = [pnl for time, pnl in zip(PNL_timestamps, historic_PNL) if earliest_kline_time <= time <= latest_kline_time]
    pnl_color = ""

    # Trading conditions based on the calculated indicators

    if last_price > kline_data['close'].iloc[-1] and kline_data['rsi'].iloc[-1] > 70 and kline_data['macd'].iloc[-1] > kline_data['macd_signal'].iloc[-1] and kline_data['close'].iloc[-1] > kline_data['bollinger_lband'].iloc[-1]:
        open_short_position(latest_ETH_price)

    if last_price < kline_data['close'].iloc[-1] and kline_data['rsi'].iloc[-1] < 30 and kline_data['macd'].iloc[-1] < kline_data['macd_signal'].iloc[-1] and kline_data['close'].iloc[-1] < kline_data['bollinger_lband'].iloc[-1]:
        open_long_position(latest_ETH_price)


    # Plotting the data on the canvas
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()

    kline_data['timestamp'] = mdates.date2num(kline_data.index.to_pydatetime())
    candlestick_data = [tuple(x) for x in kline_data[['timestamp', 'open', 'high', 'low', 'close']].values]
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    candlestick_ohlc(ax1, candlestick_data, width=0.125*2, colorup='#8A93B2', colordown='#464A76')

    ax1.plot(kline_data.index, kline_data['sma5'], label='SMA(5)', color='white')
    ax1.plot(kline_data.index, kline_data['bollinger_mavg'], color='white', linestyle=':')
    ax1.plot(kline_data.index, kline_data['bollinger_hband'], color='grey', linestyle='--')
    ax1.plot(kline_data.index, kline_data['bollinger_lband'], color='grey', linestyle='--')
    ax1.scatter(filtered_buy_times, filtered_buy_prices, color='green', label='Buy Orders', zorder=5, marker='^')
    ax1.scatter(filtered_sell_times, filtered_sell_prices, color='red', label='Sell Orders', zorder=5, marker='v')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(axis='both', colors='white')
    ax1.legend()

    if accumulated_pnl < 0:
        pnl_color = "#464A76"
    else:
        pnl_color = "#8A93B2"

    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.plot(filtered_PNL_timestamps, filtered_historic_PNL, color=pnl_color, label='Cumulative P&L')
    ax2.set_ylabel('PNL', color='white')
    ax2.tick_params(axis='both', colors='white')
    ax2.legend()

    ax3.plot(kline_data.index, kline_data['macd'], label='MACD', color='#8A93B2')
    ax3.plot(kline_data.index, kline_data['macd_signal'], label='Signal', color='#464A76')
    ax3.set_ylabel('MACD', color='white')
    ax3.tick_params(axis='both', colors='white')
    ax3.legend()


    ax4.plot(kline_data.index, kline_data['rsi'], label='RSI', color='#8A93B2')
    ax4.axhline(68, color='grey', linestyle='--')
    ax4.axhline(32, color='grey', linestyle='--')
    ax4.tick_params(axis='both', colors='white')
    ax4.plot(kline_data.index, kline_data['stoch_rsi'], label='Stoch RSI', color='#464A76')
    ax4.set_ylabel('RSI and Stoch RSI', color='white')
    ax4.tick_params(axis='both', colors='white')
    ax4.legend()

    ax5.bar(kline_data.index, buy_volume, label='Buy Volume', color='#8A93B2', alpha=0.7, width=0.125*2)
    ax5.bar(kline_data.index, sell_volume, label='Sell Volume', color='#464A76', alpha=0.7, width=0.125*2, bottom=buy_volume)
    ax5.set_ylabel('Volume', color='white')
    ax5.tick_params(axis='both', colors='white')
    ax5.legend()

    ax1.set_ylabel('Price', color='white')
    ax3.set_ylabel('MACD', color='white')
    ax5.set_ylabel('Volume', color='white')
    ax5.set_xlabel('Time', color='white')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', labelbottom=False)

    if kline_data['sma5'].iloc[-1] < kline_data['close'].iloc[-1]:
        variation_percent = (kline_data['close'].iloc[-1] / kline_data['sma5'].iloc[-1]-1) * 100
    
    else:
        variation_percent = (kline_data['sma5'].iloc[-1] / kline_data['close'].iloc[-1]-1) * -100

    conditions_label.config(text=f"Variation from SMA(5): {variation_percent:.2f} % | RSI(7): {kline_data['rsi'].iloc[-1]:.2f}")

    title = f"ETH Trading Bot - Current Price: ${latest_ETH_price:.2f} ETH"
    
    fig.suptitle(title, fontsize=16, color='white')

    if pnl_var.get() == "Loading...":
        
        accumulated_pnl_label.config(text=f"Loading...", fg="white")
    else:
        accumulated_pnl_label.config(text=f"Accumulated PNL: ${pnl_var}", fg="#8A93B2" if float(pnl_var.get()) >= 0 else "#464A76")
    
    last_price = kline_data['close'].iloc[-1]

    canvas.draw()
    root.after(1000, update_kline_data)

def update_log_viewer(message):
    log_viewer.config(state=tk.NORMAL)
    log_viewer.insert(tk.END, message + "\n")
    log_viewer.config(state=tk.DISABLED)
    log_viewer.yview(tk.END)

class TextHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        update_log_viewer(log_entry)

handler = TextHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(handler)

update_kline_data()
root.mainloop()