import requests
import time
import hmac
import hashlib
from tkinter import Tk, Label, StringVar

# Base URL for Binance API
base_url = 'https://fapi.binance.com'

# Set your API credentials
API_KEY = ""
API_SECRET = ""

# Generate signature for request
def generate_signature(data, secret):
    return hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()

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

# Create Tkinter window
root = Tk()
root.title("Unrealized PnL")
root.config(bg="#242F36")  # Set background color

pnl_var = StringVar()
pnl_var.set("Loading...")

# Create a label and pack it
label = Label(root, textvariable=pnl_var, font=("Helvetica", 99), bg="#242F36")
label.pack()

def update_label():
    get_position_risk('ETHUSDT', pnl_var)
    pnl = pnl_var.get()
    
    try:
        pnl = float(pnl)
        if pnl > 0:
            label.config(fg="green")
        else:
            label.config(fg="red")
    except ValueError:
        label.config(fg="black")
    
    root.after(1000, update_label)  # Update every 1000ms

root.after(1000, update_label)  # First update after 1000ms
root.mainloop()