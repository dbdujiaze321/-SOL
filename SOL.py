import tkinter as tk
import requests
import threading

# CoinGecko API URL for SOL price
API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"

def get_sol_price():
    try:
        response = requests.get(API_URL)
        data = response.json()
        price = data['solana']['usd']
        return price
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def update_price():
    price = get_sol_price()
    if price is not None:
        price_label.config(text=f"SOL Price: ${price:.2f}")
    # Schedule the next update in 60 seconds (you can adjust this interval)
    root.after(60000, update_price)

def start_refresh():
    global refresh_thread
    if not refresh_thread or not refresh_thread.is_alive():
        refresh_thread = threading.Thread(target=update_price)
        refresh_thread.start()

def stop_refresh():
    global refresh_thread
    if refresh_thread and refresh_thread.is_alive():
        # You can't really "stop" a thread in Python, but we can prevent it from scheduling more updates
        root.after_cancel(refresh_thread)
        refresh_thread = None

# Create the main window
root = tk.Tk()
root.title("SOL Real-Time Price Viewer")

# Create a label to display the price
price_label = tk.Label(root, text="Fetching price...", font=("Arial", 24))
price_label.pack(pady=20)

# Create buttons to start and stop refreshing
start_button = tk.Button(root, text="Start Refresh", command=start_refresh)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Refresh", command=stop_refresh)
stop_button.pack(pady=10)

# Global variable to hold the refresh thread
refresh_thread = None

# Start the GUI event loop
root.mainloop()
