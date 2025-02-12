import tkinter as tk
from tkinter import ttk
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# CoinGecko API URL for SOL price
API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"

# 初始化价格和时间列表
prices = []
timestamps = []

# 获取 SOL 价格
def get_sol_price():
    try:
        response = requests.get(API_URL)
        data = response.json()
        price = data['solana']['usd']
        return price
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

# 更新价格和走势图
def update_price_and_chart():
    global prices, timestamps
    price = get_sol_price()
    if price is not None:
        prices.append(price)
        timestamps.append(time.time())
        price_label.config(text=f"SOL Price: ${price:.2f}")
        update_chart()
    # 每 1 秒更新一次
    root.after(1000, update_price_and_chart)

# 更新走势图
def update_chart():
    ax.clear()
    ax.plot(timestamps, prices, marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('SOL Price (USD)')
    ax.set_title('SOL Price Trend')
    canvas.draw()

# 开始刷新
def start_refresh():
    global refresh_thread
    if not refresh_thread or not refresh_thread.is_alive():
        refresh_thread = threading.Thread(target=update_price_and_chart)
        refresh_thread.start()

# 停止刷新
def stop_refresh():
    global refresh_thread
    if refresh_thread and refresh_thread.is_alive():
        # 不能真正停止线程，但可以防止它安排更多更新
        root.after_cancel(root.after_id)
        refresh_thread = None

# 创建主窗口
root = tk.Tk()
root.title("SOL Real-Time Price Viewer")

# 创建价格标签
price_label = tk.Label(root, text="Fetching price...", font=("Arial", 24))
price_label.pack(pady=20)

# 创建开始和停止按钮
start_button = tk.Button(root, text="Start Refresh", command=start_refresh)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Refresh", command=stop_refresh)
stop_button.pack(pady=10)

# 创建 Matplotlib 图形和轴
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# 全局变量来保存刷新线程
refresh_thread = None

# 启动 GUI 事件循环
root.mainloop()
