import tkinter as tk
from tkinter import ttk
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import numpy as np
from datetime import datetime

# CoinGecko API URL for SOL price
API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies"

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
        print(f"获取价格时出错: {e}")
        return None

# 简单移动平均线交叉策略预测
def predict_trend(prices, short_window=5, long_window=20):
    if len(prices) < long_window:
        return "数据不足，无法预测"
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    if short_ma > long_ma:
        return "预计上涨"
    elif short_ma < long_ma:
        return "预计下跌"
    else:
        return "趋势不明"

# 更新价格、走势图和预测结果
def update_price_and_chart():
    global prices, timestamps
    price = get_sol_price()
    if price is not None:
        prices.append(price)
        timestamps.append(time.time())
        price_label.config(text=f"SOL 价格: ${price:.2f}")
        trend = predict_trend(prices)
        show_prediction_in_new_window(trend)
        update_chart()
    # 每 60 秒更新一次
    root.after(60000, update_price_and_chart)

# 更新走势图
def update_chart():
    ax.clear()
    ax.plot(timestamps, prices, marker='o')
    ax.set_xlabel('时间')
    ax.set_ylabel('SOL 价格 (美元)')
    ax.set_title('SOL 价格走势')
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

# 在新窗口显示预测结果和时间
def show_prediction_in_new_window(trend):
    prediction_window = tk.Toplevel(root)
    prediction_window.title("SOL 价格预测结果")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
    prediction_label = tk.Label(prediction_window, text=prediction_text, font=("Arial", 18))
    prediction_label.pack(pady=20)

# 创建主窗口
root = tk.Tk()
root.title("SOL 实时价格查看器")

# 创建价格标签
price_label = tk.Label(root, text="正在获取价格...", font=("Arial", 24))
price_label.pack(pady=20)

# 创建开始和停止按钮
start_button = tk.Button(root, text="开始刷新", command=start_refresh)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="停止刷新", command=stop_refresh)
stop_button.pack(pady=10)

# 创建 Matplotlib 图形和轴
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# 全局变量来保存刷新线程
refresh_thread = None

# 启动 GUI 事件循环
root.mainloop()
