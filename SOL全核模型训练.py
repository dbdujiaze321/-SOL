import tkinter as tk
from tkinter import ttk
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Binance API URL for SOL price
API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"

# 初始化价格和时间列表
prices = []
timestamps = []
# 线程锁，用于保护共享资源
price_lock = threading.Lock()

# 模型文件路径
MODEL_FILE = 'sol_price_rnn_model.pth'

# 定义简单的 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 尝试加载已保存的模型
if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # 设备选择，优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(SimpleRNN(input_size=1, hidden_size=10, output_size=1))
    else:
        model = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
    model.to(device)

    try:
        model.load_state_dict(torch.load(MODEL_FILE))
    except FileNotFoundError:
        pass

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 获取 SOL 价格
    def get_sol_price():
        try:
            response = requests.get(API_URL)
            response.raise_for_status()  # 检查响应状态码，如果不是 200 会抛出异常
            data = response.json()
            price = float(data['price'])
            return price
        except requests.RequestException as e:
            print(f"请求 API 时出错: {e}")
        except KeyError as e:
            print(f"数据格式错误，未找到键 {e}，原始数据: {data}")
        except Exception as e:
            print(f"发生未知错误: {e}")
        return None

    # 准备训练数据
    def prepare_data(prices):
        if len(prices) < 2:
            return None, None
        X = np.array(prices[:-1]).reshape(-1, 1, 1)
        y = np.array(prices[1:]).reshape(-1, 1)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        return X, y

    # 训练 RNN 模型
    def train_model(model, X, y):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), MODEL_FILE)

    # RNN 预测
    def predict_trend_rnn(prices):
        if len(prices) < 2:
            return "数据不足，无法预测"
        last_price = np.array([prices[-1]]).reshape(-1, 1, 1)
        last_price = torch.tensor(last_price, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            predicted_price = model(last_price).item()
        if predicted_price > prices[-1]:
            return "预计上涨"
        elif predicted_price < prices[-1]:
            return "预计下跌"
        else:
            return "趋势不明"

    # 更新价格、走势图和预测结果
    def update_price_and_chart():
        global prices, timestamps
        price = get_sol_price()
        if price is not None:
            with price_lock:
                prices.append(price)
                timestamps.append(time.time())
            price_label.config(text=f"SOL 价格: ${price:.2f}")

            if len(prices) >= 2:
                X, y = prepare_data(prices)
                train_model(model, X, y)

            trend = predict_trend_rnn(prices)
            show_prediction_in_window(trend)
            update_chart()
        # 每 0.5 秒更新一次
        root.after(500, update_price_and_chart)

    # 更新走势图
    def update_chart():
        with price_lock:
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
            try:
                # 取消定时任务
                root.after_cancel(refresh_id)
            except NameError:
                pass
            refresh_thread = None

    # 在窗口显示预测结果和时间
    def show_prediction_in_window(trend):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
        prediction_label.config(text=prediction_text)

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

    # 创建预测结果窗口
    prediction_window = tk.Toplevel(root)
    prediction_window.title("SOL 价格预测结果")
    prediction_label = tk.Label(prediction_window, text="等待数据", font=("Arial", 18))
    prediction_label.pack(pady=20)

    # 全局变量来保存刷新线程
    refresh_thread = None
    # 用于保存定时任务的 ID
    refresh_id = None

    # 定义程序关闭时的处理函数
    def on_closing():
        stop_refresh()
        try:
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), MODEL_FILE)
        except Exception as e:
            print(f"保存模型时出错: {e}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 启动 GUI 事件循环
    refresh_id = root.after(0, update_price_and_chart)
    root.mainloop()
