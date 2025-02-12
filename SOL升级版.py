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
import os
from sklearn.preprocessing import MinMaxScaler
import asyncio
import aiohttp


# Binance API URL for SOL price and volume
API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT"

# 初始化价格、交易量和时间列表
prices = []
volumes = []
timestamps = []
# 线程锁，用于保护共享资源
price_lock = threading.Lock()

# 模型文件路径
MODEL_FILE ='sol_price_rnn_model.pth'

# 定义更复杂的 LSTM 模型
class SolPriceComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SolPriceComplexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 全局定义 device
device = torch.device("cpu")


# 尝试加载已保存的模型或创建新模型
def load_or_create_model():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    input_size = 2  # 价格和交易量两个特征
    hidden_size = 256  # 增加隐藏单元数量
    num_layers = 4  # 增加层数
    output_size = 1
    model = SolPriceComplexLSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            print("成功加载现有模型")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    else:
        print("未找到模型文件，创建新模型")
        # 这里可以添加一些初始训练逻辑，例如使用一些模拟数据进行简单训练
        mock_prices = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
        mock_volumes = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        mock_data = np.column_stack((mock_prices, mock_volumes))
        X, y = prepare_data(mock_data)
        if X is not None and y is not None:
            train_model(model, X, y, num_epochs=300)  # 增加训练轮数
            print("新模型训练完成并保存")

    return model


# 异步获取 SOL 价格和交易量
async def async_get_sol_price_and_volume():
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL_PRICE) as response_price:
            if response_price.status == 200:
                data_price = await response_price.json()
                price = float(data_price['price'])
            else:
                price = None

        async with session.get(API_URL_VOLUME) as response_volume:
            if response_volume.status == 200:
                data_volume = await response_volume.json()
                volume = float(data_volume['volume'])
            else:
                volume = None

        return price, volume


# 准备训练数据
def prepare_data(data):
    if len(data) < 2:
        return None, None
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    look_back = 60
    X = []
    y = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back, 0])  # 预测价格
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X, y, scaler


# 训练 RNN 模型
def train_model(model, X, y, num_epochs=100):
    best_loss = float('inf')
    patience = 20
    counter = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 减小学习率
    for epoch in range(num_epochs):
        model.train()
        if X.dim() != 3:
            print(f"输入 X 的维度错误，期望 3D，实际 {X.dim()}D，形状为 {X.shape}")
            continue
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# RNN 预测
def predict_trend_rnn(data, scaler):
    if len(data) < 2:
        return "数据不足，无法预测"
    look_back = 60
    last_sequence = data[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    if last_sequence_tensor.dim() != 3:
        print(f"预测输入的维度错误，期望 3D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
        return "输入维度错误，无法预测"
    model.eval()
    with torch.no_grad():
        predicted_price_scaled = model(last_sequence_tensor).item()
    prediction_input = np.zeros((1, 2))
    prediction_input[0, 0] = predicted_price_scaled
    prediction = scaler.inverse_transform(prediction_input)[0, 0]
    last_price = data[-1, 0]
    if prediction > last_price:
        return "预计上涨"
    elif prediction < last_price:
        return "预计下跌"
    else:
        return "趋势不明"


# 计算简单移动平均线
def calculate_sma(data, period):
    return np.convolve(data, np.ones(period) / period, mode='valid')


# 更新价格、走势图和预测结果
def update_price_and_chart():
    async def wrapper():
        price, volume = await async_get_sol_price_and_volume()
        if price is not None and volume is not None:
            with price_lock:
                prices.append(price)
                volumes.append(volume)
                timestamps.append(time.time())
            price_label.config(text=f"SOL 价格: ${price:.2f}")

            data = np.column_stack((prices, volumes))
            if len(data) >= 2:
                result = prepare_data(data)
                if result is not None:
                    X, y, scaler = result
                    train_model(model, X, y, num_epochs=10)  # 每次更新数据后进行一定轮数的训练
                    trend = predict_trend_rnn(data, scaler)
                    show_prediction_in_window(trend)
                    update_chart()
                    # 计算并显示简单移动平均线
                    sma_short = calculate_sma(prices, 10)
                    sma_long = calculate_sma(prices, 50)
                    if len(sma_short) > 0 and len(sma_long) > 0:
                        if sma_short[-1] > sma_long[-1]:
                            signal = "买入"
                        elif sma_short[-1] < sma_long[-1]:
                            signal = "卖出"
                        else:
                            signal = "持有"
                        show_trading_signal(signal)

        root.after(1000, lambda: asyncio.run_coroutine_threadsafe(wrapper(), loop).result())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(wrapper(), loop).result()


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
    prediction_window = tk.Toplevel(root)
    prediction_window.title("SOL 价格预测结果")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
    prediction_label = tk.Label(prediction_window, text=prediction_text, font=("Arial", 18))
    prediction_label.pack(pady=20)


# 显示交易信号
def show_trading_signal(signal):
    signal_window = tk.Toplevel(root)
    signal_window.title("交易信号")
    signal_label = tk.Label(signal_window, text=f"交易信号: {signal}", font=("Arial", 18))
    signal_label.pack(pady=20)


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
# 用于保存定时任务的 ID
refresh_id = None

# 加载或创建模型
model = load_or_create_model()

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