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
import talib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging


# 配置日志记录
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API URL for SOL price and volume
API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT"

# 线程锁，用于保护共享资源
price_lock = threading.Lock()

# 模型文件路径
MODEL_FILE ='sol_price_rnn_model.pth'

# 全局定义 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义更复杂的 LSTM 模型，结合注意力机制和 Transformer
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        output = x * attn_weights
        return output.sum(dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        linear_output = self.linear(x)
        x = x + linear_output
        x = self.norm2(x)
        return x


class SolPriceComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SolPriceComplexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionBlock(hidden_size)
        self.transformer = TransformerBlock(hidden_size, num_heads=4)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.attention(out)
        out = self.transformer(out.unsqueeze(0)).squeeze(0)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 尝试加载已保存的模型或创建新模型
def load_or_create_model():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    input_size = 7
    hidden_size = 1024
    num_layers = 8
    output_size = 1
    model = SolPriceComplexLSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            logging.info("成功加载现有模型")
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
    else:
        logging.info("未找到模型文件，创建新模型")
        # 这里不使用模拟数据，新模型创建后直接返回，等待实时数据训练
        return model

    return model


# 获取 SOL 价格和交易量，同时计算买卖价差、VWAP、RSI
def get_sol_price_and_volume():
    try:
        response_price = requests.get(API_URL_PRICE)
        response_price.raise_for_status()
        data_price = response_price.json()
        price = float(data_price['price'])

        response_volume = requests.get(API_URL_VOLUME)
        response_volume.raise_for_status()
        data_volume = response_volume.json()
        volume = float(data_volume['volume'])

        # 假设没有直接获取买卖价差的 API，这里暂时不计算
        # bid_price = price - np.random.rand() * 0.1
        # ask_price = price + np.random.rand() * 0.1
        # bid_ask_spread = ask_price - bid_price
        bid_ask_spread = 0

        vwap = talib.VWAP(np.array(prices), np.array(volumes))[-1] if prices else 0
        rsi = talib.RSI(np.array(prices), timeperiod=14)[-1] if prices else 50

        # 这里没有市场总市值和社交媒体热度的实时 API，暂时设为 0
        market_cap = 0
        social_heat = 0

        return price, volume, bid_ask_spread, vwap, rsi, market_cap, social_heat
    except requests.RequestException as e:
        if isinstance(e, requests.HTTPError):
            logging.error(f"请求 API 时出错，状态码: {e.response.status_code}")
        else:
            logging.error(f"请求 API 时出错: {e}")
    except KeyError as e:
        logging.error(f"数据格式错误，未找到键 {e}，原始数据: {data_price if 'data_price' in locals() else data_volume}")
    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    return None, None, None, None, None, None, None


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
        y.append(scaled_data[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X, y, scaler


# 训练 RNN 模型，使用更复杂的优化器和多 GPU 支持，添加学习率预热
def train_model(model, X, y, num_epochs=100, warmup_epochs=10):
    best_loss = float('inf')
    patience = 50
    counter = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001 * (epoch + 1) / warmup_epochs
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

        model.train()
        if X.dim()!= 3:
            logging.error(f"输入 X 的维度错误，期望 3D，实际 {X.dim()}D，形状为 {X.shape}")
            continue
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(), MODEL_FILE)
        else:
            counter += 1

        if counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 定期保存检查点
        if (epoch + 1) % 50 == 0 and dist.get_rank() == 0:
            checkpoint_path = f'sol_price_rnn_model_epoch_{epoch + 1}.pth'
            torch.save(model.module.state_dict(), checkpoint_path)

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


# 计算评估指标
def calculate_metrics(y_true, y_pred):
    mse = nn.MSELoss()(y_pred, y_true)
    rmse = torch.sqrt(mse)
    mae = nn.L1Loss()(y_pred, y_true)
    return mse.item(), rmse.item(), mae.item()


# RNN 预测
def predict_trend_rnn(data, scaler):
    if len(data) < 2:
        return "数据不足，无法预测"
    look_back = 60
    last_sequence = data[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    if last_sequence_tensor.dim()!= 3:
        logging.error(f"预测输入的维度错误，期望 3D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
        return "输入维度错误，无法预测"
    model.eval()
    with torch.no_grad():
        predicted_price_scaled = model(last_sequence_tensor).item()
    prediction_input = np.zeros((1, 7))
    prediction_input[0, 0] = predicted_price_scaled
    prediction = scaler.inverse_transform(prediction_input)[0, 0]
    last_price = data[-1, 0]
    if prediction > last_price:
        return "预计上涨"
    elif prediction < last_price:
        return "预计下跌"
    else:
        return "趋势不明"


# 预测大概量
def predict_amount(data, scaler):
    if len(data) < 2:
        return "数据不足，无法预测"
    look_back = 60
    last_sequence = data[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    if last_sequence_tensor.dim()!= 3:
        logging.error(f"预测输入的维度错误，期望 3D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
        return "输入维度错误，无法预测"
    model.eval()
    with torch.no_grad():
        predicted_price_scaled = model(last_sequence_tensor).item()
    prediction_input = np.zeros((1, 7))
    prediction_input[0, 0] = predicted_price_scaled
    prediction = scaler.inverse_transform(prediction_input)[0, 0]
    last_price = data[-1, 0]
    price_std = np.std(data[:, 0]) if len(data) > 1 else 0
    if prediction > last_price:
        amount = price_std * 0.5
    elif prediction < last_price:
        amount = -price_std * 0.5
    else:
        amount = 0
    return amount


# 更新价格、走势图和预测结果
def update_price_and_chart():
    global prices, volumes, timestamps
    price, volume, bid_ask_spread, vwap, rsi, market_cap, social_heat = get_sol_price_and_volume()
    if price is not None and volume is not None:
        with price_lock:
            prices.append(price)
            volumes.append(volume)
            timestamps.append(time.time())
        price_label.config(text=f"SOL 价格: ${price:.2f}")

        data = np.column_stack((prices, volumes, bid_ask_spread, vwap, rsi, market_cap, social_heat))
        if len(data) >= 2:
            result = prepare_data(data)
            if result is not None:
                X, y, scaler = result
                train_model(model, X, y, num_epochs=10)
                trend = predict_trend_rnn(data, scaler)
                amount = predict_amount(data, scaler)
                show_prediction_in_window(trend, amount)
                update_chart()
    root.after(1000, update_price_and_chart)


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
        refresh_thread.join()


# 在窗口显示预测结果和时间
def show_prediction_in_window(trend, amount):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
    amount_text = f"预测大概量: {'上涨' if amount > 0 else '下跌' if amount < 0 else '无变化'} {abs(amount):.2f} 美元"
    prediction_label.config(text=f"{prediction_text}\n{amount_text}")


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

# 全局变量来保存