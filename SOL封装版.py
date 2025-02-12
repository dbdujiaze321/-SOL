import sys
import tkinter as tk
from tkinter import ttk
import requests
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from sklearn.preprocessing import MinMaxScaler
from queue import Queue
import matplotlib.dates as mdates


def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发和 PyInstaller 打包 """
    try:
        # PyInstaller 创建一个临时文件夹并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 打印模块搜索路径，用于排查问题
print("sys.path:", sys.path)

# Binance API URL for SOL price and volume
API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT"

# 初始化价格、交易量和时间列表，使用 deque 优化内存
from collections import deque
prices = deque(maxlen=1000)
volumes = deque(maxlen=1000)
timestamps = deque(maxlen=1000)
# 线程锁，用于保护共享资源
price_lock = threading.Lock()
# 用于线程间通信的队列
data_queue = Queue()

# 模型文件路径
MODEL_FILE = resource_path('sol_price_rnn_model.pth')

# 全局定义 device
device = torch.device("cpu")

# 日志文件路径
LOG_FILE = resource_path('training.log')


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


# 封装异常处理函数
def handle_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"在函数 {func.__name__} 中发生错误: {e}"
            print(error_msg)
            # 可以在这里添加记录日志等操作
            root.after(0, lambda: tk.messagebox.showerror("错误", error_msg))
            sys.exit(1)
    return wrapper


# 尝试加载已保存的模型或创建新模型
@handle_exception
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
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print("成功加载现有模型")
    else:
        print("未找到模型文件，创建新模型")
        mock_prices = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
        mock_volumes = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        mock_data = np.column_stack((mock_prices, mock_volumes))
        X, y = prepare_data(mock_data)
        if X is not None and y is not None:
            train_model(model, X, y, num_epochs=300)  # 增加训练轮数
            print("新模型训练完成并保存")
    return model


# 获取 SOL 价格和交易量
@handle_exception
def get_sol_price_and_volume():
    responses = requests.get(API_URL_PRICE), requests.get(API_URL_VOLUME)
    for resp in responses:
        resp.raise_for_status()
    data_price, data_volume = (resp.json() for resp in responses)
    price = float(data_price['price'])
    volume = float(data_volume['volume'])
    print(f"获取到价格: {price}, 交易量: {volume}")
    return price, volume


# 准备训练数据
@handle_exception
def prepare_data(data):
    if len(data) < 60 + 1:
        print("数据长度不足，无法准备训练数据")
        return None, None, None
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    look_back = 60
    X = np.array([scaled_data[i:i + look_back] for i in range(len(scaled_data) - look_back)])
    y = np.array([scaled_data[i + look_back, 0] for i in range(len(scaled_data) - look_back)])
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    print(f"准备好训练数据，X 形状: {X.shape}, y 形状: {y.shape}")
    return X, y, scaler


# 训练 RNN 模型，使用学习率调整策略
@handle_exception
def train_model(model, X, y, num_epochs=100, lr=0.00001):
    assert X.dim() == 3, f"输入 X 的维度错误，期望 3D，实际 {X.dim()}D，形状为 {X.shape}"
    best_loss = float('inf')
    patience = 20
    counter = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    with open(LOG_FILE, 'a') as f:
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
                torch.save(model.state_dict(), MODEL_FILE)
            else:
                counter += 1

            if counter >= patience:
                f.write(f"Early stopping at epoch {epoch + 1}\n")
                print(f"早停，在第 {epoch + 1} 轮")
                break

            if (epoch + 1) % 50 == 0:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}\n')
                print(f"第 {epoch + 1} 轮，损失: {loss.item():.4f}")


# 提取预测的公共部分
@handle_exception
def predict_common(data, scaler):
    if len(data) < 60 + 1:
        print("数据长度不足，无法预测")
        return None
    look_back = 60
    last_sequence = np.array([data[-look_back:]])
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).to(device)
    assert last_sequence_tensor.dim() == 3, f"预测输入的维度错误，期望 3D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}"
    model.eval()
    with torch.no_grad():
        predicted_price_scaled = model(last_sequence_tensor).item()
    prediction_input = np.zeros((1, 2))
    prediction_input[0, 0] = predicted_price_scaled
    prediction = scaler.inverse_transform(prediction_input)[0, 0]
    print(f"预测价格: {prediction}")
    return prediction


# RNN 预测趋势
def predict_trend_rnn(data, scaler):
    prediction = predict_common(data, scaler)
    if prediction is None:
        return "数据不足，无法预测"
    last_price = data[-1, 0]
    if prediction > last_price:
        return "预计上涨"
    elif prediction < last_price:
        return "预计下跌"
    else:
        return "趋势不明"


# 预测大概量
def predict_amount(data, scaler):
    prediction = predict_common(data, scaler)
    if prediction is None:
        return "数据不足，无法预测"
    last_price = data[-1, 0]
    price_std = np.std(data[:, 0]) if len(data) > 1 else 0
    if prediction > last_price:
        amount = price_std * 0.5  # 假设上涨时大概量为标准差的0.5倍
    elif prediction < last_price:
        amount = -price_std * 0.5  # 假设下跌时大概量为标准差的 -0.5倍
    else:
        amount = 0
    print(f"预测大概量: {amount}")
    return amount


# 更新价格、走势图和预测结果
def update_price_and_chart_worker():
    while True:
        try:
            price, volume = get_sol_price_and_volume()
            if price is not None and volume is not None:
                with price_lock:
                    prices.append(price)
                    volumes.append(volume)
                    timestamps.append(time.time())
                data_queue.put((prices.copy(), timestamps.copy()))
        except Exception as e:
            print(f"更新数据时出错: {e}")
        time.sleep(1)


# 优化后的更新走势图
def update_chart():
    if not data_queue.empty():
        new_prices, new_timestamps = data_queue.get()
        new_x = np.array([mdates.epoch2num(t) for t in new_timestamps])
        new_y = np.array(new_prices)

        if line.get_xdata().size == 0:
            line.set_xdata(new_x)
            line.set_ydata(new_y)
            ax.relim()
            ax.autoscale_view()
        else:
            current_x = line.get_xdata()
            current_y = line.get_ydata()
            new_x = np.concatenate((current_x, new_x))
            new_y = np.concatenate((current_y, new_y))
            line.set_xdata(new_x)
            line.set_ydata(new_y)
            ax.relim()
            ax.autoscale_view()

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        canvas.draw()
        canvas.flush_events()

    root.after(100, update_chart)


# 开始刷新
def start_refresh():
    global refresh_thread
    if not refresh_thread or not refresh_thread.is_alive():
        try:
            refresh_thread = threading.Thread(target=update_price_and_chart_worker)
            refresh_thread.start()
            update_chart()
        except Exception as e:
            print(f"启动刷新线程时出错: {e}")


# 停止刷新
def stop_refresh():
    global refresh_thread
    if refresh_thread and refresh_thread.is_alive():
        try:
            refresh_thread.join()
        except Exception as e:
            print(f"停止刷新线程时出错: {e}")
        refresh_thread = None


# 在窗口显示预测结果和时间
def show_prediction_in_window(trend, amount):
    try:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
        amount_text = f"预测大概量: {'上涨' if amount > 0 else '下跌' if amount < 0 else '无变化'} {abs(amount):.2f} 美元"
        prediction_label.config(text=f"{prediction_text}\n{amount_text}")
        print(f"显示预测结果: 时间 {current_time}, 趋势 {trend}, 大概量 {amount}")
    except Exception as e:
        print(f"显示预测结果时出错: {e}")


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
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('#f0f0f5')
    fig.set_facecolor('#f0f0f5')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#555555')
    ax.tick_params(axis='y', colors='#555555')
    ax.xaxis.label.set_color('#555555')
    ax.yaxis.label.set_color('#555555')
    ax.title.set_color('#333333')
    line, = ax.plot([], [], marker='o', color='#007acc', linewidth=2)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(pady=20)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
except Exception as e:
    error_msg = f"创建图形和轴时出错: {e}"
    print(error_msg)
    root.after(0, lambda: tk.messagebox.showerror("错误", error_msg))
    sys.exit(1)

# 创建预测结果框架
prediction_frame = tk.Frame(root)
prediction_frame.pack(pady=20)

# 创建预测结果标签
prediction_label = tk.Label(prediction_frame, text="等待数据", font=("Arial", 18))
prediction_label.pack()

# 全局变量来保存刷新线程
refresh_thread = None
# 用于保存定时任务的 ID
refresh_id = None

# 加载或创建模型
model = load_or_create_model()

# 训练 RNN 模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# 定义程序关闭时的处理函数
def on_closing():
    stop_refresh()
    try:
        if hasattr(model,'state_dict'):
            torch.save(model.state_dict(), MODEL_FILE)
    except Exception as e:
        print(f"保存模型时出错: {e}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

try:
    root.after(0, start_refresh)
    root.mainloop()
except Exception as e:
    print(f"主循环中发生错误: {e}")