import tkinter as tk
from tkinter import ttk
import asyncio
import aiohttp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from sklearn.preprocessing import MinMaxScaler


class SolPricePredictor:
    API_URL_PRICE = "https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT"
    API_URL_VOLUME = "https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT"
    MODEL_FILE ='sol_price_rnn_model.pth'
    DATA_LENGTH_LIMIT = 1000
    LOOK_BACK = 60
    DEVICE = torch.device("cpu")

    def __init__(self, root):
        self.root = root
        self.root.title("SOL 实时价格查看器")

        self.prices = []
        self.volumes = []
        self.timestamps = []
        self.price_lock = threading.Lock()
        self.refresh_thread = None
        self.refresh_id = None

        self.model = self.load_or_create_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

        self.create_widgets()
        self.update_price_and_chart()

    def create_widgets(self):
        self.price_label = tk.Label(self.root, text="正在获取价格...", font=("Arial", 24))
        self.price_label.pack(pady=20)

        start_frame = ttk.Frame(self.root)
        start_frame.pack(pady=10)

        start_button = ttk.Button(start_frame, text="开始刷新", command=self.start_refresh)
        start_button.pack(side=tk.LEFT, padx=5)

        stop_button = ttk.Button(start_frame, text="停止刷新", command=self.stop_refresh)
        stop_button.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=20)

    def load_or_create_model(self):
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        input_size = 2
        hidden_size = 256
        num_layers = 4
        output_size = 1
        model = SolPriceComplexLSTM(input_size, hidden_size, num_layers, output_size).to(self.DEVICE)

        if os.path.exists(self.MODEL_FILE):
            try:
                model.load_state_dict(torch.load(self.MODEL_FILE, map_location=self.DEVICE))
                print("成功加载现有模型")
            except Exception as e:
                print(f"加载模型时出错: {e}")
        else:
            print("未找到模型文件，创建新模型")
            mock_prices = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
            mock_volumes = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
            mock_data = np.column_stack((mock_prices, mock_volumes))
            X, y = self.prepare_data(mock_data)
            if X is not None and y is not None:
                self.train_model(model, X, y, num_epochs=300)
                print("新模型训练完成并保存")

        return model

    async def async_get_sol_price_and_volume(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.API_URL_PRICE) as response_price:
                price = await self.fetch_price(response_price)
            async with session.get(self.API_URL_VOLUME) as response_volume:
                volume = await self.fetch_volume(response_volume)
        return price, volume

    async def fetch_price(self, response):
        if response.status == 200:
            data = await response.json()
            return float(data['price'])
        return None

    async def fetch_volume(self, response):
        if response.status == 200:
            data = await response.json()
            return float(data['volume'])
        return None

    def prepare_data(self, data):
        if len(data) < 2:
            return None, None
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        X = []
        y = []
        for i in range(len(scaled_data) - self.LOOK_BACK):
            X.append(scaled_data[i:i + self.LOOK_BACK])
            y.append(scaled_data[i + self.LOOK_BACK, 0])
        X = np.array(X)
        y = np.array(y)
        X = torch.tensor(X, dtype=torch.float32).to(self.DEVICE)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.DEVICE)
        return X, y, scaler

    def train_model(self, model, X, y, num_epochs=100):
        best_loss = float('inf')
        patience = 20
        counter = 0

        for epoch in range(num_epochs):
            model.train()
            if X.dim() != 3:
                print(f"输入 X 的维度错误，期望 3D，实际 {X.dim()}D，形状为 {X.shape}")
                continue
            outputs = model(X)
            loss = self.criterion(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
                torch.save(model.state_dict(), self.MODEL_FILE)
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict_trend_rnn(self, data, scaler):
        if len(data) < 2:
            return "数据不足，无法预测"
        last_sequence = data[-self.LOOK_BACK:]
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)
        if last_sequence_tensor.dim() != 3:
            print(f"预测输入的维度错误，期望 3D，实际 {last_sequence_tensor.dim()}D，形状为 {last_sequence_tensor.shape}")
            return "输入维度错误，无法预测"
        self.model.eval()
        with torch.no_grad():
            predicted_price_scaled = self.model(last_sequence_tensor).item()
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

    def calculate_sma(self, data, period):
        return np.convolve(data, np.ones(period) / period, mode='valid')

    def update_data(self, price, volume):
        with self.price_lock:
            self.prices.append(price)
            self.volumes.append(volume)
            self.timestamps.append(time.time())
            if len(self.prices) > self.DATA_LENGTH_LIMIT:
                self.prices.pop(0)
                self.volumes.pop(0)
                self.timestamps.pop(0)

    async def update_price_and_chart(self):
        price, volume = await self.async_get_sol_price_and_volume()
        if price is not None and volume is not None:
            self.update_data(price, volume)
            self.price_label.config(text=f"SOL 价格: ${price:.2f}")

            data = np.column_stack((self.prices, self.volumes))
            if len(data) >= 2:
                result = self.prepare_data(data)
                if result is not None:
                    X, y, scaler = result
                    # 检查是否需要训练模型，减少不必要的训练
                    if len(self.prices) % 5 == 0:
                        self.train_model(self.model, X, y, num_epochs=10)
                    trend = self.predict_trend_rnn(data, scaler)
                    self.show_prediction_in_window(trend)
                    self.update_chart()
                    sma_short = self.calculate_sma(self.prices, 10)
                    sma_long = self.calculate_sma(self.prices, 50)
                    if len(sma_short) > 0 and len(sma_long) > 0:
                        if sma_short[-1] > sma_long[-1]:
                            signal = "买入"
                        elif sma_short[-1] < sma_long[-1]:
                            signal = "卖出"
                        else:
                            signal = "持有"
                        self.show_trading_signal(signal)

        self.refresh_id = self.root.after(1000, lambda: asyncio.run_coroutine_threadsafe(self.update_price_and_chart(), self.get_event_loop()).result())

    def start_refresh(self):
        if not self.refresh_thread or not self.refresh_thread.is_alive():
            self.refresh_thread = threading.Thread(target=lambda: asyncio.run_coroutine_threadsafe(self.update_price_and_chart(), self.get_event_loop()).result())
            self.refresh_thread.start()

    def stop_refresh(self):
        if self.refresh_thread and self.refresh_thread.is_alive():
            try:
                self.root.after_cancel(self.refresh_id)
            except NameError:
                pass
            self.refresh_thread = None

    def show_prediction_in_window(self, trend):
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("SOL 价格预测结果")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_text = f"时间: {current_time}\n预测趋势: {trend}"
        prediction_label = tk.Label(prediction_window, text=prediction_text, font=("Arial", 18))
        prediction_label.pack(pady=20)

    def show_trading_signal(self, signal):
        signal_window = tk.Toplevel(self.root)
        signal_window.title("交易信号")
        signal_label = tk.Label(signal_window, text=f"交易信号: {signal}", font=("Arial", 18))
        signal_label.pack(pady=20)

    def update_chart(self):
        with self.price_lock:
            self.ax.clear()
            self.ax.plot(self.timestamps, self.prices, marker='o')
            self.ax.set_xlabel('时间')
            self.ax.set_ylabel('SOL 价格 (美元)')
            self.ax.set_title('SOL 价格走势')
            self.canvas.draw()

    def on_closing(self):
        self.stop_refresh()
        try:
            if hasattr(self.model,'state_dict'):
                torch.save(self.model.state_dict(), self.MODEL_FILE)
        except Exception as e:
            print(f"保存模型时出错: {e}")
        self.root.destroy()

    def get_event_loop(self):
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.new_event_loop()


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


if __name__ == "__main__":
    root = tk.Tk()
    app = SolPricePredictor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()