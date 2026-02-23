import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- 1. Data load ---
symbol = "AAPL"
df = yf.download(symbol, start="2022-01-01", end="2026-01-01")
df = df[['Close']]  # dirbame tik su uždarymo kaina

# --- 2. Patikrink MultiIndex ---
if isinstance(df.index, pd.MultiIndex):
    df.index = df.index.get_level_values(0)
df = df.copy()

# --- 3. MACD signalai ---
ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
macd = ema_fast - ema_slow
signal = macd.ewm(span=9, adjust=False).mean()

df["MACD_signal"] = 0
df.loc[macd > signal, "MACD_signal"] = 1
df.loc[macd < signal, "MACD_signal"] = -1

# --- 4. LSTM preparation ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]])

def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(scaled_data, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --- 5. LSTM model ---
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# --- 6. LSTM prediction ---
predicted = model.predict(X)
df = df.iloc[look_back:].copy()
df["LSTM_pred"] = scaler.inverse_transform(predicted)

df["LSTM_signal"] = 0
df.loc[df["LSTM_pred"].shift(1) < df["Close"], "LSTM_signal"] = 1
df.loc[df["LSTM_pred"].shift(1) > df["Close"], "LSTM_signal"] = -1

# --- 7. Dash app ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1(f"{symbol} MACD & LSTM Dashboard"),
    dcc.Graph(
        id="price-chart",
        figure={
            "data": [
                {"x": df.index, "y": df["Close"], "type": "line", "name": "Close"},
                {"x": df.index, "y": df["LSTM_pred"], "type": "line", "name": "LSTM_pred"},
            ],
            "layout": {"title": f"{symbol} Price & LSTM Prediction"}
        }
    ),
    dcc.Graph(
        id="macd-chart",
        figure={
            "data": [
                {"x": df.index, "y": macd.iloc[look_back:], "type": "line", "name": "MACD"},
                {"x": df.index, "y": signal.iloc[look_back:], "type": "line", "name": "Signal"},
            ],
            "layout": {"title": "MACD & Signal"}
        }
    )
])

if __name__ == "__main__":
    # Nauja Dash sintaksė
    app.run(debug=True)