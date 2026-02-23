import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import dcc, html
import plotly.graph_objs as go

# --- 1. Duomenų parsisiuntimas ---
def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=df.index.names.difference(['Date']), drop=True)
    return df[['Close']].copy()

symbol = "AAPL"
df = load_data(symbol, start_date="2022-01-01", end_date="2026-01-01")

# --- 2. MACD skaičiavimas ---
def calculate_macd(df):
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

macd, signal = calculate_macd(df)

# --- 3. MACD signalų generavimas ---
df["MACD_signal"] = np.where(macd > signal, 1, np.where(macd < signal, -1, 0))

# --- 4. LSTM duomenų paruošimas ---
def prepare_lstm_data(df, look_back=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Close"]].values)

    def create_dataset(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

X, y, scaler = prepare_lstm_data(df, look_back=10)

# --- 5. LSTM modelio apmokymas ---
def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model

model = train_lstm(X, y)

# --- 6. LSTM prognozių generavimas ---
predicted = model.predict(X)
df_lstm = df.iloc[10:].copy()
df_lstm["LSTM_pred"] = scaler.inverse_transform(predicted).flatten()

# --- 7. LSTM signalų generavimas ---
lstm_pred_aligned, close_aligned = df_lstm["LSTM_pred"].align(df_lstm["Close"], axis=0, copy=False)

# Išmetame pirmą eilutę, kuri tampa NaN po shift(1)
lstm_pred_shifted = lstm_pred_aligned.iloc[1:].reset_index(drop=True)
close_shifted = close_aligned.iloc[1:].reset_index(drop=True)

# Generuojame signalus
df_lstm["LSTM_signal"] = 0
df_lstm.iloc[1:].loc[lstm_pred_shifted < close_shifted, "LSTM_signal"] = 1
df_lstm.iloc[1:].loc[lstm_pred_shifted > close_shifted, "LSTM_signal"] = -1

# --- 8. Dash interaktyvi priemonė ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"{symbol} MACD & LSTM Analizė"),
    dcc.Graph(
        id="price-chart",
        figure={
            "data": [
                {"x": df_lstm.index, "y": df_lstm["Close"], "type": "line", "name": "Uždarymo kaina"},
                {"x": df_lstm.index, "y": df_lstm["LSTM_pred"], "type": "line", "name": "LSTM prognozė"},
            ],
            "layout": {"title": f"{symbol} Kainos ir LSTM prognozės"}
        }
    ),
    dcc.Graph(
        id="macd-chart",
        figure={
            "data": [
                {"x": df.index, "y": macd, "type": "line", "name": "MACD"},
                {"x": df.index, "y": signal, "type": "line", "name": "Signalas"},
            ],
            "layout": {"title": "MACD ir Signalas"}
        }
    ),
    dcc.Graph(
        id="signals-chart",
        figure={
            "data": [
                {"x": df_lstm.index[1:], "y": df_lstm["MACD_signal"].iloc[1:], "type": "scatter", "mode": "markers", "name": "MACD signalai", "marker": {"color": "blue"}},
                {"x": df_lstm.index[1:], "y": df_lstm["LSTM_signal"].iloc[1:], "type": "scatter", "mode": "markers", "name": "LSTM signalai", "marker": {"color": "red"}},
            ],
            "layout": {"title": "Prekybos signalai (MACD ir LSTM)"}
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

print(df_lstm.shape)
print(predicted.shape)
