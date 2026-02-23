import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta

# --- 1. Duomenų parsisiuntimas ir paruošimas ---
def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Close']]  # Naudojame tik uždarymo kainą
    return df

symbol = "AAPL"
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 metų duomenys
df = load_data(symbol, start_date, end_date)

# --- 2. MACD skaičiavimas ---
def calculate_macd(df):
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

macd, signal = calculate_macd(df)

# Generuojame MACD signalus
df["MACD_signal"] = 0
df.loc[macd > signal, "MACD_signal"] = 1
df.loc[macd < signal, "MACD_signal"] = -1

# --- 3. LSTM modelio paruošimas ---
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

# --- 4. LSTM modelio apmokymas ---
def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    return model

model = train_lstm(X, y)

# --- 5. LSTM prognozės generavimas ---
def generate_predictions(model, X, scaler, df, look_back=10):
    predicted = model.predict(X)
    df = df.iloc[look_back:].copy()
    df["LSTM_pred"] = scaler.inverse_transform(predicted)
    return df

df = generate_predictions(model, X, scaler, df, look_back=10)

# Generuojame LSTM signalus
df["LSTM_signal"] = 0
df.loc[df["LSTM_pred"].shift(1) < df["Close"], "LSTM_signal"] = 1  # Pirkimo signalas
df.loc[df["LSTM_pred"].shift(1) > df["Close"], "LSTM_signal"] = -1  # Pardavimo signalas

# --- 6. Dash interaktyvi priemonė ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"{symbol} MACD & LSTM Analizė"),
    dcc.Graph(
        id="price-chart",
        figure={
            "data": [
                {"x": df.index, "y": df["Close"], "type": "line", "name": "Uždarymo kaina"},
                {"x": df.index, "y": df["LSTM_pred"], "type": "line", "name": "LSTM prognozė"},
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
                {"x": df.index, "y": df["MACD_signal"], "type": "scatter", "mode": "markers", "name": "MACD signalai", "marker": {"color": "blue"}},
                {"x": df.index, "y": df["LSTM_signal"], "type": "scatter", "mode": "markers", "name": "LSTM signalai", "marker": {"color": "red"}},
            ],
            "layout": {"title": "Prekybos signalai (MACD ir LSTM)"}
        }
    )
])

# --- 7. Paleidimas ---
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
