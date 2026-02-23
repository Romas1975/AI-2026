import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ------------------------------
# 1️⃣ Duomenų parsisiuntimas
# ------------------------------
ticker = "AAPL"
df = yf.download(ticker, start="2022-01-01", end="2023-01-01")
df = df[['Close']]

# ------------------------------
# 2️⃣ Normalizacija
# ------------------------------
scaler = MinMaxScaler()
df["Close_scaled"] = scaler.fit_transform(df[["Close"]])

# ------------------------------
# 3️⃣ Paruošiame LSTM duomenis
# ------------------------------
lookback = 5
X, y = [], []
for i in range(lookback, len(df)):
    X.append(df["Close_scaled"].values[i-lookback:i])
    y.append(1 if df["Close"].values[i] > df["Close"].values[i-1] else 0)

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# ------------------------------
# 4️⃣ LSTM modelis (super lengvas)
# ------------------------------
model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(X, y, epochs=5, batch_size=8, verbose=1)  # mažai epoch, greitai

# ------------------------------
# 5️⃣ Prognozės
# ------------------------------
df_pred = df.iloc[lookback:].copy()
df_pred["LSTM_signal"] = model.predict(X, verbose=0)
df_pred["LSTM_signal_binary"] = (df_pred["LSTM_signal"] > 0.5).astype(int)

# ------------------------------
# 6️⃣ Dash dashboard
# ------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(f"LSTM Trading Signals for {ticker}"),
    dcc.Graph(
        id="graph",
        figure={
            "data": [
                {"x": df.index, "y": df["Close"], "type": "line", "name": "Close Price"},
                {"x": df_pred.index, "y": df_pred["LSTM_signal_binary"]*df["Close"].max(),
                 "type": "line", "name": "LSTM Buy Signals"}
            ],
            "layout": {"title": f"{ticker} Price & LSTM Signals"}
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)