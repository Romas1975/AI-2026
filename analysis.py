import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import html, dcc
import plotly.graph_objects as go

# --- 1. Duomenų paruošimas ---
ticker = "AAPL"
df = yf.download(ticker, period="2y", interval="1d")

# Papildomi stulpeliai
df["Return"] = df["Close"].pct_change()
df["Vol_20"] = df["Return"].rolling(20).std() * np.sqrt(252)

# --- 2. MACD signalas ---
ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
macd = ema_fast - ema_slow
signal = macd.ewm(span=9, adjust=False).mean()
df["MACD_signal"] = 0
df.loc[macd > signal, "MACD_signal"] = 1
df.loc[macd < signal, "MACD_signal"] = -1

# --- 3. LSTM modelio paruošimas ---
scaler = MinMaxScaler()
df["Close_scaled"] = scaler.fit_transform(df[["Close"]])

# Sudarom X, y
sequence_length = 10
X, y = [], []
for i in range(sequence_length, len(df)):
    X.append(df["Close_scaled"].iloc[i-sequence_length:i].values)
    y.append(df["Close_scaled"].iloc[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM modelis
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# --- 4. LSTM prognozės ---
lstm_pred = model.predict(X)
lstm_pred_rescaled = scaler.inverse_transform(lstm_pred)

# Pridedam prie df (suderinant ilgį)
df = df.iloc[sequence_length:]
df["LSTM_pred"] = lstm_pred_rescaled

# --- 5. LSTM signalai su alignment pataisa ---
df["LSTM_pred_shifted"] = df["LSTM_pred"].shift(1)
df["LSTM_signal"] = 0
df.loc[df["LSTM_pred_shifted"] < df["Close"], "LSTM_signal"] = 1   # buy
df.loc[df["LSTM_pred_shifted"] > df["Close"], "LSTM_signal"] = -1  # sell
df.drop(columns=["LSTM_pred_shifted"], inplace=True)

# --- 6. Statistikos apskaičiavimas ---
strategy_returns = df["Return"] * df["LSTM_signal"].shift(1)
total_market_return = df["Return"].sum()
total_strategy_return = strategy_returns.sum()
market_sharpe = df["Return"].mean() / df["Return"].std() * np.sqrt(252)
strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
market_max_dd = (df["Close"] / df["Close"].cummax() - 1).min()
strategy_max_dd = (strategy_returns.cumsum() / (strategy_returns.cumsum().cummax()) - 1).min()
strategy_exposure = df["LSTM_signal"].abs().mean()
buy_signals = (df["LSTM_signal"] == 1).sum()
sell_signals = (df["LSTM_signal"] == -1).sum()

print(f"Total Market Return: {total_market_return:.3f}")
print(f"Total Strategy Return: {total_strategy_return:.3f}")
print(f"Market Sharpe: {market_sharpe:.3f}")
print(f"Strategy Sharpe: {strategy_sharpe:.3f}")
print(f"Market Max Drawdown: {market_max_dd:.3f}")
print(f"Strategy Max Drawdown: {strategy_max_dd:.3f}")
print(f"Strategy Exposure (mean): {strategy_exposure:.3f}")
print(f"Number of Buy signals: {buy_signals}")
print(f"Number of Sell signals: {sell_signals}")

# --- 7. Dash vizualizacija ---
app = dash.Dash(__name__)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
fig.add_trace(go.Scatter(x=df.index, y=df["LSTM_pred"], mode="lines", name="LSTM Pred"))
fig.add_trace(go.Bar(x=df.index, y=df["LSTM_signal"], name="LSTM Signal"))

app.layout = html.Div([
    html.H1("Trading Dashboard"),
    dcc.Graph(figure=fig)
])

# Nauja Dash sintaksė
if __name__ == "__main__":
    app.run(debug=True)