import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1️⃣ Load data
# -----------------------------
ticker = "SPY"
df = yf.download(ticker, period="1y", auto_adjust=True)

# MultiIndex fix
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# -----------------------------
# 2️⃣ Features
# -----------------------------
df["Market_returns"] = df["Close"].pct_change()
df["Vol_20"] = df["Market_returns"].rolling(20).std() * np.sqrt(252)

# MACD
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9).mean()

df["AI_signal_MACD"] = np.where(df["MACD"] > df["Signal_line"], 1, 0)

df.dropna(inplace=True)

# -----------------------------
# 3️⃣ LSTM Model for AI Signal
# -----------------------------
features = ["Close", "Market_returns", "Vol_20"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

X, y = [], []
seq_len = 5
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(df["AI_signal_MACD"].iloc[i])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

preds = model.predict(X, verbose=0).flatten()
df = df.iloc[seq_len:]
df["AI_signal_LSTM"] = (preds > 0.5).astype(int)

# Use LSTM signal as strategy
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal_LSTM"]
df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()
df["Drawdown_market"] = df["Cumulative_market"]/df["Cumulative_market"].cummax() - 1
df["Drawdown_strategy"] = df["Cumulative_strategy"]/df["Cumulative_strategy"].cummax() - 1

# -----------------------------
# 4️⃣ Metrics
# -----------------------------
def sharpe_ratio(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_drawdown(cum):
    return (cum / cum.cummax() - 1).min()

metrics = {
    "Total Market Return": df["Cumulative_market"].iloc[-1] - 1,
    "Total Strategy Return": df["Cumulative_strategy"].iloc[-1] - 1,
    "Market Sharpe": sharpe_ratio(df["Market_returns"]),
    "Strategy Sharpe": sharpe_ratio(df["Strategy_returns"]),
    "Market Max Drawdown": max_drawdown(df["Cumulative_market"]),
    "Strategy Max Drawdown": max_drawdown(df["Cumulative_strategy"]),
    "Strategy Exposure (mean)": df["AI_signal_LSTM"].mean(),
    "Buy signals": int(df["AI_signal_LSTM"].sum()),
    "Sell signals": int(len(df) - df["AI_signal_LSTM"].sum())
}

# -----------------------------
# 5️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "LSTM AI Strategy Dashboard"

app.layout = html.Div([
    html.H1("LSTM AI Strategy vs Market", style={"textAlign":"center"}),
    dcc.DatePickerRange(
        id="date-range",
        start_date=df.index.min(),
        end_date=df.index.max(),
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max()
    ),
    dcc.Graph(id="cum_returns"),
    dcc.Graph(id="drawdowns"),
    html.Div(id="metrics", style={"textAlign":"center", "fontSize":16})
])

@app.callback(
    Output("cum_returns","figure"),
    Output("drawdowns","figure"),
    Output("metrics","children"),
    Input("date-range","start_date"),
    Input("date-range","end_date")
)
def update_dashboard(start_date, end_date):
    dff = df.loc[start_date:end_date]

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff["Cumulative_market"], name="Buy & Hold"))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff["Cumulative_strategy"], name="AI Strategy"))

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff["Drawdown_market"], name="Market Drawdown"))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff["Drawdown_strategy"], name="Strategy Drawdown"))

    metrics_text = [html.P(f"{k}: {round(v,3)}") for k,v in metrics.items()]
    return fig_cum, fig_dd, metrics_text

if __name__ == "__main__":
    app.run(debug=True)