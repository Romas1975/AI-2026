# analysis_full.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1️⃣ Load data
# -----------------------------
ticker = "SPY"
df = yf.download(ticker, period="1y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty:
    raise ValueError("Dataframe is empty. Yahoo returned no data.")

# -----------------------------
# 2️⃣ MACD signals
# -----------------------------
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9).mean()

# 1 = Buy, -1 = Sell, 0 = Hold
df["MACD_signal"] = np.where(df["MACD"] > df["Signal_line"], 1,
                             np.where(df["MACD"] < df["Signal_line"], -1, 0))

# -----------------------------
# 3️⃣ LSTM model for AI signal
# -----------------------------
# Prepare data
df["Market_returns"] = df["Close"].pct_change()
df.dropna(inplace=True)
values = df["Market_returns"].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

X, y = [], []
window = 20
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i,0])
    y.append(scaled[i,0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1],1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Predict
preds = model.predict(X)
df = df.iloc[window:]  # Align with predictions
df["LSTM_signal"] = np.where(preds.flatten() > 0, 1, -1)

# -----------------------------
# 4️⃣ Combine signals (example: MACD + LSTM)
# -----------------------------
df["AI_signal"] = df["MACD_signal"] * df["LSTM_signal"]

# Strategy returns
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"]
df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()
df["Drawdown_market"] = df["Cumulative_market"] / df["Cumulative_market"].cummax() - 1
df["Drawdown_strategy"] = df["Cumulative_strategy"] / df["Cumulative_strategy"].cummax() - 1

# -----------------------------
# 5️⃣ Metrics
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    return (cumulative / roll_max - 1).min()

exposure_mean = df["AI_signal"].replace(-1,0).mean()
buy_signals = (df["AI_signal"] == 1).sum()
sell_signals = (df["AI_signal"] == -1).sum()

metrics_text = f"""
Total Market Return: {df['Cumulative_market'].iloc[-1]-1:.3f}
Total Strategy Return: {df['Cumulative_strategy'].iloc[-1]-1:.3f}
Market Sharpe: {sharpe_ratio(df['Market_returns']):.3f}
Strategy Sharpe: {sharpe_ratio(df['Strategy_returns']):.3f}
Market Max Drawdown: {max_drawdown(df['Cumulative_market']):.3f}
Strategy Max Drawdown: {max_drawdown(df['Cumulative_strategy']):.3f}
Strategy Exposure (mean): {exposure_mean:.3f}
Number of Buy signals: {buy_signals}
Number of Sell signals: {sell_signals}
"""

print(metrics_text)

# -----------------------------
# 6️⃣ Dash Dashboard
# -----------------------------
app = dash.Dash(__name__)
app.title = "AI Strategy Dashboard"

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold", style={'textAlign':'center'}),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df.index.min(),
        end_date=df.index.max(),
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max()
    ),
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('metrics','children'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(start_date, end_date):
    dff = df.loc[start_date:end_date]

    # Cumulative returns
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Market', line=dict(color='blue')))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='Strategy', line=dict(color='green')))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # Drawdowns
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market DD', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy DD', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # Rolling Sharpe
    window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(window).mean() / dff['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(window).mean() / dff['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.update_layout(title='Rolling Sharpe', xaxis_title='Date')

    # Metrics text
    metrics_display = [
        html.P(f"Total Market Return: {dff['Cumulative_market'].iloc[-1]-1:.3f}"),
        html.P(f"Total Strategy Return: {dff['Cumulative_strategy'].iloc[-1]-1:.3f}"),
        html.P(f"Market Sharpe: {sharpe_ratio(dff['Market_returns']):.3f}"),
        html.P(f"Strategy Sharpe: {sharpe_ratio(dff['Strategy_returns']):.3f}"),
        html.P(f"Market Max Drawdown: {max_drawdown(dff['Cumulative_market']):.3f}"),
        html.P(f"Strategy Max Drawdown: {max_drawdown(dff['Cumulative_strategy']):.3f}"),
        html.P(f"Strategy Exposure (mean): {dff['AI_signal'].replace(-1,0).mean():.3f}"),
        html.P(f"Number of Buy signals: {(dff['AI_signal']==1).sum()}"),
        html.P(f"Number of Sell signals: {(dff['AI_signal']==-1).sum()}")
    ]

    return fig_cum, fig_dd, fig_rolling, metrics_display

if __name__ == "__main__":
    app.run(debug=True)