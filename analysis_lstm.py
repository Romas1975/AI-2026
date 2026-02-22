import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1️⃣ Load data
# -----------------------------
ticker = "AAPL"
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
df["MACD_signal"] = np.where(df["MACD"] > df["Signal_line"], 1, 0)

# -----------------------------
# 3️⃣ LSTM dummy signal
# -----------------------------
# Use previous 5 closes to predict next day direction
window = 5
df["Return"] = df["Close"].pct_change().shift(-1)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[["Close"]])

X, y = [], []
for i in range(window, len(scaled_close)):
    X.append(scaled_close[i-window:i])
    y.append(1 if df["Return"].iloc[i] > 0 else 0)
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

lstm_pred = model.predict(X, verbose=0)
df = df.iloc[window:]  # align
df["LSTM_signal"] = (lstm_pred.flatten() > 0.5).astype(int)

# -----------------------------
# 4️⃣ Strategy returns
# -----------------------------
df["Market_returns"] = df["Close"].pct_change()
df["MACD_strategy"] = df["Market_returns"] * df["MACD_signal"]
df["LSTM_strategy"] = df["Market_returns"] * df["LSTM_signal"]

df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_MACD"] = (1 + df["MACD_strategy"]).cumprod()
df["Cumulative_LSTM"] = (1 + df["LSTM_strategy"]).cumprod()

df["Drawdown_market"] = df["Cumulative_market"]/df["Cumulative_market"].cummax() - 1
df["Drawdown_MACD"] = df["Cumulative_MACD"]/df["Cumulative_MACD"].cummax() - 1
df["Drawdown_LSTM"] = df["Cumulative_LSTM"]/df["Cumulative_LSTM"].cummax() - 1

# -----------------------------
# 5️⃣ Metrics functions
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252)*returns.mean()/returns.std()

def max_drawdown(cumulative):
    return (cumulative/cumulative.cummax() - 1).min()

# -----------------------------
# 6️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "AI Strategy vs Buy & Hold"

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),
    dcc.Dropdown(
        id='strategy-type',
        options=[
            {'label':'MACD', 'value':'MACD'},
            {'label':'LSTM', 'value':'LSTM'}
        ],
        value='MACD',
        clearable=False
    ),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df.index.min(),
        end_date=df.index.max(),
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max()
    ),
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 7️⃣ Callback
# -----------------------------
@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('metrics','children'),
    Input('strategy-type','value'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(strategy, start_date, end_date):
    dff = df.loc[start_date:end_date]

    # select strategy
    if strategy=='MACD':
        strat_col = "MACD_strategy"
        cum_col = "Cumulative_MACD"
        draw_col = "Drawdown_MACD"
        signal_col = "MACD_signal"
    else:
        strat_col = "LSTM_strategy"
        cum_col = "Cumulative_LSTM"
        draw_col = "Drawdown_LSTM"
        signal_col = "LSTM_signal"

    # --- Cumulative Returns
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff["Cumulative_market"], name="Buy & Hold", line=dict(color='blue')))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff[cum_col], name=f"{strategy} Strategy", line=dict(color='green')))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # --- Drawdowns
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff["Drawdown_market"], name="Market Drawdown", line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff[draw_col], name=f"{strategy} Drawdown", line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # --- Metrics
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns'].dropna()),3)}"),
        html.P(f"{strategy} Sharpe: {round(sharpe_ratio(dff[strat_col].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"{strategy} Max Drawdown: {round(max_drawdown(dff[cum_col]),3)}"),
        html.P(f"{strategy} Exposure: {round(dff[signal_col].mean(),3)}"),
        html.P(f"Number of Buy signals: {int((dff[signal_col]==1).sum())}"),
        html.P(f"Number of Sell signals: {int((dff[signal_col]==0).sum())}")
    ]

    return fig_cum, fig_dd, metrics_text

# -----------------------------
# 8️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)