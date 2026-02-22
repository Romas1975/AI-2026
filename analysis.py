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
df = yf.download(ticker, period="2y", auto_adjust=True)

# Ensure single-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Check if empty
if df.empty:
    raise ValueError("Yahoo returned no data")

# -----------------------------
# 2️⃣ MACD signal (optional benchmark)
# -----------------------------
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['Signal_line'] = df['MACD'].ewm(span=9).mean()
df['MACD_signal'] = np.where(df['MACD'] > df['Signal_line'], 1, 0)

# -----------------------------
# 3️⃣ LSTM preprocessing
# -----------------------------
window = 20
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(window, len(df)):
    X.append(df['Close_scaled'].values[i-window:i])
    y.append(df['Close_scaled'].values[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# 4️⃣ LSTM model
# -----------------------------
model = Sequential([
    LSTM(50, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Predict next day prices
y_pred = model.predict(X, verbose=0)
df['LSTM_pred'] = np.nan
df['LSTM_pred'].iloc[window:] = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

# -----------------------------
# 5️⃣ Generate AI signals from LSTM
# -----------------------------
df['AI_signal'] = 0
df['AI_signal'].iloc[window:] = (df['LSTM_pred'].iloc[window:] > df['Close'].iloc[window:]).astype(int)

# -----------------------------
# 6️⃣ Compute returns & drawdowns
# -----------------------------
df['Market_returns'] = df['Close'].pct_change()
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
df['Drawdown_market'] = df['Cumulative_market']/df['Cumulative_market'].cummax() - 1
df['Drawdown_strategy'] = df['Cumulative_strategy']/df['Cumulative_strategy'].cummax() - 1

# Metrics functions
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

# -----------------------------
# 7️⃣ Dash app
# -----------------------------
app = dash.Dash(__name__)
app.title = "LSTM AI Strategy Dashboard"

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
    dcc.Graph(id='heatmap_returns'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 8️⃣ Callback
# -----------------------------
@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('heatmap_returns','figure'),
    Output('metrics','children'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(start_date, end_date):
    dff = df.loc[start_date:end_date]

    # Cumulative returns
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))

    # Drawdowns
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market Drawdown'))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy Drawdown'))

    # Rolling metrics
    window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(window).mean() / dff['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(window).mean() / dff['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = dff['Cumulative_market'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = dff['Cumulative_strategy'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Rolling Sharpe'))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe'))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_market, name='Market Rolling DD'))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_strategy, name='Strategy Rolling DD'))

    # Heatmap
    dff['Weekday'] = dff.index.day_name()
    heatmap_data = dff.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Market_returns'].values, x=heatmap_data.index, y=['Market'], colorscale='Blues'))
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Strategy_returns'].values, x=heatmap_data.index, y=['AI Strategy'], colorscale='Greens'))

    # Metrics panel
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(dff['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(dff['Cumulative_strategy']),3)}"),
        html.P(f"Strategy Exposure (mean): {round(dff['AI_signal'].mean(),3)}"),
        html.P(f"Number of Buy signals: {int((dff['AI_signal']==1).sum())}"),
        html.P(f"Number of Sell signals: {int((dff['AI_signal']==0).sum())}")
    ]

    return fig_cum, fig_dd, fig_rolling, fig_heatmap, metrics_text

# -----------------------------
# 9️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
