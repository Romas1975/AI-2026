# analysis_lstm_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1️⃣ Load SPY data
# -----------------------------
ticker = "SPY"
df = yf.download(ticker, period="1y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty:
    raise ValueError("Yahoo Finance grąžino tuščią dataframe!")

df["Market_returns"] = df["Close"].pct_change()

# -----------------------------
# 2️⃣ Prepare LSTM data
# -----------------------------
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
window = 20
for i in range(window, len(scaled_data)-1):
    X.append(scaled_data[i-window:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -----------------------------
# 3️⃣ LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# -----------------------------
# 4️⃣ Generate AI signal
# -----------------------------
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

ai_signal = np.zeros(len(df))
ai_signal[window:len(predicted)+window] = np.where(predicted.flatten() > data[window:len(predicted)+window].flatten(), 1, 0)
df['AI_signal'] = ai_signal
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
df['Drawdown_strategy'] = df['Cumulative_strategy'] / df['Cumulative_strategy'].cummax() - 1
df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Drawdown_market'] = df['Cumulative_market'] / df['Cumulative_market'].cummax() - 1

# -----------------------------
# 5️⃣ Metrics functions
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

# -----------------------------
# 6️⃣ Dash app
# -----------------------------
app = dash.Dash(__name__)
app.title = "LSTM AI Strategy vs Buy & Hold"

app.layout = html.Div([
    html.H1("LSTM AI Strategy vs Buy & Hold", style={'textAlign':'center'}),
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df.index.min(),
            end_date=df.index.max(),
            min_date_allowed=df.index.min(),
            max_date_allowed=df.index.max()
        )
    ], style={'textAlign':'center', 'marginBottom':'20px'}),
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    dcc.Graph(id='heatmap_returns'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 7️⃣ Callback
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

    # --- Cumulative Returns ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # --- Drawdowns ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # --- Rolling metrics ---
    window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(window).mean() / dff['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(window).mean() / dff['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = dff['Cumulative_market'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = dff['Cumulative_strategy'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))
    fig_rolling.update_layout(title='Rolling Sharpe & Max Drawdown', xaxis_title='Date')

    # --- Heatmap: avg returns per weekday ---
    dff['Weekday'] = dff.index.day_name()
    heatmap_data = dff.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Market_returns'].values, x=heatmap_data.index, y=['Buy & Hold'], colorscale='Blues', showscale=True))
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Strategy_returns'].values, x=heatmap_data.index, y=['AI Strategy'], colorscale='Greens', showscale=True))
    fig_heatmap.update_layout(title='Average Daily Returns by Weekday', yaxis_title='Strategy')

    # --- Metrics panel ---
    metrics_text = [
        html.P(f"Total Market Return: {round((dff['Market_returns']+1).prod()-1,3)}"),
        html.P(f"Total Strategy Return: {round((dff['Strategy_returns']+1).prod()-1,3)}"),
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
# 8️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)