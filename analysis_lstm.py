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

# Ensure single-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Calculate Market returns
df['Market_returns'] = df['Close'].pct_change()
df.dropna(inplace=True)

# -----------------------------
# 2️⃣ Prepare LSTM data
# -----------------------------
window = 20  # past 20 days
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(window, len(df_scaled)-1):
    X.append(df_scaled[i-window:i, 0])
    y.append(1 if df_scaled[i+1, 0] > df_scaled[i, 0] else 0)  # binary up/down

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# Train/Test split
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 3️⃣ LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(20, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Predict
lstm_pred = model.predict(X)
lstm_signal = (lstm_pred.flatten() > 0.5).astype(int)

# Align with original dataframe
df = df.iloc[window+1:].copy()
df['AI_signal'] = lstm_signal

# -----------------------------
# 4️⃣ Strategy returns
# -----------------------------
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
df['Drawdown_market'] = df['Cumulative_market']/df['Cumulative_market'].cummax()-1
df['Drawdown_strategy'] = df['Cumulative_strategy']/df['Cumulative_strategy'].cummax()-1

# Metrics
def sharpe_ratio(returns):
    return np.sqrt(252)*returns.mean()/returns.std()

def max_drawdown(cumulative):
    return (cumulative/cumulative.cummax()-1).min()

# -----------------------------
# 5️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "AI Strategy vs Buy & Hold (LSTM)"

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df.index.min(),
        end_date=df.index.max(),
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max()
    ),
    dcc.Dropdown(id='strategy-dropdown',
                 options=[{'label':'LSTM','value':'LSTM'}, {'label':'MACD','value':'MACD'}],
                 value='LSTM', clearable=False),
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 6️⃣ Callback
# -----------------------------
@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('metrics','children'),
    Input('date-range','start_date'),
    Input('date-range','end_date'),
    Input('strategy-dropdown','value')
)
def update_dashboard(start_date, end_date, strategy_choice):
    dff = df.loc[start_date:end_date].copy()

    # If MACD is selected
    if strategy_choice=='MACD':
        ema12 = dff['Close'].ewm(span=12).mean()
        ema26 = dff['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        dff['AI_signal'] = (macd > signal_line).astype(int)
        dff['Strategy_returns'] = dff['Market_returns']*dff['AI_signal']
        dff['Cumulative_strategy'] = (1 + dff['Strategy_returns']).cumprod()
        dff['Drawdown_strategy'] = dff['Cumulative_strategy']/dff['Cumulative_strategy'].cummax()-1

    # --- Cumulative Returns
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Buy & Hold'))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='AI Strategy'))

    # --- Drawdowns
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))

    # --- Rolling metrics
    window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(window).mean()/dff['Market_returns'].rolling(window).std()*np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(window).mean()/dff['Strategy_returns'].rolling(window).std()*np.sqrt(252)

    rolling_dd_market = dff['Cumulative_market'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = dff['Cumulative_strategy'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))

    # --- Metrics panel
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns']),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(dff['Strategy_returns']),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(dff['Cumulative_strategy']),3)}"),
        html.P(f"Strategy Exposure (mean): {round(dff['AI_signal'].mean(),3)}"),
        html.P(f"Number of Buy signals: {int((dff['AI_signal']==1).sum())}"),
        html.P(f"Number of Sell signals: {int((dff['AI_signal']==0).sum())}")
    ]

    return fig_cum, fig_dd, fig_rolling, metrics_text

# -----------------------------
# 7️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
