import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# LSTM imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. DATA FETCH
# =========================
symbol = "AAPL"
start_date = "2022-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
df = yf.download(symbol, start=start_date, end=end_date)
df.reset_index(inplace=True)

# =========================
# 2. MACD CALCULATION
# =========================
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_signal"] = np.where(df["MACD"] > df["Signal_line"], 1, 0)

# =========================
# 3. LSTM PREPARATION
# =========================
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1,1))

X, y = [], []
look_back = 10
for i in range(look_back, len(scaled_close)):
    X.append(scaled_close[i-look_back:i, 0])
    y.append(scaled_close[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM MODEL
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# LSTM PREDICTION
lstm_pred = model.predict(X)
df = df.iloc[look_back:]
df["LSTM_pred"] = scaler.inverse_transform(lstm_pred)

# =========================
# 4. DASHBOARD SETUP
# =========================
app = Dash(__name__)

app.layout = html.Div([
    html.H1(f"{symbol} Dashboard: MACD + LSTM"),
    
    html.Div([
        html.Label("Select feature:"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[
                {'label': 'Close', 'value': 'Close'},
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'Signal_line', 'value': 'Signal_line'},
                {'label': 'LSTM Prediction', 'value': 'LSTM_pred'}
            ],
            value='Close'
        )
    ], style={'width':'50%'}),
    
    dcc.Graph(id='feature-graph'),
    
    html.Div(id='stats-output', style={'marginTop':20})
])

# =========================
# 5. CALLBACKS
# =========================
@app.callback(
    Output('feature-graph', 'figure'),
    Output('stats-output', 'children'),
    Input('feature-dropdown', 'value')
)
def update_graph(feature):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close'
    ))
    
    if feature != 'Close':
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[feature],
            mode='lines',
            name=feature
        ))
    
    # STATISTICS
    returns = df['Close'].pct_change().dropna()
    strategy_returns = df['MACD_signal'].shift(1) * returns
    total_market = round((1 + returns).prod() - 1, 3)
    total_strategy = round((1 + strategy_returns).prod() - 1, 3)
    market_sharpe = round(returns.mean()/returns.std()*np.sqrt(252), 3)
    strategy_sharpe = round(strategy_returns.mean()/strategy_returns.std()*np.sqrt(252), 3)
    max_dd_market = round((returns.cumsum() - returns.cumsum().cummax()).min(), 3)
    max_dd_strategy = round((strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min(), 3)
    exposure = round(df['MACD_signal'].mean(), 3)
    buy_signals = int(df['MACD_signal'].sum())
    sell_signals = int((df['MACD_signal'] == 0).sum())
    
    stats_text = f"""
    Total Market Return: {total_market}
    Total Strategy Return: {total_strategy}
    
    Market Sharpe: {market_sharpe}
    Strategy Sharpe: {strategy_sharpe}
    
    Market Max Drawdown: {max_dd_market}
    Strategy Max Drawdown: {max_dd_strategy}
    
    Strategy Exposure (mean): {exposure}
    Number of Buy signals: {buy_signals}
    Number of Sell signals: {sell_signals}
    """
    
    return fig, stats_text

# =========================
# 6. RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)