import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- 1. Duomenys ---
ticker = "AAPL"
df = yf.download(ticker, period="2y", interval="1d")
df = df[['Close']].dropna()

# --- 2. MACD ---
short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
macd = short_ema - long_ema
signal = macd.ewm(span=9, adjust=False).mean()
df['MACD_signal'] = 0
df.loc[macd > signal, 'MACD_signal'] = 1
df.loc[macd < signal, 'MACD_signal'] = -1

# --- 3. LSTM ---
lookback = 10
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(lookback, len(scaled_close)):
    X.append(scaled_close[i-lookback:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# LSTM prediction
preds = model.predict(X, verbose=0)
preds_rescaled = scaler.inverse_transform(preds)

# Sutrumpiname df, kad ilgiai sutaptų
df_lstm = df.iloc[lookback:].copy()
df_lstm['LSTM_pred'] = preds_rescaled

# LSTM signalai
df_lstm['LSTM_signal'] = np.where(
    df_lstm['LSTM_pred'].shift(1) < df_lstm['Close'], 1,
    np.where(df_lstm['LSTM_pred'].shift(1) > df_lstm['Close'], -1, 0)
)

# --- 4. Dash ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1(f"{ticker} MACD + LSTM Signals"),
    dcc.Graph(id='price-graph'),
])

@app.callback(
    Output('price-graph', 'figure'),
    Input('price-graph', 'id')
)
def update_graph(_):
    fig = {
        'data': [
            {'x': df_lstm.index, 'y': df_lstm['Close'], 'type': 'line', 'name': 'Close'},
            {'x': df_lstm.index, 'y': df_lstm['MACD_signal']*df_lstm['Close'].max(), 
             'type': 'bar', 'name': 'MACD Signal'},
            {'x': df_lstm.index, 'y': df_lstm['LSTM_pred'], 'type': 'line', 'name': 'LSTM Pred'},
            {'x': df_lstm.index, 'y': df_lstm['LSTM_signal']*df_lstm['Close'].max(), 
             'type': 'bar', 'name': 'LSTM Signal'},
        ],
        'layout': {
            'title': f'{ticker} Price + Signals'
        }
    }
    return fig

# Nauja dash.run() sintaksė
if __name__ == "__main__":
    app.run(debug=True)
