import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dash import Dash, dcc, html
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ Parsisiunčiame duomenis
df = yf.download("AAPL", period="2y", interval="1d")

# Flatten MultiIndex, jei reikia
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if col[0] else col[1] for col in df.columns]

if 'Close' not in df.columns:
    raise ValueError("Stulpelio 'Close' nėra duomenyse!")

close = df['Close'].values.reshape(-1,1)

# 2️⃣ MACD signalas
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()

df['MACD_signal'] = 0
df.loc[macd > signal, 'MACD_signal'] = 1
df.loc[macd < signal, 'MACD_signal'] = -1

# 3️⃣ LSTM paruošimas
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close)

X, y = [], []
seq_len = 10
for i in range(seq_len, len(close_scaled)):
    X.append(close_scaled[i-seq_len:i,0])
    y.append(close_scaled[i,0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4️⃣ LSTM modelis
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1],1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# 5️⃣ LSTM prognozės
lstm_pred = model.predict(X)
lstm_pred_full = np.concatenate([np.full((seq_len,1), np.nan), lstm_pred])
df['LSTM_pred'] = lstm_pred_full

# 6️⃣ LSTM signalai
df['LSTM_signal'] = 0
df['LSTM_signal'] = np.where(df['LSTM_pred'] < df['Close'], 1,
                              np.where(df['LSTM_pred'] > df['Close'], -1, 0))

# 7️⃣ Dash vizualizacija su pirkimo/pardavimo ženklais
app = Dash(__name__)

# Buy/Sell markers
buy_idx = df.index[df['LSTM_signal'] == 1]
sell_idx = df.index[df['LSTM_signal'] == -1]

app.layout = html.Div([
    html.H1("AAPL LSTM + MACD Trading Dashboard"),
    
    dcc.Graph(
        id='price-chart',
        figure={
            'data': [
                go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'),
                go.Scatter(x=df.index, y=df['LSTM_pred'], mode='lines', name='LSTM_pred'),
                go.Scatter(x=buy_idx, y=df.loc[buy_idx, 'Close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=12)),
                go.Scatter(x=sell_idx, y=df.loc[sell_idx, 'Close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=12)),
            ],
            'layout': go.Layout(title='AAPL Close vs LSTM Pred + Signals', xaxis={'title':'Date'}, yaxis={'title':'Price'})
        }
    ),

    dcc.Graph(
        id='macd-chart',
        figure={
            'data': [
                go.Scatter(x=df.index, y=macd, mode='lines', name='MACD'),
                go.Scatter(x=df.index, y=signal, mode='lines', name='Signal'),
            ],
            'layout': go.Layout(title='MACD Indicator', xaxis={'title':'Date'}, yaxis={'title':'MACD'})
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)