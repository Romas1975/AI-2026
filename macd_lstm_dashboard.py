import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html

# ======================
# 1. Duomenų parsisiuntimas
# ======================
ticker = "AAPL"
df = yf.download(ticker, period="2y", interval="1d")

# Flatten MultiIndex stulpelius, jei jie yra
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[1] if col[1] else col[0] for col in df.columns]

# Tik Close naudoti LSTM
close = df['Close'].values.reshape(-1, 1)

# ======================
# 2. MACD signalai
# ======================
short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
macd = short_ema - long_ema
signal = macd.ewm(span=9, adjust=False).mean()

df['MACD_signal'] = 0
df.loc[macd > signal, 'MACD_signal'] = 1
df.loc[macd < signal, 'MACD_signal'] = -1

# ======================
# 3. LSTM modelis
# ======================
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close)

X, y = [], []
look_back = 10

for i in range(look_back, len(scaled_close)):
    X.append(scaled_close[i-look_back:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# LSTM predikcijos
lstm_pred = model.predict(X)
lstm_pred_values = scaler.inverse_transform(lstm_pred)

# ======================
# 4. LSTM signalai
# ======================
df_lstm = df.copy().iloc[look_back:]
df_lstm['LSTM_pred'] = lstm_pred_values
df_lstm['LSTM_signal'] = 0

close_values = df_lstm['Close'].values
df_lstm['LSTM_signal'] = np.where(lstm_pred_values.flatten() < close_values, 1,
                                  np.where(lstm_pred_values.flatten() > close_values, -1, 0))

# ======================
# 5. Dash app
# ======================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"{ticker} MACD + LSTM Signals"),
    dcc.Graph(
        figure={
            'data': [
                {'x': df_lstm.index, 'y': df_lstm['Close'], 'type': 'line', 'name': 'Close'},
                {'x': df_lstm.index, 'y': df_lstm['LSTM_pred'], 'type': 'line', 'name': 'LSTM Pred'},
                {'x': df_lstm.index, 'y': df_lstm['MACD_signal']*df_lstm['Close'].max(), 
                 'type': 'scatter', 'mode': 'markers', 'name': 'MACD Signal'}
            ],
            'layout': {'title': f'{ticker} Close, LSTM & MACD Signals'}
        }
    )
])

# Dash >=2.9.0 naudoja .run(), ne .run_server()
if __name__ == "__main__":
    app.run(debug=True)