import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import dash
from dash import dcc, html
import plotly.graph_objs as go

# --- 1. Duomenų parsisiuntimas ---
ticker = "AAPL"
df = yf.download(ticker, period="2y", interval="1d")

# Reset indeksą, kad nebūtų tuple problemos
df = df.reset_index()
df = df[['Date', 'Close']]

# --- 2. MACD signalai ---
short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
macd = short_ema - long_ema
signal = macd.ewm(span=9, adjust=False).mean()

df['MACD_signal'] = 0
df.loc[macd > signal, 'MACD_signal'] = 1
df.loc[macd < signal, 'MACD_signal'] = -1

# --- 3. LSTM modelis ---
# Normalizavimas
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Sukuriam X ir y
look_back = 10
X, y = [], []
for i in range(look_back, len(scaled_close)):
    X.append(scaled_close[i-look_back:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM modelis
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treniruojam LSTM
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# LSTM prognozės
lstm_pred = model.predict(X, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred)

# --- 4. LSTM signalai ---
# Suderinam ilgį
df_lstm = df.iloc[look_back:].copy()
df_lstm['LSTM_pred'] = lstm_pred
df_lstm['LSTM_signal'] = 0

close_values = df_lstm['Close'].values
lstm_pred_values = df_lstm['LSTM_pred'].values

df_lstm['LSTM_signal'] = np.where(lstm_pred_values > close_values, 1,
                                   np.where(lstm_pred_values < close_values, -1, 0))

# --- 5. Dash grafikas ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"{ticker} MACD + LSTM Signals"),
    dcc.Graph(
        id='price-chart',
        figure={
            'data': [
                go.Scatter(x=df_lstm['Date'], y=df_lstm['Close'], mode='lines', name='Close'),
                go.Scatter(x=df_lstm['Date'], y=df_lstm['LSTM_pred'], mode='lines', name='LSTM Pred'),
                go.Scatter(x=df_lstm[df_lstm['MACD_signal']==1]['Date'],
                           y=df_lstm[df_lstm['MACD_signal']==1]['Close'],
                           mode='markers', marker=dict(color='green', size=8), name='MACD Buy'),
                go.Scatter(x=df_lstm[df_lstm['MACD_signal']==-1]['Date'],
                           y=df_lstm[df_lstm['MACD_signal']==-1]['Close'],
                           mode='markers', marker=dict(color='red', size=8), name='MACD Sell')
            ],
            'layout': go.Layout(title=f"{ticker} Price & Signals", xaxis_title="Date", yaxis_title="Price")
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)