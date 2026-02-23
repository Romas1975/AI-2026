import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- 1. Duomenų parsisiuntimas ---
symbol = "AAPL"
df = yf.download(symbol, start="2022-01-01", end="2026-01-01")

# Išlyginame MultiIndex į paprastą indeksą (datą)
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index(level=df.index.names.difference(['Date']), drop=True)

df = df[['Close']].copy()  # Naudojame tik uždarymo kainą

# --- 2. MACD skaičiavimas ---
def calculate_macd(df):
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()

    # Užtikriname, kad `macd` ir `signal` yra vienmatės serijos
    macd = pd.Series(macd.values.squeeze(), index=df.index)
    signal = pd.Series(signal.values.squeeze(), index=df.index)

    return macd, signal

macd, signal = calculate_macd(df)

# --- 3. Signalų priskyrimas ---
df["MACD_signal"] = 0
df.loc[macd > signal, "MACD_signal"] = 1
df.loc[macd < signal, "MACD_signal"] = -1

# --- 4. LSTM paruošimas ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]].values)

def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(scaled_data, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --- 5. LSTM modelio apmokymas ---
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# --- 6. LSTM prognozės ---
predicted = model.predict(X)
df = df.iloc[look_back:].copy()  # Iškartojame `df` pagal `predicted` ilgį
df["LSTM_pred"] = scaler.inverse_transform(predicted).flatten()  # Naudojame `.flatten()`

# --- 7. LSTM signalų generavimas ---
df["LSTM_signal"] = 0
df.loc[df["LSTM_pred"].shift(1) < df["Close"], "LSTM_signal"] = 1
df.loc[df["LSTM_pred"].shift(1) > df["Close"], "LSTM_signal"] = -1

print(macd.shape)
print(signal.shape)
print(df.index)
