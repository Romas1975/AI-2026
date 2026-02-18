import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

data = yf.download("SPY", start="2015-01-01")

# MACD
data["EMA12"] = data["Close"].ewm(span=12).mean()
data["EMA26"] = data["Close"].ewm(span=26).mean()
data["MACD"] = data["EMA12"] - data["EMA26"]
data["Signal"] = data["MACD"].ewm(span=9).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

data.dropna(inplace=True)

features = data[["MACD", "Signal", "RSI"]]
target = data["Target"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,
    target,
    test_size=0.2,
    shuffle=False
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, batch_size=32)


