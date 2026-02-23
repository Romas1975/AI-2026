import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. Load Data
# =========================

ticker = "SPY"
df = yf.download(ticker, period="5y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["Return"] = df["Close"].pct_change()

# =========================
# 2. Feature Engineering
# =========================

# Lag returns
for i in range(1,6):
    df[f"lag_{i}"] = df["Return"].shift(i)

# Volatility
df["vol_20"] = df["Return"].rolling(20).std()

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# Moving average slope
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA_slope"] = df["MA20"] - df["MA50"]

# Target
df["Target"] = df["Return"].shift(-1)

df.dropna(inplace=True)

# =========================
# 3. Prepare Data
# =========================

features = ["lag_1","lag_2","lag_3","lag_4","lag_5",
            "vol_20","RSI","MA_slope"]

X = df[features].values
y = df["Target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# reshape for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# train/test split (chronological)
split = int(len(X)*0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 4. Build LSTM
# =========================

model = Sequential([
    Input(shape=(1, X.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 5. Predictions
# =========================

pred = model.predict(X_test).flatten()

df_test = df.iloc[split:].copy()
df_test["Predicted_Return"] = pred

df_test["Signal"] = np.where(df_test["Predicted_Return"] > 0, 1, 0)

df_test["Strategy_Return"] = df_test["Return"] * df_test["Signal"]

df_test["Cumulative_Market"] = (1 + df_test["Return"]).cumprod()
df_test["Cumulative_Strategy"] = (1 + df_test["Strategy_Return"]).cumprod()

# =========================
# 6. Metrics
# =========================

def sharpe(r):
    return np.sqrt(252)*r.mean()/r.std()

print("Market Sharpe:", round(sharpe(df_test["Return"]),3))
print("Strategy Sharpe:", round(sharpe(df_test["Strategy_Return"]),3))

print("Market Max DD:", round((df_test["Cumulative_Market"]/df_test["Cumulative_Market"].cummax()-1).min(),3))
print("Strategy Max DD:", round((df_test["Cumulative_Strategy"]/df_test["Cumulative_Strategy"].cummax()-1).min(),3))

# =========================
# 7. Dash Dashboard
# =========================

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("LSTM Return Prediction Strategy"),
    dcc.Graph(
        figure={
            "data":[
                go.Scatter(x=df_test.index, y=df_test["Cumulative_Market"], name="Market"),
                go.Scatter(x=df_test.index, y=df_test["Cumulative_Strategy"], name="Strategy")
            ],
            "layout":go.Layout(title="Cumulative Returns")
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)