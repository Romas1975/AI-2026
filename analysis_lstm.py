import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# =========================
# 1️⃣ LOAD DATA
# =========================

ticker = "SPY"
df = yf.download(ticker, period="3y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["Market_returns"] = df["Close"].pct_change()

df.dropna(inplace=True)

# =========================
# 2️⃣ LSTM MODEL
# =========================

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[["Close"]])

sequence_length = 30

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential([
    Input(shape=(X.shape[1],1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

predictions = model.predict(X, verbose=0)
predictions = scaler.inverse_transform(predictions)

df = df.iloc[sequence_length:].copy()
df["Predicted_Close"] = predictions.flatten()

# =========================
# 3️⃣ SIGNAL GENERATION
# =========================

df["AI_signal"] = np.where(df["Predicted_Close"] > df["Close"], 1, 0)

# =========================
# 4️⃣ RISK ENGINE
# =========================

target_vol = 0.15
rolling_window = 20
max_leverage = 2.0

df["Rolling_vol"] = df["Market_returns"].rolling(rolling_window).std() * np.sqrt(252)
df["Vol_position"] = target_vol / df["Rolling_vol"]
df["Vol_position"] = df["Vol_position"].clip(upper=max_leverage)

df["Final_position"] = df["AI_signal"] * df["Vol_position"]

df["Strategy_returns"] = df["Final_position"] * df["Market_returns"]

df.dropna(inplace=True)

# =========================
# 5️⃣ PERFORMANCE
# =========================

df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()

df["Drawdown_market"] = df["Cumulative_market"] / df["Cumulative_market"].cummax() - 1
df["Drawdown_strategy"] = df["Cumulative_strategy"] / df["Cumulative_strategy"].cummax() - 1

def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    return (cumulative / roll_max - 1).min()

print("Total Market Return:", round(df["Cumulative_market"].iloc[-1]-1,3))
print("Total Strategy Return:", round(df["Cumulative_strategy"].iloc[-1]-1,3))
print()
print("Market Sharpe:", round(sharpe_ratio(df["Market_returns"]),3))
print("Strategy Sharpe:", round(sharpe_ratio(df["Strategy_returns"]),3))
print()
print("Market Max Drawdown:", round(max_drawdown(df["Cumulative_market"]),3))
print("Strategy Max Drawdown:", round(max_drawdown(df["Cumulative_strategy"]),3))
print()
print("Strategy Exposure (mean):", round(df["Final_position"].mean(),3))
print("Max leverage used:", round(df["Final_position"].max(),3))
print("Number of Buy signals:", int(df["AI_signal"].sum()))
print("Number of Sell signals:", int((df["AI_signal"]==0).sum()))

# =========================
# 6️⃣ DASHBOARD
# =========================

app = dash.Dash(__name__)
app.title = "LSTM AI Strategy"

app.layout = html.Div([
    html.H1("LSTM AI Strategy with Risk Engine", style={'textAlign':'center'}),
    dcc.Graph(
        figure={
            "data":[
                go.Scatter(x=df.index, y=df["Cumulative_market"], name="Buy & Hold"),
                go.Scatter(x=df.index, y=df["Cumulative_strategy"], name="AI Strategy")
            ],
            "layout":go.Layout(title="Cumulative Returns")
        }
    ),
    dcc.Graph(
        figure={
            "data":[
                go.Scatter(x=df.index, y=df["Drawdown_market"], name="Market DD"),
                go.Scatter(x=df.index, y=df["Drawdown_strategy"], name="Strategy DD")
            ],
            "layout":go.Layout(title="Drawdowns")
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)