import pandas as pd
import matplotlib.pyplot as plt

url = "https://stooq.com/q/d/l/?s=spy.us&i=d"

data = pd.read_csv(url)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")
data.set_index("Date", inplace=True)

print(data.tail())

plt.figure()
plt.plot(data["Close"])
plt.title("SPY Closing Price (Stooq)")
plt.show()



import numpy as np

# EMA
data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()

# MACD
data["MACD"] = data["EMA12"] - data["EMA26"]
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["Histogram"] = data["MACD"] - data["Signal"]

plt.figure()
plt.plot(data["MACD"], label="MACD")
plt.plot(data["Signal"], label="Signal")
plt.bar(data.index, data["Histogram"])
plt.legend()
plt.title("MACD - SPY (Stooq)")
plt.show()

data["Buy"] = np.where(
    (data["MACD"] > data["Signal"]) &
    (data["MACD"].shift(1) <= data["Signal"].shift(1)),
    1, 0
)

data["Sell"] = np.where(
    (data["MACD"] < data["Signal"]) &
    (data["MACD"].shift(1) >= data["Signal"].shift(1)),
    1, 0
)

print("BUY signals:")
print(data[data["Buy"] == 1].tail())

print("SELL signals:")
print(data[data["Sell"] == 1].tail())

# Dienos grąža
data["Return"] = data["Close"].pct_change()

data["Position"] = 0

data.loc[data["Buy"] == 1, "Position"] = 1
data.loc[data["Sell"] == 1, "Position"] = 0

# Forward fill kad laikytume poziciją iki SELL
data["Position"] = data["Position"].replace(to_replace=0, method="ffill")
data["Position"] = data["Position"].fillna(0)




