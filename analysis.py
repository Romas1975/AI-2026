import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download("SPY", start="2015-01-01")

print(data.head())

plt.figure()
plt.plot(data["Close"])
plt.title("SPY Closing Price")
plt.show()
# 50 dienų slenkantis vidurkis
data["MA50"] = data["Close"].rolling(window=50).mean()

# 200 dienų slenkantis vidurkis
data["MA200"] = data["Close"].rolling(window=200).mean()

plt.figure()
plt.plot(data["Close"])
plt.plot(data["MA50"])
plt.plot(data["MA200"])
plt.title("SPY su MA50 ir MA200")
plt.show()
import yfinance as yf

data = yf.download("SPY", start="2015-01-01")

print("Data shape:", data.shape)
print(data.tail())
import yfinance as yf

data = yf.download("SPY", start="2015-01-01", auto_adjust=True)

print("Data shape:", data.shape)
print(data.tail())
import yfinance as yf

data = yf.download("SPY", period="10y", auto_adjust=True)

data.to_csv("spy_data.csv")

print("Saved locally.")
import pandas as pd

data = pd.read_csv("spy_data.csv", index_col=0, parse_dates=True)
data = yf.download("SPY", period="5y", auto_adjust=True)
print(data.tail())


