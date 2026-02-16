import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download("SPY", start="2015-01-01")

print(data.head())

plt.figure()
plt.plot(data["Close"])
plt.title("SPY Closing Price")
plt.show()

