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

Open
High
Low
Close
Volume

