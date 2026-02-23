import numpy as np
import pandas as pd
import yfinance as yf

# ==============================
# 1️⃣ Load 5 years data
# ==============================
ticker = "SPY"
df = yf.download(ticker, period="5y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["Market_returns"] = df["Close"].pct_change()

# ==============================
# 2️⃣ Walk-forward parameters
# ==============================
train_size = 252      # 1 year
test_size = 63        # 3 months

start = 0
equity_curve = []
all_returns = []

# ==============================
# 3️⃣ Walk-forward loop
# ==============================
while start + train_size + test_size < len(df):

    train = df.iloc[start:start+train_size].copy()
    test = df.iloc[start+train_size:start+train_size+test_size].copy()

    # ---- MACD computed ONLY on train
    ema12 = train["Close"].ewm(span=12).mean()
    ema26 = train["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    last_macd = macd.iloc[-1]
    last_signal = signal.iloc[-1]

    # ---- Apply fixed rule on test
    test["Signal"] = np.where(last_macd > last_signal, 1, 0)

    test["Strategy_returns"] = test["Market_returns"] * test["Signal"]

    all_returns.append(test["Strategy_returns"])

    start += test_size

# ==============================
# 4️⃣ Combine results
# ==============================
strategy_returns = pd.concat(all_returns).dropna()
market_returns = df.loc[strategy_returns.index, "Market_returns"]

cum_market = (1 + market_returns).cumprod()
cum_strategy = (1 + strategy_returns).cumprod()

# ==============================
# 5️⃣ Metrics
# ==============================
def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_dd(cum):
    roll_max = cum.cummax()
    return (cum / roll_max - 1).min()

print("\n===== WALK-FORWARD RESULTS =====\n")
print("Market Sharpe:", round(sharpe(market_returns),3))
print("Strategy Sharpe:", round(sharpe(strategy_returns),3))
print("Market Max DD:", round(max_dd(cum_market),3))
print("Strategy Max DD:", round(max_dd(cum_strategy),3))
print("Strategy Exposure:", round(strategy_returns.ne(0).mean(),3))