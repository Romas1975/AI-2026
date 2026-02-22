import numpy as np
import pandas as pd
import yfinance as yf

# ==============================
# 1️⃣ LOAD DATA
# ==============================

ticker = "SPY"

df = yf.download(ticker, period="1y", auto_adjust=True)

# Fix MultiIndex (Yahoo sometimes returns it)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty:
    raise ValueError("No data returned from Yahoo Finance")

# ==============================
# 2️⃣ BASIC RETURNS
# ==============================

df["Market_returns"] = df["Close"].pct_change()

# Drop first NaN
df = df.dropna().copy()

# ==============================
# 3️⃣ VOLATILITY TARGETING STRATEGY
# ==============================

# 20-day rolling annualized volatility
df["Vol_20"] = df["Market_returns"].rolling(20).std() * np.sqrt(252)

# Fill initial NaN properly
df["Vol_20"] = df["Vol_20"].bfill()

# Target annual volatility
target_vol = 0.15

# Position sizing
df["AI_signal"] = target_vol / df["Vol_20"]

# Clip exposure
df["AI_signal"] = df["AI_signal"].clip(0.2, 1.5)

# Strategy returns
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"]

# ==============================
# 4️⃣ PERFORMANCE
# ==============================

df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()

def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    return (cumulative / roll_max - 1).min()

market_return = df["Cumulative_market"].iloc[-1] - 1
strategy_return = df["Cumulative_strategy"].iloc[-1] - 1

market_sharpe = sharpe_ratio(df["Market_returns"])
strategy_sharpe = sharpe_ratio(df["Strategy_returns"])

market_dd = max_drawdown(df["Cumulative_market"])
strategy_dd = max_drawdown(df["Cumulative_strategy"])

exposure = df["AI_signal"].mean()

# ==============================
# 5️⃣ OUTPUT
# ==============================

print("\n===== PROFESSIONAL STRATEGY REPORT =====\n")

print(f"Total Market Return: {round(market_return,3)}")
print(f"Total Strategy Return: {round(strategy_return,3)}")

print(f"\nMarket Sharpe: {round(market_sharpe,3)}")
print(f"Strategy Sharpe: {round(strategy_sharpe,3)}")

print(f"\nMarket Max Drawdown: {round(market_dd,3)}")
print(f"Strategy Max Drawdown: {round(strategy_dd,3)}")

print(f"\nStrategy Exposure: {round(exposure,3)}")