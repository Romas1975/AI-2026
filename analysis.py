# ================================
# AI-2026 Professional Backtest Framework
# ================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load data
# ------------------------------
def get_data(ticker='SPY', start='2024-01-01', end=None):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    return df

# ------------------------------
# 2️⃣ Generate dummy AI signals
# ------------------------------
def generate_signals(df):
    # Dummy AI: long if yesterday return > 0, else flat
    df['AI_signal'] = (df['Returns'].shift(1) > 0).astype(int)
    return df

# ------------------------------
# 3️⃣ Strategy / Backtest
# ------------------------------
def apply_strategy(df):
    df['Strategy_returns'] = df['AI_signal'] * df['Returns']
    df['Cumulative_market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
    return df

# ------------------------------
# 4️⃣ Performance metrics
# ------------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    drawdown = cumulative / roll_max - 1
    return drawdown.min()

def analytics(df):
    metrics = {}
    metrics['Market Sharpe'] = round(sharpe_ratio(df['Returns'].dropna()), 3)
    metrics['Strategy Sharpe'] = round(sharpe_ratio(df['Strategy_returns'].dropna()), 3)
    metrics['Market Max Drawdown'] = round(max_drawdown(df['Cumulative_market']), 3)
    metrics['Strategy Max Drawdown'] = round(max_drawdown(df['Cumulative_strategy']), 3)
    metrics['AI_signal distribution'] = df['AI_signal'].value_counts()
    return metrics

# ------------------------------
# 5️⃣ Plot equity curves
# ------------------------------
def plot_equity(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Cumulative_market'], label='Buy & Hold', marker='o')
    plt.plot(df.index, df['Cumulative_strategy'], label='AI Strategy', marker='x')
    
    # Missed opportunities
    threshold = 0.005
    missed = (df['AI_signal'] == 0) & (df['Returns'].shift(-1) > threshold)
    plt.scatter(df.index[missed], df['Cumulative_market'][missed], color='red',
                label='Missed Opportunities', s=100, marker='^')
    
    plt.title('AI Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# 6️⃣ Main
# ------------------------------
if __name__ == "__main__":
    df = get_data()
    df = generate_signals(df)
    df = apply_strategy(df)
    
    metrics = analytics(df)
    print("\n=== PERFORMANCE METRICS ===")
    for k,v in metrics.items():
        print(k, ":", v)
    
    plot_equity(df)
