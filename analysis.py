import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# --- 1. Pavyzdiniai duomenys ---
data = {
    'Close': [669.42, 674.48, 680.59, 684.83, 687.96, 690.38, 690.31, 687.85, 687.01, 681.92],
    'AI_signal': [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    'Cumulative_market': [1.538, 1.550, 1.564, 1.573, 1.581, 1.586, 1.586, 1.580, 1.578, 1.567],
    'Cumulative_strategy': [1.216, 1.225, 1.225, 1.233, 1.239, 1.239, 1.239, 1.234, 1.233, 1.224]
}
dates = pd.date_range(start='2025-12-17', periods=10, freq='B')
spy_data = pd.DataFrame(data, index=dates)

# --- 2. Apskaičiuojame daily returns ---
spy_data['Market_returns'] = spy_data['Close'].pct_change()
spy_data['Strategy_returns'] = spy_data['Market_returns'] * spy_data['AI_signal']

# --- 3. Drawdowns ---
spy_data['Drawdown_market'] = spy_data['Cumulative_market'] / spy_data['Cumulative_market'].cummax() - 1
spy_data['Drawdown_strategy'] = spy_data['Cumulative_strategy'] / spy_data['Cumulative_strategy'].cummax() - 1

# --- 4. Sharpe ratio ---
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

market_sharpe = sharpe_ratio(spy_data['Market_returns'].dropna())
strategy_sharpe = sharpe_ratio(spy_data['Strategy_returns'].dropna())

# --- 5. Max drawdown ---
def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

market_dd = max_drawdown(spy_data['Cumulative_market'])
strategy_dd = max_drawdown(spy_data['Cumulative_strategy'])

# --- 6. Equity chart su drawdown ---
plt.figure(figsize=(14,6))
plt.plot(spy_data.index, spy_data['Cumulative_market'], label='Buy & Hold', marker='o')
plt.plot(spy_data.index, spy_data['Cumulative_strategy'], label='AI Strategy', marker='x')
plt.fill_between(spy_data.index, spy_data['Cumulative_market'], spy_data['Cumulative_market'].cummax(),
                 color='red', alpha=0.2, label='Market Drawdown')
plt.fill_between(spy_data.index, spy_data['Cumulative_strategy'], spy_data['Cumulative_strategy'].cummax(),
                 color='blue', alpha=0.2, label='Strategy Drawdown')
plt.title('Equity Curve & Drawdowns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7. Daily returns histogram ---
plt.figure(figsize=(12,4))
sns.histplot(spy_data['Market_returns'].dropna(), color='orange', label='Market', kde=True, stat='density', bins=10)
sns.histplot(spy_data['Strategy_returns'].dropna(), color='blue', label='AI Strategy', kde=True, stat='density', bins=10)
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Signal statistics ---
print("\n=== AI SIGNAL STATISTICS ===")
print(spy_data['AI_signal'].value_counts())
print("\nCorrelation of AI_signal with market returns:")
print(spy_data[['AI_signal', 'Market_returns']].corr())

# --- 9. Performance metrics ---
print("\n=== PERFORMANCE METRICS ===")
print(f"Market Sharpe : {market_sharpe:.3f}")
print(f"Strategy Sharpe : {strategy_sharpe:.3f}")
print(f"Market Max Drawdown : {market_dd:.3f}")
print(f"Strategy Max Drawdown : {strategy_dd:.3f}")
print(f"Total Return Market: {spy_data['Cumulative_market'].iloc[-1]-1:.3f}")
print(f"Total Return Strategy: {spy_data['Cumulative_strategy'].iloc[-1]-1:.3f}")
