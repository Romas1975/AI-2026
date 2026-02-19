import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Dummy duomenys (vietoje tavo spy_data) ---
data = {
    'Close': [669.42, 674.48, 680.59, 684.83, 687.96, 690.38, 690.31, 687.85, 687.01, 681.92],
    'AI_signal': [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    'Cumulative_market': [1.538, 1.550, 1.564, 1.573, 1.581, 1.586, 1.586, 1.580, 1.578, 1.567],
    'Cumulative_strategy': [1.216, 1.225, 1.225, 1.233, 1.239, 1.239, 1.239, 1.234, 1.233, 1.224]
}
dates = pd.date_range(start='2025-12-17', periods=10, freq='B')
spy_data = pd.DataFrame(data, index=dates)

# --- 2. Daily Returns ---
spy_data['Market_returns'] = spy_data['Close'].pct_change()
spy_data['Strategy_returns'] = spy_data['Market_returns'] * spy_data['AI_signal']

# --- 3. Drawdowns ---
spy_data['Drawdown_market'] = spy_data['Cumulative_market'] / spy_data['Cumulative_market'].cummax() - 1
spy_data['Drawdown_strategy'] = spy_data['Cumulative_strategy'] / spy_data['Cumulative_strategy'].cummax() - 1

# --- 4. Performance Metrics ---
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

metrics = {
    'Market Sharpe': sharpe_ratio(spy_data['Market_returns'].dropna()),
    'Strategy Sharpe': sharpe_ratio(spy_data['Strategy_returns'].dropna()),
    'Market Max Drawdown': max_drawdown(spy_data['Cumulative_market']),
    'Strategy Max Drawdown': max_drawdown(spy_data['Cumulative_strategy']),
    'AI_signal 1 count': spy_data['AI_signal'].sum(),
    'AI_signal 0 count': len(spy_data) - spy_data['AI_signal'].sum()
}

print("=== PERFORMANCE METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {round(v,3)}")

# --- 5. Plotas Dashboard ---
plt.figure(figsize=(14,8))

# a) Cumulative Returns
plt.subplot(2,2,1)
plt.plot(spy_data.index, spy_data['Cumulative_market'], label='Buy & Hold', marker='o')
plt.plot(spy_data.index, spy_data['Cumulative_strategy'], label='AI Strategy', marker='x')
plt.title('Cumulative Returns')
plt.legend()
plt.grid(True)

# b) Daily Returns
plt.subplot(2,2,2)
plt.plot(spy_data.index, spy_data['Market_returns'], label='Market Returns', marker='o')
plt.plot(spy_data.index, spy_data['Strategy_returns'], label='Strategy Returns', marker='x')
plt.title('Daily Returns')
plt.legend()
plt.grid(True)

# c) Drawdowns
plt.subplot(2,2,3)
plt.plot(spy_data.index, spy_data['Drawdown_market'], label='Market Drawdown', marker='o')
plt.plot(spy_data.index, spy_data['Drawdown_strategy'], label='Strategy Drawdown', marker='x')
plt.title('Drawdowns')
plt.legend()
plt.grid(True)

# d) Missed Opportunities
plt.subplot(2,2,4)
threshold = 0.005
missed = (spy_data['AI_signal']==0) & (spy_data['Market_returns'] > threshold)
plt.scatter(spy_data.index[missed], spy_data['Cumulative_market'][missed], color='red', s=100, label='Missed Opportunities', marker='^')
plt.plot(spy_data.index, spy_data['Cumulative_market'], label='Buy & Hold', marker='o')
plt.plot(spy_data.index, spy_data['Cumulative_strategy'], label='AI Strategy', marker='x')
plt.title('Missed Opportunities')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
