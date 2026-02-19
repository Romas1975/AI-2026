import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Sukuriame dummy duomenis ---
dates = pd.date_range(start='2025-01-01', periods=500, freq='B')  # 500 darbo dienų
np.random.seed(42)
prices = np.cumsum(np.random.randn(500)*2 + 0.5) + 100  # pseudo SPY kaina
df = pd.DataFrame({'Close': prices}, index=dates)

# --- 2. Scale data ---
scaler = MinMaxScaler()
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# --- 3. LSTM data preparation ---
def create_lstm_data(df, col='Close_scaled', window=10):
    X, y = [], []
    for i in range(window, len(df)):
        X.append(df[col].values[i-window:i])
        y.append(1 if df[col].values[i] > df[col].values[i-1] else 0)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

window = 10
X, y = create_lstm_data(df, col='Close_scaled', window=window)

# --- 4. Dummy LSTM model (jeigu nera issaugoto) ---
model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=3, batch_size=16, verbose=0)  # greitas treniravimas demo

# --- 5. Predict AI signals ---
probs = model.predict(X)
ai_signal = (probs.flatten() > 0.5).astype(int)

# --- 6. Backtest ---
df_test = df.iloc[window:].copy()
df_test['AI_signal'] = ai_signal
df_test['Market_returns'] = df_test['Close'].pct_change()
df_test['Strategy_returns'] = df_test['Market_returns'] * df_test['AI_signal']
df_test['Cumulative_market'] = (1 + df_test['Market_returns']).cumprod()
df_test['Cumulative_strategy'] = (1 + df_test['Strategy_returns']).cumprod()

# --- 7. Performance metrics ---
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

market_sharpe = sharpe_ratio(df_test['Market_returns'].dropna())
strategy_sharpe = sharpe_ratio(df_test['Strategy_returns'].dropna())
market_dd = max_drawdown(df_test['Cumulative_market'])
strategy_dd = max_drawdown(df_test['Cumulative_strategy'])

print("\n=== PERFORMANCE METRICS ===")
print("Market Sharpe :", round(market_sharpe, 3))
print("Strategy Sharpe :", round(strategy_sharpe, 3))
print("Market Max Drawdown :", round(market_dd, 3))
print("Strategy Max Drawdown :", round(strategy_dd, 3))

print("\nAI_signal distribution:")
print(df_test['AI_signal'].value_counts())

print("\nCorrelation with market returns:")
print(df_test[['AI_signal', 'Market_returns']].corr())

# --- 8. Plot equity ---
plt.figure(figsize=(12,6))
plt.plot(df_test.index, df_test['Cumulative_market'], label='Buy & Hold', marker='o')
plt.plot(df_test.index, df_test['Cumulative_strategy'], label='AI Strategy', marker='x')

# missed opportunities: AI signal 0, market went up >0.5%
threshold = 0.005
missed = (df_test['AI_signal'] == 0) & (df_test['Market_returns'].shift(-1) > threshold)
plt.scatter(df_test.index[missed], df_test['Cumulative_market'][missed],
            color='red', label='Missed Opportunities', s=80, marker='^')

plt.title('AI Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
