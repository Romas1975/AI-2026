import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------------------
# 1️⃣ Load data
# ------------------------------
def get_data(ticker='SPY', start='2020-01-01', end=None):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

# ------------------------------
# 2️⃣ Prepare LSTM data
# ------------------------------
def create_lstm_data(df, window=5):
    X, y = [], []
    for i in range(window, len(df)):
        X.append(df['Close'].values[i-window:i])
        y.append(1 if df['Close'].values[i] > df['Close'].values[i-1] else 0)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# ------------------------------
# 3️⃣ Build and train LSTM
# ------------------------------
def train_lstm(X_train, y_train, epochs=20, batch_size=8):
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# ------------------------------
# 4️⃣ Generate AI signals
# ------------------------------
def generate_signals(model, X):
    probs = model.predict(X)
    return (probs.flatten() > 0.5).astype(int)

# ------------------------------
# 5️⃣ Backtest strategy
# ------------------------------
def backtest(df, signals):
    df = df.iloc[-len(signals):].copy()
    df['AI_signal'] = signals
    df['Strategy_returns'] = df['Returns'] * df['AI_signal']
    df['Cumulative_market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
    return df

# ------------------------------
# 6️⃣ Performance metrics
# ------------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    drawdown = cumulative / roll_max - 1
    return drawdown.min()

def analytics(df):
    metrics = {}
    metrics['Market Sharpe'] = round(sharpe_ratio(df['Returns']), 3)
    metrics['Strategy Sharpe'] = round(sharpe_ratio(df['Strategy_returns']), 3)
    metrics['Market Max Drawdown'] = round(max_drawdown(df['Cumulative_market']), 3)
    metrics['Strategy Max Drawdown'] = round(max_drawdown(df['Cumulative_strategy']), 3)
    metrics['AI_signal distribution'] = df['AI_signal'].value_counts()
    return metrics

# ------------------------------
# 7️⃣ Plot equity
# ------------------------------
def plot_equity(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Cumulative_market'], label='Buy & Hold', marker='o')
    plt.plot(df.index, df['Cumulative_strategy'], label='AI Strategy', marker='x')
    plt.title('AI Strategy vs Buy & Hold (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------
# 8️⃣ Main
# ------------------------------
if __name__ == "__main__":
    df = get_data()
    
    # Skalavimas kainų į [0,1]
    scaler = MinMaxScaler()
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])
    
    # LSTM duomenų paruošimas
    window = 5
    X, y = create_lstm_data(df[['Close_scaled']], window)
    
    # Train-test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train LSTM
    model = train_lstm(X_train, y_train)
    
    # Predict AI signals
    signals = generate_signals(model, X_test)
    
    # Backtest
    df_bt = backtest(df, signals)
    
    # Metrics
    metrics = analytics(df_bt)
    print("\n=== PERFORMANCE METRICS ===")
    for k,v in metrics.items():
        print(k, ":", v)
    
    # Equity curve
    plot_equity(df_bt)
