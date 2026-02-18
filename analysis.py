import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Sukuriame spy_data (arba įkelk iš CSV) ---
data = {
    'Close': [669.42, 674.48, 680.59, 684.83, 687.96, 690.38, 690.31, 687.85, 687.01, 681.92],
    'AI_signal': [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    'Cumulative_market': [1.538, 1.550, 1.564, 1.573, 1.581, 1.586, 1.586, 1.580, 1.578, 1.567],
    'Cumulative_strategy': [1.216, 1.225, 1.225, 1.233, 1.239, 1.239, 1.239, 1.234, 1.233, 1.224]
}
dates = pd.date_range(start='2025-12-17', periods=10, freq='B')  # darbo dienos
spy_data = pd.DataFrame(data, index=dates)

# --- 2. Atvaizduojame grafikus ---
plt.figure(figsize=(12,6))
plt.plot(spy_data.index, spy_data['Cumulative_market'], label='Buy & Hold', marker='o')
plt.plot(spy_data.index, spy_data['Cumulative_strategy'], label='AI Strategy', marker='x')

# --- 3. Pažymime missed opportunities ---
# jei AI signalas buvo 0, bet rinka augo daugiau nei X%, pažymime
threshold = 0.005  # 0.5% per dieną
missed = (spy_data['AI_signal'] == 0) & (spy_data['Close'].pct_change().shift(-1) > threshold)
plt.scatter(spy_data.index[missed], spy_data['Cumulative_market'][missed],
            color='red', label='Missed Opportunities', s=100, marker='^')

plt.title('AI Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
