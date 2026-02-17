import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
def load_data():
    url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df


# =========================
# ADD INDICATORS
# =========================
def add_indicators(df, fast=12, slow=26, signal=9):

    df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return"] = df["Close"].pct_change()

    return df

def run_macd_strategy(df, commission=0.001):

    df["Position"] = np.where(
        (df["MACD"] > df["Signal"]),
        1,
        0
    )

    df["Position"] = df["Position"].shift(1).fillna(0)

    df["Strategy_Return"] = df["Position"] * df["Return"]

    # Commission
    df["Trade"] = df["Position"].diff().abs()
    df["Cost"] = df["Trade"] * commission

    df["Net_Return"] = df["Strategy_Return"] - df["Cost"]
    df["Equity"] = (1 + df["Net_Return"]).cumprod()

    return df

def max_drawdown(equity):
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    return drawdown.min()

def optimize_macd(df):

    best_sharpe = -999
    best_params = None

    for fast in range(8, 20, 2):
        for slow in range(20, 40, 2):
            for signal in range(5, 15, 2):

                temp = df.copy()
                temp = add_indicators(temp, fast, slow, signal)
                temp = run_macd_strategy(temp)

                sharpe = temp["Net_Return"].mean() / temp["Net_Return"].std()

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (fast, slow, signal)

    return best_params, best_sharpe

def walk_forward_test(df):

    split_date = "2015-01-01"

    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    best_params, _ = optimize_macd(train)

    test = add_indicators(test.copy(), *best_params)
    test = run_macd_strategy(test)

    return test, best_params

def ai_filter(df):

    df["Future_Return"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0).astype(int)

    features = df[["MACD", "RSI"]].dropna()
    target = df.loc[features.index, "Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    df["AI_Prediction"] = model.predict(features)

    df["AI_Position"] = df["Position"] * df["AI_Prediction"]

    df["AI_Return"] = df["AI_Position"].shift(1) * df["Return"]
    df["AI_Equity"] = (1 + df["AI_Return"]).cumprod()

    return df

def main():

    df = load_data()

    df = add_indicators(df)
    df = run_macd_strategy(df)

    print("MACD MDD:", max_drawdown(df["Equity"]))

    wf_test, params = walk_forward_test(df)
    print("Best params:", params)

    df = ai_filter(df)

    print("AI MDD:", max_drawdown(df["AI_Equity"]))


if __name__ == "__main__":
    main()



