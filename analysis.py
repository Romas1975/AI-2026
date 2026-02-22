import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================
# 1️⃣ DATA LOADING
# ============================================

TICKER = "SPY"
PERIOD = "2y"

df = yf.download(TICKER, period=PERIOD, auto_adjust=True)

# Sutvarkom MultiIndex jei atsiranda
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.sort_index()

if df.empty:
    raise ValueError("No data downloaded from Yahoo Finance")

# ============================================
# 2️⃣ STRATEGY LOGIC
# ============================================

# Market returns
df["Market_returns"] = df["Close"].pct_change()

target_vol = 0.15

df["Vol_20"] = df["Market_returns"].rolling(20).std() * np.sqrt(252)

df["AI_signal"] = target_vol / df["Vol_20"]
df["AI_signal"] = df["AI_signal"].clip(0.2, 1.5)

# Dummy AI signal (kol neturim modelio)

# MACD
df["Close"] = df["Close"].astype(float)

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["AI_signal"] = 0
df.loc[df["MACD"] > df["Signal_line"], "AI_signal"] = 1



# Strategy returns
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"]
# Be look-ahead
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"].shift(1)

df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()

df["Drawdown_strategy"] = df["Cumulative_strategy"] / df["Cumulative_strategy"].cummax() - 1

# Rolling volatility
df["Volatility"] = df["Market_returns"].rolling(20).std() * np.sqrt(252)

vol_threshold = df["Volatility"].median()

# Scale exposure pagal volatility lygį
max_vol = df["Volatility"].quantile(0.9)

df["AI_signal"] = 1 - (df["Volatility"] / max_vol)
df["AI_signal"] = df["AI_signal"].clip(lower=0.2, upper=1.0)

df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"].shift(1)

# Cumulative
df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()

# Drawdown
df["Drawdown_market"] = df["Cumulative_market"] / df["Cumulative_market"].cummax() - 1
df["Drawdown_strategy"] = df["Cumulative_strategy"] / df["Cumulative_strategy"].cummax() - 1


# ============================================
# 3️⃣ METRICS FUNCTIONS
# ============================================

def sharpe_ratio(returns):
    returns = returns.dropna()
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    drawdown = cumulative / roll_max - 1
    return drawdown.min()

def exposure(signal):
    return signal.mean()


# ============================================
# 4️⃣ DASH APP
# ============================================

app = dash.Dash(__name__)
app.title = "AI Trading Dashboard"

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold", style={'textAlign':'center'}),

    dcc.DatePickerRange(
        id='date-range',
        start_date=df.index.min(),
        end_date=df.index.max(),
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max()
    ),

    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),

    html.Div(id='metrics', style={
        'textAlign':'center',
        'marginTop':'20px',
        'fontSize':'18px'
    })
])


# ============================================
# 5️⃣ CALLBACK
# ============================================

@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('metrics','children'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(start_date, end_date):

    dff = df.loc[start_date:end_date].copy()

    if dff.empty:
        return go.Figure(), go.Figure(), go.Figure(), "No data in selected range"

    # -----------------------
    # CUMULATIVE RETURNS
    # -----------------------
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=dff.index, y=dff['Cumulative_market'],
        name='Buy & Hold'
    ))
    fig_cum.add_trace(go.Scatter(
        x=dff.index, y=dff['Cumulative_strategy'],
        name='AI Strategy'
    ))
    fig_cum.update_layout(title='Cumulative Returns')

    # -----------------------
    # DRAWDOWNS
    # -----------------------
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dff.index, y=dff['Drawdown_market'],
        name='Market Drawdown'
    ))
    fig_dd.add_trace(go.Scatter(
        x=dff.index, y=dff['Drawdown_strategy'],
        name='Strategy Drawdown'
    ))
    fig_dd.update_layout(title='Drawdowns')

    # -----------------------
    # ROLLING SHARPE (optimized)
    # -----------------------
    window = 20

    rolling_sharpe_market = (
        dff['Market_returns'].rolling(window).mean() /
        dff['Market_returns'].rolling(window).std()
    ) * np.sqrt(252)

    rolling_sharpe_strategy = (
        dff['Strategy_returns'].rolling(window).mean() /
        dff['Strategy_returns'].rolling(window).std()
    ) * np.sqrt(252)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=dff.index, y=rolling_sharpe_market,
        name='Market Rolling Sharpe'
    ))
    fig_roll.add_trace(go.Scatter(
        x=dff.index, y=rolling_sharpe_strategy,
        name='Strategy Rolling Sharpe'
    ))
    fig_roll.update_layout(title='Rolling Sharpe (20d)')

    # -----------------------
    # METRICS PANEL
    # -----------------------
    metrics_panel = [
        html.P(f"Total Market Return: {round(dff['Cumulative_market'].iloc[-1]-1,3)}"),
        html.P(f"Total Strategy Return: {round(dff['Cumulative_strategy'].iloc[-1]-1,3)}"),
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns']),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(dff['Strategy_returns']),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(dff['Cumulative_strategy']),3)}"),
        html.P(f"Strategy Exposure: {round(exposure(dff['AI_signal']),3)}")
    ]

    return fig_cum, fig_dd, fig_roll, metrics_panel


# ============================================
# 6️⃣ RUN
# ============================================

if __name__ == '__main__':
    app.run(debug=False)