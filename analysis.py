import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
ticker = "SPY"
df = yf.download(ticker, period="2y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty:
    raise ValueError("Dataframe is empty. Yahoo returned no data.")

# -----------------------------
# 2️⃣ MACD Strategy
# -----------------------------
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# AI_signal: 1 = MACD > Signal line, else 0
df["AI_signal"] = np.where(df["MACD"] > df["Signal_line"], 1, 0)

# Returns
df["Market_returns"] = df["Close"].pct_change()
df["Strategy_returns"] = df["Market_returns"] * df["AI_signal"]

# Cumulative & Drawdowns
df["Cumulative_market"] = (1 + df["Market_returns"]).cumprod()
df["Cumulative_strategy"] = (1 + df["Strategy_returns"]).cumprod()
df["Drawdown_market"] = df["Cumulative_market"]/df["Cumulative_market"].cummax() - 1
df["Drawdown_strategy"] = df["Cumulative_strategy"]/df["Cumulative_strategy"].cummax() - 1

# Rolling volatility
window = 20
df["Vol_20"] = df["Market_returns"].rolling(window).std() * np.sqrt(252)
df["Exposure"] = df["AI_signal"].rolling(window).mean()

# -----------------------------
# 3️⃣ Metrics functions
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    return (cumulative / cumulative.cummax() - 1).min()

# -----------------------------
# 4️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "MACD Strategy Dashboard"

app.layout = html.Div([
    html.H1("MACD Strategy Dashboard", style={'textAlign':'center'}),
    
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df.index.min(),
            end_date=df.index.max(),
            min_date_allowed=df.index.min(),
            max_date_allowed=df.index.max()
        )
    ], style={'textAlign':'center','marginBottom':'20px'}),

    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    dcc.Graph(id='strategy_exposure'),
    dcc.Graph(id='heatmap_returns'),

    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 5️⃣ Callbacks
# -----------------------------
@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('strategy_exposure','figure'),
    Output('heatmap_returns','figure'),
    Output('metrics','children'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(start_date, end_date):
    dff = df.loc[start_date:end_date].copy()

    # --- Cumulative Returns + signals ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Buy & Hold', line=dict(color='blue')))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='MACD Strategy', line=dict(color='green')))

    # Signal markers
    buy_signals = dff[(dff['AI_signal'] == 1) & (dff['AI_signal'].shift(1)==0)]
    sell_signals = dff[(dff['AI_signal'] == 0) & (dff['AI_signal'].shift(1)==1)]
    fig_cum.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Cumulative_strategy'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
    fig_cum.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Cumulative_strategy'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))

    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # --- Drawdowns ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # --- Rolling metrics ---
    roll_window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(roll_window).mean() / dff['Market_returns'].rolling(roll_window).std() * np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(roll_window).mean() / dff['Strategy_returns'].rolling(roll_window).std() * np.sqrt(252)
    rolling_dd_market = dff['Cumulative_market'].rolling(roll_window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = dff['Cumulative_strategy'].rolling(roll_window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_market, name='Market DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_strategy, name='Strategy DD', line=dict(color='orange', dash='dot')))
    fig_rolling.update_layout(title='Rolling Sharpe & Drawdown', xaxis_title='Date')

    # --- Strategy Exposure ---
    fig_exposure = go.Figure()
    fig_exposure.add_trace(go.Scatter(x=dff.index, y=dff['Exposure'], name='Strategy Exposure', line=dict(color='purple')))
    fig_exposure.update_layout(title='Strategy Exposure (Rolling 20 days)', xaxis_title='Date', yaxis_title='Exposure')

    # --- Heatmap: average returns per weekday ---
    dff['Weekday'] = dff.index.day_name()
    heatmap_data = dff.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Market_returns'].values, x=heatmap_data.index, y=['Market'], colorscale='Blues', showscale=True))
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Strategy_returns'].values, x=heatmap_data.index, y=['Strategy'], colorscale='Greens', showscale=True))
    fig_heatmap.update_layout(title='Average Daily Returns by Weekday', yaxis_title='Strategy')

    # --- Metrics panel ---
    metrics_text = [
        html.P(f"Total Market Return: {round(dff['Cumulative_market'].iloc[-1]-1,3)}"),
        html.P(f"Total Strategy Return: {round(dff['Cumulative_strategy'].iloc[-1]-1,3)}"),
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(dff['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(dff['Cumulative_strategy']),3)}"),
        html.P(f"Strategy Exposure (mean): {round(dff['Exposure'].mean(),3)}"),
        html.P(f"Number of Buy signals: {len(buy_signals)}"),
        html.P(f"Number of Sell signals: {len(sell_signals)}")
    ]

    return fig_cum, fig_dd, fig_rolling, fig_exposure, fig_heatmap, metrics_text

# -----------------------------
# 6️⃣ Run Server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)