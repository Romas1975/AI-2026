import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------
# 1️⃣ Helper functions
# -----------------------------
def compute_macd(df, span_fast, span_slow, span_signal):
    ema_fast = df['Close'].ewm(span=span_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal

def compute_metrics(df):
    df['Market_returns'] = df['Close'].pct_change()
    df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
    df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
    df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
    df['Drawdown_market'] = df['Cumulative_market'] / df['Cumulative_market'].cummax() - 1
    df['Drawdown_strategy'] = df['Cumulative_strategy'] / df['Cumulative_strategy'].cummax() - 1
    
    metrics = {
        'Total Market Return': df['Cumulative_market'].iloc[-1] - 1,
        'Total Strategy Return': df['Cumulative_strategy'].iloc[-1] - 1,
        'Market Sharpe': np.sqrt(252)*df['Market_returns'].mean()/df['Market_returns'].std(),
        'Strategy Sharpe': np.sqrt(252)*df['Strategy_returns'].mean()/df['Strategy_returns'].std(),
        'Market Max Drawdown': df['Drawdown_market'].min(),
        'Strategy Max Drawdown': df['Drawdown_strategy'].min(),
        'Strategy Exposure (mean)': df['AI_signal'].mean(),
        'Buy Signals': int((df['AI_signal']==1).sum()),
        'Sell Signals': int((df['AI_signal']==0).sum())
    }
    return df, metrics

# -----------------------------
# 2️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "AI Strategy vs Buy & Hold"

# -----------------------------
# 3️⃣ Layout
# -----------------------------
app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),
    html.Div([
        html.Label("Select Ticker:"),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': t, 'value': t} for t in ['SPY','AAPL','MSFT','GOOG']],
            value='SPY',
            clearable=False
        ),
        html.Br(),
        html.Label("MACD Parameters:"),
        html.Div(["Fast EMA:", dcc.Slider(id='ema-fast', min=5, max=30, step=1, value=12, tooltip={"placement":"bottom"})]),
        html.Div(["Slow EMA:", dcc.Slider(id='ema-slow', min=20, max=60, step=1, value=26, tooltip={"placement":"bottom"})]),
        html.Div(["Signal span:", dcc.Slider(id='signal-span', min=5, max=20, step=1, value=9, tooltip={"placement":"bottom"})])
    ], style={'marginBottom':'20px','padding':'10px','border':'1px solid #ccc'}),
    
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    dcc.Graph(id='heatmap_returns'),
    html.Div(id='metrics', style={'textAlign':'center','marginTop':'20px','fontSize':'18px'})
])

# -----------------------------
# 4️⃣ Callback
# -----------------------------
@app.callback(
    Output('cum_returns','figure'),
    Output('drawdowns','figure'),
    Output('rolling_metrics','figure'),
    Output('heatmap_returns','figure'),
    Output('metrics','children'),
    Input('ticker-dropdown','value'),
    Input('ema-fast','value'),
    Input('ema-slow','value'),
    Input('signal-span','value')
)
def update_dashboard(ticker, span_fast, span_slow, span_signal):
    df = yf.download(ticker, period="1y", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    macd, signal = compute_macd(df, span_fast, span_slow, span_signal)
    df['MACD'] = macd
    df['Signal_line'] = signal
    df['AI_signal'] = np.where(df['MACD'] > df['Signal_line'],1,0)

    df, metrics = compute_metrics(df)
    
    # --- Cumulative Returns ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))

    # --- Drawdowns ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))

    # --- Rolling Metrics ---
    window = 20
    rolling_sharpe_market = df['Market_returns'].rolling(window).mean() / df['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = df['Strategy_returns'].rolling(window).mean() / df['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = df['Cumulative_market'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = df['Cumulative_strategy'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))

    # --- Heatmap ---
    df['Weekday'] = df.index.day_name()
    heatmap_data = df.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Market_returns'].values, x=heatmap_data.index, y=['Buy & Hold'], colorscale='Blues', showscale=True))
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Strategy_returns'].values, x=heatmap_data.index, y=['AI Strategy'], colorscale='Greens', showscale=True))

    # --- Metrics Panel ---
    metrics_text = [html.P(f"{k}: {round(v,3) if isinstance(v,float) else v}") for k,v in metrics.items()]

    return fig_cum, fig_dd, fig_rolling, fig_heatmap, metrics_text

# -----------------------------
# 5️⃣ Run Server
# -----------------------------
if __name__ == '__main__':
    app.run()
