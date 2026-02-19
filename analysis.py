# analysis_dash.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------
# 1️⃣ Parsisiunčiame SPY duomenis
# -----------------------------
ticker = "SPY"
start_date = "2024-01-01"
end_date = "2026-02-18"

df = yf.download(ticker, start=start_date, end=end_date)
df['Market_returns'] = df['Close'].pct_change()
df['AI_signal'] = np.random.randint(0, 2, len(df))  # Dummy AI signals
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']

df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()

df['Drawdown_market'] = df['Cumulative_market'] / df['Cumulative_market'].cummax() - 1
df['Drawdown_strategy'] = df['Cumulative_strategy'] / df['Cumulative_strategy'].cummax() - 1

# -----------------------------
# 2️⃣ Rizikos ir performance metrikos
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def sortino_ratio(returns):
    neg_std = returns[returns < 0].std()
    return np.sqrt(252) * returns.mean() / neg_std if neg_std != 0 else np.nan

def max_drawdown(cum_returns):
    return (cum_returns / cum_returns.cummax() - 1).min()

def var_95(returns):
    return np.percentile(returns.dropna(), 5)

metrics = {
    "Market Sharpe": round(sharpe_ratio(df['Market_returns'].dropna()), 3),
    "Strategy Sharpe": round(sharpe_ratio(df['Strategy_returns'].dropna()), 3),
    "Market Sortino": round(sortino_ratio(df['Market_returns'].dropna()), 3),
    "Strategy Sortino": round(sortino_ratio(df['Strategy_returns'].dropna()), 3),
    "Market Max Drawdown": round(max_drawdown(df['Cumulative_market']), 3),
    "Strategy Max Drawdown": round(max_drawdown(df['Cumulative_strategy']), 3),
    "Market VaR 95%": round(var_95(df['Market_returns']), 3),
    "Strategy VaR 95%": round(var_95(df['Strategy_returns']), 3),
    "AI_signal count": df['AI_signal'].value_counts().to_dict()
}

# -----------------------------
# 3️⃣ Dash app
# -----------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),
    
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df.index.min(),
            end_date=df.index.max(),
            min_date_allowed=df.index.min(),
            max_date_allowed=df.index.max()
        )
    ], style={'textAlign':'center', 'marginBottom':'20px'}),
    
    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    dcc.Graph(id='heatmap_returns'),
    
    html.Div(id='metrics', style={'textAlign':'center', 'marginTop':'20px', 'fontSize':'18px'})
])

# -----------------------------
# 4️⃣ Callbacks
# -----------------------------
@app.callback(
    Output('cum_returns', 'figure'),
    Output('drawdowns', 'figure'),
    Output('rolling_metrics', 'figure'),
    Output('heatmap_returns', 'figure'),
    Output('metrics', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_dashboard(start_date, end_date):
    df_filtered = df.loc[start_date:end_date]
    
    # --- Cumulative Returns ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))
    
    # Missed opportunities
    threshold = 0.005
    missed = (df_filtered['AI_signal']==0) & (df_filtered['Market_returns'].shift(-1) > threshold)
    fig_cum.add_trace(go.Scatter(
        x=df_filtered.index[missed],
        y=df_filtered['Cumulative_market'][missed],
        mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'),
        name='Missed Opportunities'
    ))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')
    
    # --- Drawdowns ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')
    
    # --- Rolling metrics ---
    window = 20
    rolling_sharpe_market = df_filtered['Market_returns'].rolling(window).mean() / df_filtered['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = df_filtered['Strategy_returns'].rolling(window).mean() / df_filtered['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = df_filtered['Cumulative_market'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min())
    rolling_dd_strategy = df_filtered['Cumulative_strategy'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min())
    
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df_filtered.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=df_filtered.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=df_filtered.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=df_filtered.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))
    fig_rolling.update_layout(title='Rolling Sharpe & Max Drawdown', xaxis_title='Date')
    
    # --- Heatmap per weekday ---
    df_filtered['Weekday'] = df_filtered.index.day_name()
    heatmap_data = df_filtered.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(
        z=heatmap_data['Market_returns'].values,
        x=heatmap_data.index,
        y=['Buy & Hold'],
        colorscale='Blues', showscale=True
    ))
    fig_heatmap.add_trace(go.Heatmap(
        z=heatmap_data['Strategy_returns'].values,
        x=heatmap_data.index,
        y=['AI Strategy'],
        colorscale='Greens', showscale=True
    ))
    fig_heatmap.update_layout(title='Average Daily Returns by Weekday', yaxis_title='Strategy')
    
    # --- Metrics panel ---
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(df_filtered['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(df_filtered['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Sortino: {round(sortino_ratio(df_filtered['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sortino: {round(sortino_ratio(df_filtered['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(df_filtered['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(df_filtered['Cumulative_strategy']),3)}"),
        html.P(f"Market VaR 95%: {round(var_95(df_filtered['Market_returns']),3)}"),
        html.P(f"Strategy VaR 95%: {round(var_95(df_filtered['Strategy_returns']),3)}"),
        html.P(f"AI_signal distribution: {df_filtered['AI_signal'].value_counts().to_dict()}")
    ]
    
    return fig_cum, fig_dd, fig_rolling, fig_heatmap, metrics_text

# -----------------------------
# 5️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
