import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------
# 1️⃣ Load SPY data (pavyzdžiui)
# -----------------------------
ticker = "SPY"
df = yf.download(ticker, start="2024-01-01", end="2026-02-18")
df['AI_signal'] = np.random.randint(0,2,len(df))  # Dummy AI signal, vėliau naudok tikrą modelį

# Returns & cumulative
df['Market_returns'] = df['Close'].pct_change()
df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
df['Drawdown_market'] = df['Cumulative_market'] / df['Cumulative_market'].cummax() - 1
df['Drawdown_strategy'] = df['Cumulative_strategy'] / df['Cumulative_strategy'].cummax() - 1

# -----------------------------
# 2️⃣ Metrics functions
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    return (cumulative / roll_max - 1).min()

# -----------------------------
# 3️⃣ Dash App
# -----------------------------
app = dash.Dash(__name__)
app.title = "AI Strategy vs Buy & Hold"

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),
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
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_dashboard(start_date, end_date):
    dff = df.loc[start_date:end_date]

    # --- Cumulative Returns
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=dff.index, y=dff['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))
    threshold = 0.005
    missed = (dff['AI_signal']==0) & (dff['Market_returns'].shift(-1) > threshold)
    fig_cum.add_trace(go.Scatter(x=dff.index[missed], y=dff['Cumulative_market'][missed], mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'), name='Missed Opportunities'))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # --- Drawdowns
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=dff.index, y=dff['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # --- Rolling metrics
    window = 20
    rolling_sharpe_market = dff['Market_returns'].rolling(window).mean() / dff['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = dff['Strategy_returns'].rolling(window).mean() / dff['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = dff['Cumulative_market'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())
    rolling_dd_strategy = dff['Cumulative_strategy'].rolling(window).apply(lambda x: (x/x.cummax()-1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=dff.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))
    fig_rolling.update_layout(title='Rolling Sharpe & Max Drawdown', xaxis_title='Date')

    # --- Heatmap: average returns per weekday
    dff['Weekday'] = dff.index.day_name()
    heatmap_data = dff.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc='mean')
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Market_returns'].values, x=heatmap_data.index, y=['Buy & Hold'], colorscale='Blues', showscale=True))
    fig_heatmap.add_trace(go.Heatmap(z=heatmap_data['Strategy_returns'].values, x=heatmap_data.index, y=['AI Strategy'], colorscale='Greens', showscale=True))
    fig_heatmap.update_layout(title='Average Daily Returns by Weekday', yaxis_title='Strategy')

    # --- Metrics panel
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(dff['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(dff['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(dff['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(dff['Cumulative_strategy']),3)}"),
        html.P(f"AI_signal distribution: {dff['AI_signal'].value_counts().to_dict()}")
    ]

    return fig_cum, fig_dd, fig_rolling, fig_heatmap, metrics_text

# -----------------------------
# 5️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run()
