import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Dummy data (vietoje šių įkelk savo spy_data)
# -----------------------------
dates = pd.date_range(start='2024-01-01', periods=500, freq='B')
np.random.seed(42)
spy_data = pd.DataFrame({
    'Close': np.cumsum(np.random.randn(500)) + 100,
    'AI_signal': np.random.randint(0,2,500)
}, index=dates)

# Returns & cumulative
spy_data['Market_returns'] = spy_data['Close'].pct_change()
spy_data['Cumulative_market'] = (1 + spy_data['Market_returns']).cumprod()
spy_data['Strategy_returns'] = spy_data['Market_returns'] * spy_data['AI_signal']
spy_data['Cumulative_strategy'] = (1 + spy_data['Strategy_returns']).cumprod()

# Drawdowns
spy_data['Drawdown_market'] = spy_data['Cumulative_market'] / spy_data['Cumulative_market'].cummax() - 1
spy_data['Drawdown_strategy'] = spy_data['Cumulative_strategy'] / spy_data['Cumulative_strategy'].cummax() - 1

# -----------------------------
# Helper functions
# -----------------------------
def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(cumulative):
    roll_max = cumulative.cummax()
    return (cumulative / roll_max - 1).min()

# -----------------------------
# 2️⃣ Dash app
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard", style={'textAlign':'center'}),

    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=spy_data.index.min(),
            end_date=spy_data.index.max(),
            min_date_allowed=spy_data.index.min(),
            max_date_allowed=spy_data.index.max()
        )
    ], style={'textAlign':'center', 'marginBottom':'20px'}),

    dcc.Graph(id='cum_returns'),
    dcc.Graph(id='drawdowns'),
    dcc.Graph(id='rolling_metrics'),
    dcc.Graph(id='heatmap_returns'),

    html.Div(id='metrics', style={'textAlign':'center', 'marginTop':'20px', 'fontSize':'18px'})
])

# -----------------------------
# 3️⃣ Callbacks
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
    df = spy_data.loc[start_date:end_date]

    # --- Cumulative Returns ---
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_market'], name='Buy & Hold', mode='lines+markers'))
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['Cumulative_strategy'], name='AI Strategy', mode='lines+markers'))
    threshold = 0.005
    missed = (df['AI_signal']==0) & (df['Market_returns'].shift(-1) > threshold)
    fig_cum.add_trace(go.Scatter(
        x=df.index[missed],
        y=df['Cumulative_market'][missed],
        mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'),
        name='Missed Opportunities'
    ))
    fig_cum.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

    # --- Drawdowns ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown_market'], name='Market Drawdown', line=dict(color='blue')))
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Drawdown_strategy'], name='Strategy Drawdown', line=dict(color='green')))
    fig_dd.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown')

    # --- Rolling Metrics (Sharpe & Max Drawdown) ---
    window = 20
    rolling_sharpe_market = df['Market_returns'].rolling(window).mean() / df['Market_returns'].rolling(window).std() * np.sqrt(252)
    rolling_sharpe_strategy = df['Strategy_returns'].rolling(window).mean() / df['Strategy_returns'].rolling(window).std() * np.sqrt(252)
    rolling_dd_market = df['Cumulative_market'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min())
    rolling_dd_strategy = df['Cumulative_strategy'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min())

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_sharpe_market, name='Market Rolling Sharpe', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_sharpe_strategy, name='Strategy Rolling Sharpe', line=dict(color='green')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_dd_market, name='Market Rolling DD', line=dict(color='red', dash='dot')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_dd_strategy, name='Strategy Rolling DD', line=dict(color='orange', dash='dot')))
    fig_rolling.update_layout(title='Rolling Sharpe & Max Drawdown', xaxis_title='Date')

    # --- Heatmap: returns per weekday ---
    df['Weekday'] = df.index.day_name()
    heatmap_data = df.pivot_table(index='Weekday', values=['Market_returns','Strategy_returns'], aggfunc=np.mean)
    fig_heatmap = go.Figure()
    fig_heatmap.add_trace(go.Heatmap(
        z=heatmap_data['Market_returns'].values,
        x=heatmap_data.index,
        y=['Buy & Hold'],
        colorscale='Blues', showscale=True, name='Market'
    ))
    fig_heatmap.add_trace(go.Heatmap(
        z=heatmap_data['Strategy_returns'].values,
        x=heatmap_data.index,
        y=['AI Strategy'],
        colorscale='Greens', showscale=True, name='Strategy'
    ))
    fig_heatmap.update_layout(title='Average Daily Returns by Weekday', yaxis_title='Strategy')

    # --- Metrics panel ---
    metrics_text = [
        html.P(f"Market Sharpe: {round(sharpe_ratio(df['Market_returns'].dropna()),3)}"),
        html.P(f"Strategy Sharpe: {round(sharpe_ratio(df['Strategy_returns'].dropna()),3)}"),
        html.P(f"Market Max Drawdown: {round(max_drawdown(df['Cumulative_market']),3)}"),
        html.P(f"Strategy Max Drawdown: {round(max_drawdown(df['Cumulative_strategy']),3)}"),
        html.P(f"AI_signal distribution: {df['AI_signal'].value_counts().to_dict()}")
    ]

    return fig_cum, fig_dd, fig_rolling, fig_heatmap, metrics_text

# -----------------------------
# 4️⃣ Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
