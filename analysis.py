import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html

# --- 1. Dummy data ---
data = {
    'Close': [669.42, 674.48, 680.59, 684.83, 687.96, 690.38, 690.31, 687.85, 687.01, 681.92],
    'AI_signal': [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
}
dates = pd.date_range(start='2025-12-17', periods=10, freq='B')
df = pd.DataFrame(data, index=dates)
df['Market_returns'] = df['Close'].pct_change()
df['Strategy_returns'] = df['Market_returns'] * df['AI_signal']
df['Cumulative_market'] = (1 + df['Market_returns']).cumprod()
df['Cumulative_strategy'] = (1 + df['Strategy_returns']).cumprod()
df['Drawdown_market'] = df['Cumulative_market'] / df['Cumulative_market'].cummax() - 1
df['Drawdown_strategy'] = df['Cumulative_strategy'] / df['Cumulative_strategy'].cummax() - 1

# --- 2. Missed opportunities ---
threshold = 0.005
df['Missed'] = (df['AI_signal']==0) & (df['Market_returns'] > threshold)

# --- 3. Dash App ---
app = Dash(__name__)

fig = make_subplots(rows=2, cols=2, subplot_titles=("Cumulative Returns", "Daily Returns", "Drawdowns", "Missed Opportunities"))

# a) Cumulative Returns
fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_market'], mode='lines+markers', name='Buy & Hold'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_strategy'], mode='lines+markers', name='AI Strategy'), row=1, col=1)

# b) Daily Returns
fig.add_trace(go.Bar(x=df.index, y=df['Market_returns'], name='Market Returns'), row=1, col=2)
fig.add_trace(go.Bar(x=df.index, y=df['Strategy_returns'], name='Strategy Returns'), row=1, col=2)

# c) Drawdowns
fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown_market'], mode='lines+markers', name='Market Drawdown'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown_strategy'], mode='lines+markers', name='Strategy Drawdown'), row=2, col=1)

# d) Missed Opportunities
fig.add_trace(go.Scatter(x=df.index[df['Missed']], y=df['Cumulative_market'][df['Missed']], mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'), name='Missed'), row=2, col=2)
fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_market'], mode='lines+markers', name='Buy & Hold'), row=2, col=2)
fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_strategy'], mode='lines+markers', name='AI Strategy'), row=2, col=2)

fig.update_layout(height=800, width=1200, title_text="Professional Trading Dashboard")

app.layout = html.Div([
    html.H1("AI Strategy vs Buy & Hold Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
