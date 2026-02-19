# test_dash.py
import dash
from dash import html

# Sukuriame paprastą Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dash veikia! ✅"),
    html.P("Jei matai šį tekstą, Dash yra įdiegtas ir veikia.")
])

if __name__ == '__main__':
    app.run_server(debug=True)
