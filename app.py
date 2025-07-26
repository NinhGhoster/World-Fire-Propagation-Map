# app.py
import dash
import dash_bootstrap_components as dbc
from modules.layout import create_layout
from modules.callbacks import register_callbacks
from config import FIRMS_API_KEY

# Add dbc.icons.BOOTSTRAP to the list of stylesheets
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
)
server = app.server

app.layout = create_layout(app)
register_callbacks(app, FIRMS_API_KEY)

if __name__ == '__main__':
    app.run(debug=True)