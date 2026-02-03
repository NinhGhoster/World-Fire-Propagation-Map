# modules/layout.py - v3.0 Modern Dashboard
import dash_bootstrap_components as dbc
from dash import dcc, html
from .data_fetcher import get_country_list
from datetime import date, timedelta

def create_layout(app):
    """Creates the modern v3.0 layout."""
    try:
        country_df = get_country_list()
        country_options = [
            {'label': row['name'], 'value': row['abreviation']}
            for index, row in country_df.iterrows()
        ]
    except Exception as e:
        country_options = []

    return dbc.Container([
        # Header with status bar
        dbc.Row([
            dbc.Col([
                html.H1("🔥 World Fire Propagation Map v3.0", className="my-3"),
                html.Span("🚀 100x Better - Real-time Analytics", className="badge bg-success ms-2"),
            ], width=8),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Span(id="live-status", className="badge bg-success"),
                        html.Span(" Live", className="small text-muted"),
                    ], width="auto"),
                    dbc.Col([
                        html.Span(id="fire-count", className="badge bg-danger ms-2"),
                        html.Span(" Active Fires", className="small text-muted"),
                    ], width="auto"),
                ], justify="end"),
            ], width=4),
        ], className="mb-3"),
        
        dcc.Store(id='selected-fire-point'),
        dcc.Interval(id='refresh-interval', interval=300000, n_intervals=0),  # 5 min auto-refresh
        
        dbc.Row([
            # Left Panel - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🗺️ Region Selection", className="mb-0 d-inline"),
                        dcc.Dropdown(
                            id='country-dropdown', 
                            options=country_options,
                            placeholder="Select country...",
                            clearable=False,
                            className="mt-2"
                        ),
                    ], className="pb-2"),
                    dbc.CardBody([
                        # Analysis Section
                        html.H6("📊 Analysis", className="text-primary"),
                        dbc.Button("🔥 Find Hotspots", id="btn-hotspots", color="danger", 
                                 className="w-100 mb-2", size="sm"),
                        dbc.Button("📈 Seasonal Analysis", id="btn-seasonal", color="info", 
                                 className="w-100 mb-2", size="sm"),
                        dbc.Button("⚠️ Risk Assessment", id="btn-risk", color="warning", 
                                 className="w-100 mb-2", size="sm"),
                        
                        html.Hr(),
                        
                        # Simulation Section
                        html.H6("🧯 Simulation", className="text-success"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Grid", className="small"),
                                dcc.Dropdown(id='grid-size', options=[
                                    {'label': '5×5', 'value': 5}, {'label': '7×7', 'value': 7},
                                    {'label': '9×9', 'value': 9}, {'label': '11×11', 'value': 11}
                                ], value=7, clearable=False, className="small"),
                            ], width=6),
                            dbc.Col([
                                html.Label("λ Spread", className="small"),
                                dcc.Dropdown(id='lambda-spread', options=[
                                    {'label': '0.05', 'value': 0.05}, {'label': '0.1', 'value': 0.1},
                                    {'label': '0.2', 'value': 0.2}, {'label': '0.3', 'value': 0.3}
                                ], value=0.1, clearable=False, className="small"),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Firefighters", className="small"),
                                dcc.Dropdown(id='firefighters', options=[
                                    {'label': str(i), 'value': i} for i in range(1, 6)
                                ], value=2, clearable=False, className="small"),
                            ], width=6),
                            dbc.Col([
                                html.Label("Firefighters", className="small"),
                                dcc.Dropdown(id='strategy', options=[
                                    {'label': 'Greedy', 'value': 'greedy'},
                                    {'label': 'Random', 'value': 'random'},
                                    {'label': 'Central', 'value': 'central'}
                                ], value='greedy', clearable=False, className="small"),
                            ], width=6),
                        ], className="mb-2"),
                        dbc.Button("▶️ Run Simulation", id="btn-simulate", color="success", 
                                 className="w-100 mb-2", size="sm"),
                        
                        html.Hr(),
                        
                        # Weather Section
                        html.H6("🌤️ Weather", className="text-info"),
                        dbc.Row([
                            dbc.Col([
                                html.Span(id="weather-temp", className="h6 text-warning"),
                                html.Span("°C", className="small"),
                            ], width=4, className="text-center"),
                            dbc.Col([
                                html.Span(id="weather-humidity", className="h6 text-primary"),
                                html.Span("%", className="small"),
                            ], width=4, className="text-center"),
                            dbc.Col([
                                html.Span(id="weather-wind", className="h6 text-info"),
                                html.Span("km/h", className="small"),
                            ], width=4, className="text-center"),
                        ], className="mb-2"),
                        dbc.Badge(id="fire-danger", color="danger", className="w-100 text-center p-2"),
                        
                        html.Hr(),
                        
                        # Quick Actions
                        html.H6("⚡ Quick Actions", className="text-secondary"),
                        dbc.Button("🎯 Load Demo", id="btn-demo", color="primary", 
                                 className="w-100 mb-1", size="sm"),
                        dbc.Button("📊 Compare Strategies", id="btn-compare", color="secondary", 
                                 className="w-100 mb-1", size="sm"),
                        dbc.Button("🔄 Refresh Data", id="btn-refresh", outline=True, 
                                 className="w-100", size="sm"),
                    ])
                ], className="shadow-sm")
            ], md=3),
            
            # Center - Map
            dbc.Col([
                dcc.Loading(id="loading-map", type="default", children=[
                    dcc.Graph(id="fire-map", style={"height": "70vh"}),
                ]),
                html.Div(id="selection-status", className="mt-2 text-center"),
            ], md=6),
            
            # Right Panel - Results & Analytics
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab([
                        html.Div(id="analytics-output", className="p-3")
                    ], label="📊 Analytics"),
                    dbc.Tab([
                        html.Div(id="simulation-output", className="p-3")
                    ], label="🧯 Simulation"),
                    dbc.Tab([
                        html.Div(id="weather-output", className="p-3")
                    ], label="🌤️ Weather"),
                    dbc.Tab([
                        html.Div(id="hotspots-output", className="p-3")
                    ], label="🔥 Hotspots"),
                ], id="tabs", active_tab="analytics-output"),
            ], md=3),
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    "🔥 World Fire Propagation Map v3.0 | ",
                    "📊 NASA FIRMS + Weather API + Analytics | ",
                    "🚀 CI/CD Powered by GitHub Actions"
                ], className="text-center text-muted small")
            ])
        ])
    ], fluid=True)
