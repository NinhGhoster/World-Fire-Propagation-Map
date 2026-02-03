# modules/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html
from .data_fetcher import get_country_list
from datetime import date, timedelta

def create_layout(app):
    """Creates the layout of the Dash application."""
    try:
        country_df = get_country_list()
        country_options = [
            {'label': row['name'], 'value': row['abreviation']}
            for index, row in country_df.iterrows()
        ]
    except Exception as e:
        print(f"ERROR: Could not fetch country list: {e}")
        country_options = []

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔥 World Fire Propagation Map", className="my-4 text-center"),
                html.P("Real-time wildfire tracking, evacuation planning & firefighter dispatch", 
                      className="text-center text-muted")
            ], width=12)
        ], className="mb-4"),
        
        dcc.Store(id='selected-fire-point'),
        dcc.Store(id='grid-toggle-state', data=False),
        dcc.Store(id='selected-firefighter-stations', data=[]),
        
        dbc.Row([
            # Left Panel - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🗺️ Fire Map Controls"),
                    dbc.CardBody([
                        html.H5("1. Select Region"),
                        html.Label("Country"),
                        dcc.Dropdown(
                            id='country-dropdown', 
                            options=country_options,
                            placeholder="Select a country...",
                            clearable=False
                        ),
                        html.Br(),
                        html.Label("Date"),
                        dcc.DatePickerSingle(
                            id='analysis-date-picker',
                            max_date_allowed=date.today(),
                            date=date.today() - timedelta(days=1)
                        ),
                        html.Div(id="map-status-message", className="mt-2 text-muted"),
                        
                        html.Hr(),
                        html.H5("2. Fire Point Selection"),
                        html.P("Click on a fire point (red dot) to analyze.", 
                              className="small text-muted"),
                        html.Div(id="selection-status", className="mt-2 text-primary fw-bold"),
                        
                        html.Hr(),
                        html.H5("3. Simulation Parameters"),
                        html.Label("Grid Size"),
                        dcc.Dropdown(
                            id='grid-graph-size-dropdown',
                            options=[
                                {'label': '3×3 Grid (Small)', 'value': 3},
                                {'label': '5×5 Grid (Medium)', 'value': 5},
                                {'label': '7×7 Grid (Large)', 'value': 7},
                                {'label': '9×9 Grid (X-Large)', 'value': 9},
                            ],
                            value=7,
                            clearable=False
                        ),
                        html.Br(),
                        html.Label("Fire Spread Rate (λ)"),
                        dcc.Dropdown(
                            id='lambda-dropdown',
                            options=[
                                {'label': 'Slow (0.05)', 'value': 0.05},
                                {'label': 'Medium (0.1)', 'value': 0.1},
                                {'label': 'Fast (0.2)', 'value': 0.2},
                                {'label': 'Very Fast (0.3)', 'value': 0.3},
                            ],
                            value=0.1,
                            clearable=False
                        ),
                        html.Br(),
                        html.Label("Firefighters"),
                        dcc.Dropdown(
                            id='firefighters-dropdown',
                            options=[{'label': str(i), 'value': i} for i in range(1, 5)],
                            value=2,
                            clearable=False
                        ),
                        
                        html.Hr(),
                        html.H5("🌬️ Wind Conditions"),
                        html.Label("Wind Speed (km/h)"),
                        dcc.Dropdown(
                            id='wind-speed-dropdown',
                            options=[
                                {'label': 'Calm (0 km/h)', 'value': 0},
                                {'label': 'Light Breeze (15 km/h)', 'value': 15},
                                {'label': 'Moderate (30 km/h)', 'value': 30},
                                {'label': 'Strong (50 km/h)', 'value': 50},
                                {'label': 'Gale (70 km/h)', 'value': 70},
                                {'label': 'Storm (100 km/h)', 'value': 100},
                            ],
                            value=30,
                            clearable=False
                        ),
                        html.Br(),
                        html.Label("Wind Direction"),
                        dcc.Dropdown(
                            id='wind-direction-dropdown',
                            options=[
                                {'label': '⬆️ North', 'value': 'N'},
                                {'label': '↗️ North-East', 'value': 'NE'},
                                {'label': '➡️ East', 'value': 'E'},
                                {'label': '↘️ South-East', 'value': 'SE'},
                                {'label': '⬇️ South', 'value': 'S'},
                                {'label': '↙️ South-West', 'value': 'SW'},
                                {'label': '⬅️ West', 'value': 'W'},
                                {'label': '↖️ North-West', 'value': 'NW'},
                            ],
                            value='NE',
                            clearable=False
                        ),
                        
                        html.Hr(),
                        html.H5("🚒 Response & Evacuation"),
                        html.Label("Response Time (min)"),
                        dcc.Dropdown(
                            id='response-time-dropdown',
                            options=[
                                {'label': 'Rapid (5 min)', 'value': 5},
                                {'label': 'Fast (10 min)', 'value': 10},
                                {'label': 'Standard (15 min)', 'value': 15},
                                {'label': 'Slow (30 min)', 'value': 30},
                                {'label': 'Delayed (60 min)', 'value': 60},
                            ],
                            value=15,
                            clearable=False
                        ),
                        
                        html.Hr(),
                        dbc.Button("📊 Analyze Region", id="analyze-button", color="primary", 
                                 className="w-100 mb-2", disabled=True),
                        dbc.Button("🧯 Run Simulation", id="simulate-button", color="success",
                                 className="w-100 mb-2"),
                        dbc.Button("📈 Compare Strategies", id="compare-button", color="info",
                                 className="w-100 mb-2"),
                        dbc.Button("🚨 Get Evacuation Plan", id="evacuate-button", color="danger",
                                 className="w-100 mb-2"),
                        
                        html.Hr(),
                        html.H5("Quick Demo"),
                        dbc.Button("🎯 Load Australia Demo", id="demo-button", color="warning",
                                 className="w-100 mb-2"),
                    ])
                ], className="shadow-sm")
            ], md=4),
            
            # Right Panel - Map & Results
            dbc.Col([
                dcc.Loading(id="loading-map", type="default", children=[
                    dcc.Graph(id="fire-map", style={"height": "500px"}),
                ]),
                html.Div(id="results-output", className="mt-3")
            ], md=8)
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    "🔥 World Fire Propagation Map v2.0 | ",
                    "Data: NASA FIRMS | ",
                    "🚒 Fire Station Coverage | ",
                    "🚨 Evacuation Planning | ",
                    "Powered by Dash + Plotly"
                ], className="text-center text-muted small")
            ])
        ])
    ], fluid=True)
