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
        print(f"ERROR: Could not fetch country list on startup: {e}")
        country_options = []

    return dbc.Container(
        [
            # Header with logo and title
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src=app.get_asset_url('quantathon-logo-full.png'),
                        style={
                            'height': '100px',
                            'width': 'auto',
                            'marginRight': '20px'
                        }
                    )
                ], width=3),
                dbc.Col([
                    html.H1("QPreFire - Optimise the Wildfire Response", className="my-4 text-center")
                ], width=9)
            ], className="mb-4", align="center"),
            
            dcc.Store(id='selected-fire-point'),
            dcc.Store(id='grid-toggle-state', data=False),
            dcc.Store(id='selected-firefighter-stations', data=[]),
            dcc.Store(id='latest-saved-file', data=None),
            dcc.Store(id='mff-solution-data', data=None),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.H4("1. Select Country and Date"),
                        html.Label("Country"),
                        dcc.Dropdown(id='country-dropdown', options=country_options, placeholder="Select a country..."),
                        html.Br(),
                        html.Label("Date"),
                        dcc.DatePickerSingle(
                            id='analysis-date-picker',
                            max_date_allowed=date.today(),
                            date=date.today() - timedelta(days=1)
                        ),
                        html.Div(id="map-status-message", className="mt-3 text-muted"),
                        html.Hr(),
                        html.H4("2. Select a Fire Point & Analyze"),
                        html.Div(id="selection-status", className="mt-2 text-muted", children="No point selected."),
                        
                        html.Label("Grid Graph Size", className="mt-3"),
                        dcc.Dropdown(
                            id='grid-graph-size-dropdown',
                            options=[
                                {'label': '3x3 Grid (Small)', 'value': 3},
                                {'label': '5x5 Grid (Medium)', 'value': 5},
                                {'label': '7x7 Grid (Large)', 'value': 7},
                                {'label': '9x9 Grid (Extra Large)', 'value': 9}
                            ],
                            value=7,
                            clearable=False
                        ),
                        html.Label("Grid Spacing", className="mt-3"),
                        dcc.Dropdown(
                            id='grid-spacing-dropdown',
                            options=[
                                {'label': '0.005° (Very Close)', 'value': 0.005},
                                {'label': '0.01° (Close)', 'value': 0.01},
                                {'label': '0.02° (Medium)', 'value': 0.02},
                                {'label': '0.05° (Far)', 'value': 0.05},
                                {'label': '0.1° (Very Far)', 'value': 0.1}
                            ],
                            value=0.005,
                            clearable=False
                        ),
                        html.Label("Map Zoom Level", className="mt-3"),
                        dcc.Dropdown(
                            id='map-zoom-dropdown',
                            options=[
                                {'label': 'Zoom 8 (Country View)', 'value': 8},
                                {'label': 'Zoom 10 (Region View)', 'value': 10},
                                {'label': 'Zoom 12 (City View)', 'value': 12},
                                {'label': 'Zoom 14 (District View)', 'value': 14},
                                {'label': 'Zoom 16 (Street View)', 'value': 16},
                                {'label': 'Zoom 18 (Building View)', 'value': 18}
                            ],
                            value=12,
                            clearable=False
                        ),
                        html.Label("D Value (Fire Spread Rate)", className="mt-3"),
                        dcc.Input(
                            id='d-value-input',
                            type='number',
                            value=3,
                            min=1,
                            max=10,
                            step=1,
                            placeholder="Enter D value (1-10)"
                        ),
                        html.Label("B Value (Budget)", className="mt-3"),
                        dcc.Input(
                            id='b-value-input',
                            type='number',
                            value=3,
                            min=1,
                            max=10,
                            step=1,
                            placeholder="Enter B value (1-10)"
                        ),
                        html.Label("Lambda (Firefighter Speed Ratio)", className="mt-3"),
                        dcc.Input(
                            id='lambda-d-input',
                            type='number',
                            value=1.0,
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            placeholder="Enter lambda (0.1-5.0)"
                        ),
                        html.Hr(),
                        html.H4("3. Map Options"),
                        dbc.Button(
                            "Toggle Grid Graph & Update Zoom", 
                            id="grid-toggle-button", 
                            color="secondary", 
                            className="mt-2 mb-3", 
                            n_clicks=0,
                            style={"width": "100%"}
                        ),
                        html.Hr(),
                        html.H4("4. Analysis & Actions"),
                        html.Div([
                            html.H5("📋 Workflow Steps:", className="mb-3 text-primary"),
                            html.Ol([
                                html.Li("Select a fire point on the map", className="mb-1"),
                                html.Li("Toggle the grid graph", className="mb-1"),
                                html.Li("Click on grid nodes to select firefighter stations", className="mb-1"),
                                html.Li("Save the graph data (required for MFF solver)", className="mb-1"),
                                html.Li("Run the MFF solver to optimize firefighter deployment", className="mb-1")
                            ], className="mb-3", style={'fontSize': '14px', 'fontWeight': '500'}),
                        ], className="mb-3 p-3 bg-primary bg-opacity-10 border border-primary rounded"),
                        dbc.Button(
                            "Analyze Selected Fire Point", 
                            id="analyze-button", 
                            color="primary", 
                            className="mt-2 mb-2", 
                            n_clicks=0, 
                            disabled=True,
                            style={"width": "100%"}
                        ),
                        dbc.Button(
                            "Save Graph Data", 
                            id="save-graph-button", 
                            color="success", 
                            className="mb-2", 
                            n_clicks=0, 
                            disabled=True,
                            style={"width": "100%"}
                        ),
                        dbc.Button(
                            "Run MFF Solver", 
                            id="run-mff-button", 
                            color="warning", 
                            className="mb-2", 
                            n_clicks=0, 
                            disabled=True,
                            style={"width": "100%"}
                        ),
                        html.Hr(),
                        html.H4("5. Load Examples"),
                        dbc.Button(
                            "Load Example 1", 
                            id="load-example-1-button", 
                            color="info", 
                            className="mb-2",
                            n_clicks=0,
                            style={"width": "100%"}
                        ),
                        dbc.Button(
                            "Load Example 2", 
                            id="load-example-2-button", 
                            color="secondary", 
                            className="mb-2", 
                            n_clicks=0,
                            style={"width": "100%"}
                        ),
                    ])),
                    md=5
                ),
                dbc.Col(
                    dcc.Loading(
                        id="loading-map",
                        type="default",
                        children=[
                            dcc.Graph(id="fire-map", style={"height": "80vh"}),
                            html.Div(id="results-output", className="mt-3")
                        ]
                    ),
                    md=7
                )
            ]),
            
            # Copyright footer
            dbc.Row([
                dbc.Col([
                    html.Hr(className="mt-5 mb-3"),
                    html.P(
                        "© 2025 Team 1 - SEA Quantathon: Ha-Ninh Nguyen, Natchapol Patamawisut, Supawit Marayat, Tan Chun Loong",
                        className="text-center mb-3",
                        style={'fontSize': '16px', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ],
        fluid=True
    )