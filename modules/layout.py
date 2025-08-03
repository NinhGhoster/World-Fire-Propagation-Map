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
            dcc.Store(id='selected-fire-point'),
            dcc.Store(id='grid-toggle-state', data=False),
            dcc.Store(id='selected-firefighter-stations', data=[]),
            dcc.Store(id='latest-saved-file', data=None),
            dcc.Store(id='mff-solution-data', data=None),
            html.H1("QPreFire - Wildfire Response and Optimization", className="my-4 text-center"),
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
                        # Grid Resolution fixed to 64x64
                        html.Label("Grid Resolution", className="mt-3"),
                        html.Div("64x64 (Fast Resolution)", className="text-muted mb-2"),
                        html.Label("Grid Graph Size", className="mt-3"),
                        dcc.Dropdown(
                            id='grid-graph-size-dropdown',
                            options=[
                                {'label': '3x3 Grid (Small)', 'value': 3},
                                {'label': '5x5 Grid (Medium)', 'value': 5},
                                {'label': '7x7 Grid (Large)', 'value': 7},
                                {'label': '9x9 Grid (Extra Large)', 'value': 9}
                            ],
                            value=3,
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
                            value=0.05,
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
                        dbc.Button("Analyze Selected Fire Point", id="analyze-button", color="primary", className="mt-3", n_clicks=0, disabled=True),
                                                     dbc.Button("Save Graph Data", id="save-graph-button", color="success", className="mt-2", n_clicks=0, disabled=True),
                             dbc.Button("Run MFF Solver", id="run-mff-button", color="warning", className="mt-2", n_clicks=0, disabled=True),
                             dbc.Button("Load Example 1", id="load-example-1-button", color="info", className="mt-2", n_clicks=0),
                             dbc.Button("Load Example 2", id="load-example-2-button", color="secondary", className="mt-2", n_clicks=0),
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
            ])
        ],
        fluid=True
    )