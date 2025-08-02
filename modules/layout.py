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
            html.H1("Wildfire Data Analysis Pipeline", className="my-4 text-center"),
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
                        html.Label("Grid Resolution", className="mt-3"),
                        dcc.Dropdown(
                            id='grid-size-dropdown',
                            options=[
                                {'label': '64x64 (Fast)', 'value': 64},
                                {'label': '128x128 (Finer)', 'value': 128},
                                {'label': '256x256 (Highest)', 'value': 256}
                            ],
                            value=256,
                            clearable=False
                        ),
                        dbc.Button("Analyze Selected Fire Point", id="analyze-button", color="primary", className="mt-3", n_clicks=0, disabled=True),
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