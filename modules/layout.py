# modules/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html
from .data_fetcher import get_country_list
from datetime import date, timedelta

def create_layout(app):
    """Creates the layout of the Dash application."""
    
    try:
        country_df = get_country_list()
        # The dropdown value is now the country's 3-letter code (abreviation)
        country_options = [
            {'label': row['name'], 'value': row['abreviation']} 
            for index, row in country_df.iterrows()
        ]
    except Exception as e:
        print(f"ERROR: Could not fetch country list on startup: {e}")
        country_options = []

    source_options = [
        {'label': 'MODIS NRT', 'value': 'MODIS_NRT'},
        {'label': 'VIIRS S-NPP NRT', 'value': 'VIIRS_SNPP_NRT'},
        {'label': 'VIIRS NOAA-20 NRT', 'value': 'VIIRS_NOAA20_NRT'},
    ]

    return dbc.Container(
        [
            html.H1("World Fire Propagation Map 🔥", className="my-4 text-center"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Controls", className="card-title"),
                                html.Label(["Select Country ", html.I(className="bi bi-info-circle-fill", id="info-country")]),
                                dcc.Dropdown(id='country-dropdown', options=country_options, placeholder="Search for a country..."),
                                dbc.Tooltip("Select a country to view fire data within its borders.", target="info-country"),
                                html.Br(),

                                html.Label(["Select Data Source ", html.I(className="bi bi-info-circle-fill", id="info-source")]),
                                dcc.Dropdown(id='source-dropdown', options=source_options, value='MODIS_NRT', clearable=False),
                                dbc.Tooltip("Choose the satellite data source.", target="info-source"),
                                html.Br(),

                                # Replaced single date picker with a date range picker
                                html.Label(["Select Date Range ", html.I(className="bi bi-info-circle-fill", id="info-date")]),
                                dcc.DatePickerRange(
                                    id='date-picker-range',
                                    min_date_allowed=date(2000, 1, 1),
                                    max_date_allowed=date.today(),
                                    start_date=date.today() - timedelta(days=1),
                                    end_date=date.today()
                                ),
                                dbc.Tooltip("Select a start and end date. The range cannot exceed 10 days.", target="info-date"),
                                html.Div(id="status-message", className="mt-4 text-muted"),
                            ])
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-map",
                            type="default",
                            children=dcc.Graph(id="fire-map", style={"height": "80vh"})
                        ),
                        md=9,
                    ),
                ]
            ),
        ],
        fluid=True,
    )