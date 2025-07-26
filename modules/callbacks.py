# modules/callbacks.py
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import no_update, callback_context
from datetime import datetime

# New imports for geographic filtering
import geopandas

from .data_fetcher import get_fire_data, get_country_list

BRIGHTNESS_COLS = {
    'MODIS_NRT': 'brightness',
    'VIIRS_SNPP_NRT': 'bright_ti4',
    'VIIRS_NOAA20_NRT': 'bright_ti4',
}

# --- Load datasets at startup for efficiency ---
try:
    # This dataframe has the bounding boxes and names for countries
    COUNTRY_DF = get_country_list().set_index('abreviation')
    
    # This geodataframe has the actual geographic shapes of all countries
    # We now load it directly from a reliable URL instead of geodatasets
    WORLD_GDF = geopandas.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json")
except Exception as e:
    print(f"Could not load country data at startup: {e}")
    COUNTRY_DF = pd.DataFrame()
    WORLD_GDF = geopandas.GeoDataFrame()

def register_callbacks(app, api_key):
    """Registers all callbacks for the application."""

    @app.callback(
        Output("fire-map", "figure"),
        Output("status-message", "children"),
        Input('country-dropdown', 'value'),
        Input('source-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
    )
    def update_map(country_code, source, start_date, end_date):
        ctx = callback_context
        if not ctx.triggered or not country_code:
            blank_fig = go.Figure(go.Scattermapbox())
            blank_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            return blank_fig, "Select a country to view fire data."

        # --- 1. Handle Date Range ---
        date_format = "%Y-%m-%d"
        d_start = datetime.strptime(start_date, date_format).date()
        d_end = datetime.strptime(end_date, date_format).date()
        day_range = (d_end - d_start).days
        
        if day_range > 10:
            return no_update, "Error: Date range cannot exceed 10 days."
        if day_range < 0:
            return no_update, "Error: End date must be on or after start date."

        # --- 2. Fetch Data using Bounding Box ---
        try:
            country_info = COUNTRY_DF.loc[country_code]
            country_bbox = country_info['bbox_coords']
            
            fire_df = get_fire_data(
                api_key=api_key.strip(),
                area_coords=country_bbox,
                source=source,
                start_date=start_date,
                day_range=day_range + 1 # Add 1 to make the range inclusive
            )
            if fire_df.empty:
                return no_update, f"No fire data found for {country_info['name']} in the selected date range."
        except Exception as e:
            return no_update, f"Error: {e}"

        # --- 3. Spatially Filter Data to Precise Country Shape ---
        fire_gdf = geopandas.GeoDataFrame(
            fire_df, 
            geometry=geopandas.points_from_xy(fire_df.longitude, fire_df.latitude),
            crs="EPSG:4326"
        )
        # The new GeoJSON file uses 'id' for the 3-letter country code
        country_shape = WORLD_GDF[WORLD_GDF['id'] == country_code]
        
        if country_shape.empty:
            return no_update, f"Could not find geographic shape for {country_code}."

        fires_in_country = geopandas.sjoin(fire_gdf, country_shape, how="inner", predicate="intersects")

        if fires_in_country.empty:
            return no_update, f"No fire data found within the precise borders of {country_info['name']}."

        # --- 4. Prepare Data for Plotting ---
        brightness_col = BRIGHTNESS_COLS.get(source, 'brightness')
        fires_in_country['temp_c'] = fires_in_country[brightness_col] - 273.15
        
        hover_text = [
            f"Temp: {row.temp_c:.1f}°C<br>FRP: {row.frp}" 
            for index, row in fires_in_country.iterrows()
        ]

        # --- 5. Create and return the figure ---
        coords = [float(c) for c in country_bbox.split(',')]
        center_lon, center_lat = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2

        fig = go.Figure(go.Scattermapbox(
            lat=fires_in_country['latitude'],
            lon=fires_in_country['longitude'],
            mode='markers',
            marker={'size': 10, 'color': 'red', 'opacity': 0.7},
            hoverinfo='text',
            text=hover_text
        ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={'lat': center_lat, 'lon': center_lon},
            mapbox_zoom=4,
            margin={"r":0, "t":0, "l":0, "b":0},
        )
        
        status_message = f"Displaying {len(fires_in_country)} fire hotspots in {country_info['name']}."
        return fig, status_message