# modules/callbacks.py
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import no_update
import geopandas

from .data_fetcher import get_fire_data, get_country_list
from .analysis_pipeline import run_analysis_pipeline
from .plotly_visuals import create_grid_heatmap

# --- Startup Loading ---
try:
    COUNTRY_DF = get_country_list().set_index('abreviation')
    WORLD_GDF = geopandas.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load country data at startup: {e}")
    COUNTRY_DF, WORLD_GDF = pd.DataFrame(), geopandas.GeoDataFrame()

def register_callbacks(app, api_key):
    """Registers all callbacks for the application."""

    # Callback 1: Update map with fire points
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Output('selection-status', 'children'),
        Output('analyze-button', 'disabled', allow_duplicate=True),
        Input('country-dropdown', 'value'),
        Input('analysis-date-picker', 'date'),
        prevent_initial_call=True
    )
    def update_fire_points_map(country_code, selected_date):
        if not country_code: return no_update, "No point selected.", True
        if WORLD_GDF.empty: return no_update, "Error: World map data failed to load.", True
        try:
            country_shape = WORLD_GDF[WORLD_GDF['id'] == country_code]
            country_info = COUNTRY_DF.loc[country_code]
            country_bbox = country_info['bbox_coords']
            fire_df = get_fire_data(api_key, country_bbox, start_date=selected_date)
            if fire_df.empty: return no_update, f"No fire data for {country_info['name']}.", True
            
            fire_gdf = geopandas.GeoDataFrame(fire_df, geometry=geopandas.points_from_xy(fire_df.longitude, fire_df.latitude), crs="EPSG:4326")
            fires_in_country = geopandas.sjoin(fire_gdf, country_shape, how="inner", predicate="intersects")
            if fires_in_country.empty: return no_update, f"No fires found within {country_info['name']}.", True
            
            coords = [float(c) for c in country_bbox.split(',')]
            center_lon, center_lat = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            hover_texts = ["Lat: {:.3f}, Lon: {:.3f}".format(lat, lon) for lat, lon in zip(fires_in_country['latitude'], fires_in_country['longitude'])]
            fig = go.Figure(go.Scattermapbox(lat=fires_in_country['latitude'], lon=fires_in_country['longitude'], mode='markers', marker={'size': 10, 'color': 'red'}, hoverinfo='text', text=hover_texts))
            fig.update_layout(mapbox_style="open-street-map", mapbox_center={'lat': center_lat, 'lon': center_lon}, mapbox_zoom=4, margin={"r":0,"t":0,"l":0,"b":0})
            
            return fig, "Select a fire point from the map.", True
        except Exception as e:
            return no_update, f"Error: {e}", True

    # Callback 2: Handle the selection of a single fire point
    @app.callback(
        Output('selection-status', 'children', allow_duplicate=True),
        Output('analyze-button', 'disabled', allow_duplicate=True),
        Output('selected-fire-point', 'data'),
        Input('fire-map', 'clickData'),
        prevent_initial_call=True
    )
    def select_fire_point(clickData):
        if clickData is None: return "No point selected.", True, no_update
        point = clickData['points'][0]
        lat, lon = point['lat'], point['lon']
        selected_point_data = {'lat': lat, 'lon': lon}
        status_message = "Selected point at Lat: {:.3f}, Lon: {:.3f}".format(lat, lon)
        return status_message, False, selected_point_data

    # Callback 3: Run the analysis when the button is clicked
    @app.callback(
        Output('results-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('selected-fire-point', 'data'),
        State('analysis-date-picker', 'date'),
        State('grid-size-dropdown', 'value'),
        prevent_initial_call=True
    )
    def run_analysis_and_display(n_clicks, selected_point, selected_date, grid_size):
        if not selected_point:
            return dbc.Alert("Error: No fire point selected for analysis.", color="danger")
        lat, lon = selected_point['lat'], selected_point['lon']
        try:
            results = run_analysis_pipeline(lat, lon, selected_date, api_key, grid_size=grid_size)
            stats = results['stats']
            stats_display = dbc.Alert([
                html.H5("Grid Statistics"),
                f"Grid Size: {stats['grid_size']}", html.Br(),
                f"Approx. Cell Size: {stats['cell_size_km']}", html.Br(),
                f"Total Fires in Area: {stats['total_fires']}", html.Br(),
                f"Cells with Fire: {stats['cells_with_fire']}",
            ], color="info", className="mt-3")

            fire_items = []
            if results['fire_grid'] is not None:
                fire_channels = results['fire_channels']
                fire_grid = results['fire_grid']
                for i, channel in enumerate(fire_channels):
                    scale = 'Reds' if channel == 'fire_count' else 'hot'
                    fig = create_grid_heatmap(fire_grid[:, :, i], f"Fire Grid: {channel}", colorscale=scale)
                    fire_items.append(dbc.AccordionItem(dcc.Graph(figure=fig), title=f"Fire: {channel}"))
            
            elevation_status = results.get('elevation_status', 'Success')
            if elevation_status != 'Success':
                elevation_content = dbc.Alert(elevation_status, color="warning")
            else:
                elevation_fig = create_grid_heatmap(results['elevation_grid'], "Elevation (m)")
                elevation_content = dcc.Graph(figure=elevation_fig)
            
            main_accordion = dbc.Accordion([
                dbc.AccordionItem(dbc.Accordion(fire_items), title="Fire Data Grids"),
                dbc.AccordionItem(elevation_content, title="Elevation Grid"),
            ], start_collapsed=False, always_open=True, className="mt-3")
            
            return [stats_display, main_accordion]
        except Exception as e:
            return dbc.Alert(f"Analysis failed: {e}", color="danger")