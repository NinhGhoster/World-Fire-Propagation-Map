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

def create_fire_network_from_center(center_lat, center_lon, all_fire_lats, all_fire_lons, max_distance=0.1):
    """Creates a network graph starting from a center fire point."""
    try:
        center_lat = float(center_lat)
        center_lon = float(center_lon)
        all_fire_lats = list(all_fire_lats) if all_fire_lats is not None else []
        all_fire_lons = list(all_fire_lons) if all_fire_lons is not None else []
        
        if len(all_fire_lats) != len(all_fire_lons):
            return {'edges': []}
        
        edges = []
        for i in range(len(all_fire_lats)):
            try:
                fire_lat = float(all_fire_lats[i])
                fire_lon = float(all_fire_lons[i])
                distance = ((fire_lat - center_lat) ** 2 + (fire_lon - center_lon) ** 2) ** 0.5
                
                if distance <= max_distance and distance > 0:
                    edges.append({
                        'lat': [center_lat, fire_lat],
                        'lon': [center_lon, fire_lon]
                    })
            except (IndexError, TypeError, ValueError):
                continue
        
        return {'edges': edges}
        
    except Exception as e:
        return {'edges': []}

def create_fire_grid_graph(fire_lats, fire_lons, max_distance=0.1):
    """Creates a grid graph using fire points as nodes."""
    try:
        fire_lats = list(fire_lats) if fire_lats is not None else []
        fire_lons = list(fire_lons) if fire_lons is not None else []
        
        if len(fire_lats) != len(fire_lons) or len(fire_lats) < 2:
            return {'edges': []}
        
        edges = []
        for i in range(len(fire_lats)):
            for j in range(i + 1, len(fire_lats)):
                try:
                    lat1, lon1 = float(fire_lats[i]), float(fire_lons[i])
                    lat2, lon2 = float(fire_lats[j]), float(fire_lons[j])
                    distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
                    
                    if distance <= max_distance:
                        edges.append({
                            'lat': [lat1, lat2],
                            'lon': [lon1, lon2]
                        })
                except (IndexError, TypeError, ValueError):
                    continue
        
        return {'edges': edges}
    except Exception as e:
        return {'edges': []}

def create_grid_graph(bounds, grid_size=8):
    """Creates a grid graph overlay for the map."""
    west, south, east, north = bounds
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    
    node_lats, node_lons, node_texts = [], [], []
    
    for row in range(grid_size):
        for col in range(grid_size):
            lat = south + row * lat_step + lat_step / 2
            lon = west + col * lon_step + lon_step / 2
            node_id = row * grid_size + col
            node_lats.append(lat)
            node_lons.append(lon)
            node_texts.append(f"Node {node_id}")
    
    edges = []
    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            if col < grid_size - 1:
                edges.append({
                    'lat': [node_lats[node], node_lats[node + 1]],
                    'lon': [node_lons[node], node_lons[node + 1]]
                })
            if row < grid_size - 1:
                edges.append({
                    'lat': [node_lats[node], node_lats[node + grid_size]],
                    'lon': [node_lons[node], node_lons[node + grid_size]]
                })
    
    return {'lats': node_lats, 'lons': node_lons, 'texts': node_texts, 'edges': edges}

def register_callbacks(app, api_key):
    """Register all Dash callbacks."""
    
    # Initial fire map
    @app.callback(
        Output('fire-map', 'figure'),
        Output('map-status-message', 'children'),
        Input('country-dropdown', 'value'),
        Input('analysis-date-picker', 'date'),
        prevent_initial_call=True
    )
    def update_fire_map(country, date):
        if not country:
            return go.Figure(), "Select a country to view fires."
        
        try:
            country_df = get_country_list()
            bbox = country_df[country_df['abreviation'] == country]['bbox_coords'].values[0]
            west, south, east, north = map(float, bbox.split(','))
            
            boundary_str = f"{west},{south},{east},{north}"
            df = get_fire_data(api_key, boundary_str, start_date=date)
            
            fig = go.Figure()
            
            if not df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    text=df.apply(lambda r: f"Lat: {r['latitude']:.2f}, Lon: {r['longitude']:.2f}<br>Brightness: {r.get('brightness', 'N/A')}K", axis=1),
                    name='Active Fires'
                ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=(south + north) / 2, lon=(west + east) / 2),
                    zoom=4
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=600
            )
            
            status = f"Showing {len(df)} fires" if not df.empty else "No fires detected in this area"
            return fig, status
            
        except Exception as e:
            return go.Figure(), f"Error: {str(e)}"
    
    # Toggle grid graph
    @app.callback(
        Output('grid-toggle-state', 'data'),
        Output('grid-toggle-button', 'children'),
        Output('fire-map', 'figure', allow_duplicate=True),
        Output('selected-firefighter-stations', 'data', allow_duplicate=True),
        Output('latest-saved-file', 'data', allow_duplicate=True),
        Output('save-graph-button', 'disabled', allow_duplicate=True),
        Output('run-mff-button', 'disabled', allow_duplicate=True),
        Input('grid-toggle-button', 'n_clicks'),
        State('grid-toggle-state', 'data'),
        State('fire-map', 'figure'),
        State('country-dropdown', 'value'),
        State('selected-fire-point', 'data'),
        State('grid-graph-size-dropdown', 'value'),
        State('grid-spacing-dropdown', 'value'),
        State('map-zoom-dropdown', 'value'),
        State('selected-firefighter-stations', 'data'),
        prevent_initial_call=True
    )
    def toggle_grid_graph(n_clicks, grid_on, fig, country, selected_point, grid_size, spacing, zoom, stations):
        if not n_clicks or n_clicks % 2 == 0:
            new_grid_on = True
            button_text = "Hide Grid Graph"
        else:
            new_grid_on = False
            button_text = "Show Grid Graph"
        
        if new_grid_on and selected_point and country:
            try:
                country_df = get_country_list()
                bbox = country_df[country_df['abreviation'] == country]['bbox_coords'].values[0]
                west, south, east, north = map(float, bbox.split(','))
                
                grid_data = create_grid_graph((west, south, east, north), grid_size)
                
                fig = go.Figure(fig)
                
                # Add grid nodes
                fig.add_trace(go.Scattermapbox(
                    lat=grid_data['lats'],
                    lon=grid_data['lons'],
                    mode='markers+text',
                    marker=dict(size=12, color='blue'),
                    text=grid_data['texts'],
                    name='Grid Nodes',
                    hoverinfo='text'
                ))
                
                # Add grid edges
                for edge in grid_data['edges']:
                    fig.add_trace(go.Scattermapbox(
                        lat=edge['lat'],
                        lon=edge['lon'],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        name='Grid Lines',
                        showlegend=False
                    ))
                
                # Re-add fire points
                country_df = get_country_list()
                bbox = country_df[country_df['abreviation'] == country]['bbox_coords'].values[0]
                west, south, east, north = map(float, bbox.split(','))
                boundary_str = f"{west},{south},{east},{north}"
                df = get_fire_data(api_key, boundary_str)
                
                if not df.empty:
                    fig.add_trace(go.Scattermapbox(
                        lat=df['latitude'],
                        lon=df['longitude'],
                        mode='markers',
                        marker=dict(size=8, color='red'),
                        name='Active Fires'
                    ))
                
                fig.update_layout(
                    mapbox=dict(center=dict(lat=(south + north) / 2, lon=(west + east) / 2), zoom=zoom)
                )
                
                return new_grid_on, button_text, fig, [], None, True, True
                
            except Exception as e:
                return grid_on, button_text, fig, stations, None, True, True
        
        return new_grid_on, button_text, fig, stations, None, True, True
    
    # Handle fire point selection
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Output('selection-status', 'children', allow_duplicate=True),
        Output('analyze-button', 'disabled', allow_duplicate=True),
        Output('save-graph-button', 'disabled', allow_duplicate=True),
        Output('selected-fire-point', 'data'),
        Output('selected-firefighter-stations', 'data'),
        Input('fire-map', 'clickData'),
        State('fire-map', 'figure'),
        State('selected-firefighter-stations', 'data'),
        State('grid-spacing-dropdown', 'value'),
        State('map-zoom-dropdown', 'value'),
        prevent_initial_call=True
    )
    def handle_fire_point_click(clickData, fig, stations, spacing, zoom):
        if clickData is None:
            return no_update, "No point selected.", True, True, None, stations
        
        point = clickData['points'][0]
        lat, lon = point['lat'], point['lon']
        
        status = f"Selected: ({lat:.4f}, {lon:.4f})"
        
        return no_update, status, False, False, {'lat': lat, 'lon': lon}, []
    
    # Enable buttons when data ready
    @app.callback(
        Output('save-graph-button', 'disabled'),
        Output('run-mff-button', 'disabled'),
        Input('selected-fire-point', 'data'),
        Input('selected-firefighter-stations', 'data'),
        Input('latest-saved-file', 'data'),
        Input('grid-toggle-state', 'data'),
    )
    def update_button_states(selected_point, stations, saved_file, grid_on):
        if selected_point and grid_on:
            return False, False
        return True, True
    
    # Update map with firefighter stations
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Input('selected-firefighter-stations', 'data'),
        State('fire-map', 'figure'),
        State('grid-toggle-state', 'data'),
        prevent_initial_call=True
    )
    def update_map_with_stations(stations, fig, grid_on):
        if not grid_on or not stations or not fig:
            return no_update
        
        fig = go.Figure(fig)
        
        # Add firefighter stations
        station_lats, station_lons = zip(*stations) if stations else ([], [])
        
        fig.add_trace(go.Scattermapbox(
            lat=station_lats,
            lon=station_lons,
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Firefighter Stations'
        ))
        
        return fig
    
    # Run analysis
    @app.callback(
        Output('results-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('selected-fire-point', 'data'),
        State('analysis-date-picker', 'date'),
        prevent_initial_call=True
    )
    def run_analysis(n_clicks, selected_point, selected_date):
        if not selected_point:
            return dbc.Alert("⚠️ Please select a fire point on the map first.", color="warning")
        
        lat, lon = selected_point['lat'], selected_point['lon']
        
        try:
            results = run_analysis_pipeline(lat, lon, selected_date, api_key, grid_size=64)
            stats = results['stats']
            
            # Check if there's fire data
            if stats['total_fires'] == 0:
                return dbc.Alert([
                    html.H5("📭 No Fire Data Found"),
                    html.P(f"No active fires detected near ({lat:.2f}, {lon:.2f})"),
                    html.Hr(),
                    html.P("Try selecting a different point on the map, or choose a region with known fire activity."),
                    html.P("💡 Tip: Australia currently has active fires. Select Australia first, then click on a fire point."),
                ], color="info", className="mt-3")
            
            stats_display = dbc.Card([
                dbc.CardHeader("Grid Statistics"),
                dbc.CardBody([
                    html.P(f"Grid Size: {stats['grid_size']}"),
                    html.P(f"Cell Size: {stats['cell_size_km']}"),
                    html.P(f"Total Fires: {stats['total_fires']}"),
                    html.P(f"Cells with Fire: {stats['cells_with_fire']}"),
                ])
            ], className="mb-3")
            
            return stats_display
            
        except Exception as e:
            return dbc.Alert(f"Analysis failed: {e}", color="danger")
    
    # Save graph data
    @app.callback(
        Output('save-graph-button', 'disabled'),
        Output('selection-status', 'children', allow_duplicate=True),
        Output('latest-saved-file', 'data'),
        Output('run-mff-button', 'disabled'),
        Input('save-graph-button', 'n_clicks'),
        State('fire-map', 'figure'),
        State('selected-fire-point', 'data'),
        prevent_initial_call=True
    )
    def save_graph_data(n_clicks, fig, selected_point):
        if not n_clicks or not selected_point:
            return True, "No point selected.", None, True
        
        return False, f"Saved: {selected_point}", "saved", False
    
    # Run MFF solver
    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Input('run-mff-button', 'n_clicks'),
        State('selected-firefighter-stations', 'data'),
        prevent_initial_call=True
    )
    def run_mff_solver(n_clicks, stations):
        if not n_clicks:
            return no_update
        
        if not stations or len(stations) < 2:
            return dbc.Alert("⚠️ Select at least 2 firefighter stations on the grid.", color="warning")
        
        return dbc.Alert([
            html.H5("🧯 MFF Optimization"),
            html.P(f"Optimizing firefighter deployment for {len(stations)} stations..."),
            html.P("This would invoke the SCIP/MIQCP solver here."),
        ], color="success")
    
    # Load example 1
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Output('selection-status', 'children', allow_duplicate=True),
        Output('selected-fire-point', 'data'),
        Output('selected-firefighter-stations', 'data'),
        Input('load-example-1-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def load_example_1(n_clicks):
        if not n_clicks:
            return no_update
        
        # Australia example
        lat, lon = -25.0, 133.0
        
        try:
            country_df = get_country_list()
            bbox = country_df[country_df['abreviation'] == 'AU']['bbox_coords'].values[0]
            west, south, east, north = map(float, bbox.split(','))
            
            fig = go.Figure()
            
            df = get_fire_data(api_key, f"{west},{south},{east},{north}")
            
            if not df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Active Fires'
                ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=-25, lon=133), zoom=4),
                margin=dict(l=0, r=0, t=0, b=0),
                height=600
            )
            
            return fig, f"Example: Australia ({lat}, {lon})", {'lat': lat, 'lon': lon}, []
            
        except Exception as e:
            return no_update, f"Error: {e}", None, []
    
    # Load example 2
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Output('selection-status', 'children', allow_duplicate=True),
        Output('selected-fire-point', 'data'),
        Output('selected-firefighter-stations', 'data'),
        Input('load-example-2-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def load_example_2(n_clicks):
        if not n_clicks:
            return no_update
        
        # USA example
        lat, lon = 36.0, -119.0
        
        try:
            fig = go.Figure()
            
            df = get_fire_data(api_key, "-125,30,-100,50")
            
            if not df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Active Fires'
                ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=36, lon=-119), zoom=5),
                margin=dict(l=0, r=0, t=0, b=0),
                height=600
            )
            
            return fig, f"Example: USA ({lat}, {lon})", {'lat': lat, 'lon': lon}, []
            
        except Exception as e:
            return no_update, f"Error: {e}", None, []
