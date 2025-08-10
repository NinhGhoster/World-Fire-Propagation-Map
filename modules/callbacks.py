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
    """
    Creates a network graph starting from a center fire point and connecting to nearby fires.
    
    Args:
        center_lat: Latitude of the center fire point
        center_lon: Longitude of the center fire point
        all_fire_lats: List of all fire point latitudes
        all_fire_lons: List of all fire point longitudes
        max_distance: Maximum distance (in degrees) to connect fires
    
    Returns:
        Dictionary containing edge data for connecting the center to nearby fire points
    """
    try:
        # Convert center coordinates to float
        center_lat = float(center_lat)
        center_lon = float(center_lon)
        
        # Ensure all fire coordinates are lists
        all_fire_lats = list(all_fire_lats) if all_fire_lats is not None else []
        all_fire_lons = list(all_fire_lons) if all_fire_lons is not None else []
        
        if len(all_fire_lats) != len(all_fire_lons):
            print(f"Warning: all_fire_lats ({len(all_fire_lats)}) and all_fire_lons ({len(all_fire_lons)}) have different lengths")
            return {'edges': []}
        
        edges = []
        connected_points = []
        
        # Find all fire points within max_distance of the center
        print(f"Checking {len(all_fire_lats)} fire points against center ({center_lat:.4f}, {center_lon:.4f})")
        print(f"Max distance: {max_distance}")
        
        for i in range(len(all_fire_lats)):
            try:
                fire_lat = float(all_fire_lats[i])
                fire_lon = float(all_fire_lons[i])
                
                # Calculate distance from center
                distance = ((fire_lat - center_lat) ** 2 + (fire_lon - center_lon) ** 2) ** 0.5
                
                print(f"Fire point {i}: ({fire_lat:.4f}, {fire_lon:.4f}), distance: {distance:.4f}")
                
                # Connect fires that are within max_distance of the center
                if distance <= max_distance and distance > 0:  # Don't connect to self
                    edges.append({
                        'lat': [center_lat, fire_lat],
                        'lon': [center_lon, fire_lon]
                    })
                    connected_points.append((fire_lat, fire_lon))
                    print(f"  -> Connected!")
                    
            except (IndexError, TypeError, ValueError) as e:
                print(f"Error processing fire point {i}: {e}")
                continue
        
        print(f"Connected {len(connected_points)} fire points to center at ({center_lat:.4f}, {center_lon:.4f})")
        return {'edges': edges, 'connected_points': connected_points}
        
    except Exception as e:
        print(f"Error in create_fire_network_from_center: {e}")
        return {'edges': []}

def create_fire_grid_graph(fire_lats, fire_lons, max_distance=0.1):
    """
    Creates a grid graph using fire points as nodes and connecting nearby fires.
    
    Args:
        fire_lats: List of fire point latitudes
        fire_lons: List of fire point longitudes
        max_distance: Maximum distance (in degrees) to connect fires
    
    Returns:
        Dictionary containing edge data for connecting nearby fire points
    """
    # Ensure inputs are lists and have the same length
    try:
        fire_lats = list(fire_lats) if fire_lats is not None else []
        fire_lons = list(fire_lons) if fire_lons is not None else []
        
        if len(fire_lats) != len(fire_lons):
            print(f"Warning: fire_lats ({len(fire_lats)}) and fire_lons ({len(fire_lons)}) have different lengths")
            return {'edges': []}
        
        if len(fire_lats) < 2:
            return {'edges': []}
        
        edges = []
        
        # Calculate distances between all fire points and connect nearby ones
        for i in range(len(fire_lats)):
            for j in range(i + 1, len(fire_lats)):
                try:
                    # Calculate distance between two fire points
                    lat1, lon1 = float(fire_lats[i]), float(fire_lons[i])
                    lat2, lon2 = float(fire_lats[j]), float(fire_lons[j])
                    
                    # Simple distance calculation (approximate)
                    distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
                    
                    # Connect fires that are within max_distance
                    if distance <= max_distance:
                        edges.append({
                            'lat': [lat1, lat2],
                            'lon': [lon1, lon2]
                        })
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error processing fire points {i} and {j}: {e}")
                    continue
        
        return {'edges': edges}
    except Exception as e:
        print(f"Error in create_fire_grid_graph: {e}")
        return {'edges': []}

def create_grid_graph(bounds, grid_size=8):
    """
    Creates a grid graph overlay for the map.
    
    Args:
        bounds: List of [west, south, east, north] coordinates
        grid_size: Number of grid cells per side (default 8 for 8x8 grid)
    
    Returns:
        Dictionary containing node and edge data for the grid graph
    """
    west, south, east, north = bounds
    
    # Calculate grid spacing
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    
    # Generate grid nodes
    node_lats = []
    node_lons = []
    node_texts = []
    
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            lat = south + i * lat_step
            lon = west + j * lon_step
            node_lats.append(lat)
            node_lons.append(lon)
            node_texts.append(f"Grid Node ({i},{j})<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}")
    
    # Generate grid edges (connections between adjacent nodes)
    edges = []
    
    # Horizontal edges
    for i in range(grid_size + 1):
        for j in range(grid_size):
            start_lat = south + i * lat_step
            start_lon = west + j * lon_step
            end_lat = south + i * lat_step
            end_lon = west + (j + 1) * lon_step
            
            edges.append({
                'lat': [start_lat, end_lat],
                'lon': [start_lon, end_lon]
            })
    
    # Vertical edges
    for i in range(grid_size):
        for j in range(grid_size + 1):
            start_lat = south + i * lat_step
            start_lon = west + j * lon_step
            end_lat = south + (i + 1) * lat_step
            end_lon = west + j * lon_step
            
            edges.append({
                'lat': [start_lat, end_lat],
                'lon': [start_lon, end_lon]
            })
    
    return {
        'node_lats': node_lats,
        'node_lons': node_lons,
        'node_texts': node_texts,
        'edges': edges
    }

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
            
            # Create base figure with fire points
            fig = go.Figure(go.Scattermapbox(
                lat=fires_in_country['latitude'], 
                lon=fires_in_country['longitude'], 
                mode='markers', 
                marker={'size': 10, 'color': 'red'}, 
                hoverinfo='text', 
                text=hover_texts,
                name='Fire Points'
            ))
            
            fig.update_layout(
                mapbox_style="open-street-map", 
                mapbox_center={'lat': center_lat, 'lon': center_lon}, 
                mapbox_zoom=4, 
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            
            return fig, "Select a fire point from the map.", True
        except Exception as e:
            return no_update, f"Error: {e}", True

    # Callback 1.5: Handle grid toggle button
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
    def toggle_grid_graph(n_clicks, current_state, current_figure, country_code, selected_point, grid_graph_size, grid_spacing, map_zoom, selected_stations):
        if n_clicks is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        new_state = not current_state
        button_text = "Hide Grid Graph" if new_state else "Show Grid Graph"
        
        # If no current figure or no country selected, just update the toggle state
        if not current_figure or not country_code:
            # Reset everything when grid is hidden
            if not new_state:
                return new_state, button_text, no_update, [], None, True, True
            else:
                return new_state, button_text, no_update, no_update, no_update, no_update, no_update
        
        # Get the current map data and layout
        existing_data = current_figure.get('data', [])
        current_layout = current_figure.get('layout', {})
        
        # Filter out existing grid lines and firefighter stations to avoid duplication
        filtered_data = []
        for trace in existing_data:
            # Keep only non-grid traces (fire points, etc.)
            if not (trace.get('name', '').startswith('Grid Line') or 
                   trace.get('name', '').startswith('Firefighter Station') or
                   trace.get('name') in ['Grid Nodes', 'Local Grid Nodes'] or
                   trace.get('name') == 'Fire Node'):
                filtered_data.append(trace)
        
        # Create new figure with filtered data (no grid lines)
        fig = go.Figure(data=filtered_data)
        
        # Preserve current map center and zoom
        if 'mapbox' in current_layout:
            fig.update_layout(
                mapbox_style=current_layout['mapbox'].get('style', 'open-street-map'),
                mapbox_center=current_layout['mapbox'].get('center', {'lat': 0, 'lon': 0}),
                mapbox_zoom=current_layout['mapbox'].get('zoom', 4),
                margin=current_layout.get('margin', {"r":0,"t":0,"l":0,"b":0})
            )
        
        # Add or remove grid based on new state
        if new_state:
            # Extract fire points from existing data
            fire_lats = []
            fire_lons = []
            
            print(f"Number of traces: {len(existing_data)}")
            for i, trace in enumerate(existing_data):
                print(f"Trace {i}: name='{trace.get('name')}', type={type(trace.get('lat'))}")
                if trace.get('name') == 'Fire Points':
                    # Get the actual coordinate data, not metadata
                    lat_data = trace.get('lat', {})
                    lon_data = trace.get('lon', {})
                    
                    print(f"Raw lat_data type: {type(lat_data)}, keys: {list(lat_data.keys()) if isinstance(lat_data, dict) else 'N/A'}")
                    print(f"Raw lon_data type: {type(lon_data)}, keys: {list(lon_data.keys()) if isinstance(lon_data, dict) else 'N/A'}")
                    
                    # Extract actual coordinate data from the dictionary structure
                    if isinstance(lat_data, dict) and 'bdata' in lat_data:
                        fire_lats = lat_data['bdata']
                        print(f"Extracted lat_data from bdata: {len(fire_lats)} items")
                    else:
                        fire_lats = []
                        print("No bdata found in lat_data")
                    
                    if isinstance(lon_data, dict) and 'bdata' in lon_data:
                        fire_lons = lon_data['bdata']
                        print(f"Extracted lon_data from bdata: {len(fire_lons)} items")
                    else:
                        fire_lons = []
                        print("No bdata found in lon_data")
                    
                    # Handle base64 encoded data or numpy arrays
                    import base64
                    import numpy as np
                    
                    def decode_coordinates(data):
                        if isinstance(data, str):
                            # Try to decode base64 string
                            try:
                                decoded_bytes = base64.b64decode(data)
                                # Try to decode as numpy array
                                try:
                                    return np.frombuffer(decoded_bytes, dtype=np.float64).tolist()
                                except:
                                    # If that fails, try as float32
                                    return np.frombuffer(decoded_bytes, dtype=np.float32).tolist()
                            except:
                                print(f"Failed to decode base64 data: {data[:50]}...")
                                return []
                        elif hasattr(data, 'tolist'):
                            return data.tolist()
                        elif hasattr(data, '__iter__') and not isinstance(data, str):
                            return list(data)
                        else:
                            return [data] if data is not None else []
                    
                    fire_lats = decode_coordinates(fire_lats)
                    fire_lons = decode_coordinates(fire_lons)
                    
                    print(f"Before filtering - fire_lats: {fire_lats[:5]}...")  # Show first 5 items
                    print(f"Before filtering - fire_lons: {fire_lons[:5]}...")  # Show first 5 items
                    
                    # Filter out non-numeric values
                    fire_lats = [lat for lat in fire_lats if isinstance(lat, (int, float)) or (isinstance(lat, str) and lat.replace('.', '').replace('-', '').isdigit())]
                    fire_lons = [lon for lon in fire_lons if isinstance(lon, (int, float)) or (isinstance(lon, str) and lon.replace('.', '').replace('-', '').isdigit())]
                    
                    print(f"After filtering - fire_lats: {fire_lats[:5]}...")  # Show first 5 items
                    print(f"After filtering - fire_lons: {fire_lons[:5]}...")  # Show first 5 items
                    print(f"Found {len(fire_lats)} fire points")
                    break
            
            if fire_lats and fire_lons:
                # Check if we have a selected fire point to use as center
                if selected_point and 'lat' in selected_point and 'lon' in selected_point:
                    center_lat = selected_point['lat']
                    center_lon = selected_point['lon']
                    
                    # Check if we're zoomed in (local view)
                    current_zoom = current_layout.get('mapbox', {}).get('zoom', 4)
                    
                    if current_zoom > 8:  # Zoomed in view - use smaller distance
                        max_distance = 0.05
                        line_color = 'lightgreen'
                    else:  # Country view - use larger distance
                        max_distance = 0.1
                        line_color = 'gray'
                    
                    # Create a grid around the selected fire point
                    grid_spacing = grid_spacing if grid_spacing else 0.05  # Use selected grid spacing
                    grid_size = grid_graph_size if grid_graph_size else 3  # Use selected grid size
                    
                    # Use the exact grid size selected by the user
                    dynamic_grid_size = grid_size  # Respect user's selection
                    
                    # Calculate grid bounds
                    half_grid = (dynamic_grid_size - 1) / 2
                    grid_lats = []
                    grid_lons = []
                    
                    # Generate grid positions
                    for i in range(dynamic_grid_size):
                        for j in range(dynamic_grid_size):
                            grid_lat = center_lat + (i - half_grid) * grid_spacing
                            grid_lon = center_lon + (j - half_grid) * grid_spacing
                            grid_lats.append(grid_lat)
                            grid_lons.append(grid_lon)
                    
                    # Add grid nodes (unburnt areas) - make them light green
                    for i, (grid_lat, grid_lon) in enumerate(zip(grid_lats, grid_lons)):
                        # Skip the center position (we already have the fire node)
                        if abs(grid_lat - center_lat) < 0.001 and abs(grid_lon - center_lon) < 0.001:
                            continue
                        
                        # Check if this node is selected as a firefighter station
                        is_selected_station = any(
                            abs(s['lat'] - grid_lat) < 0.001 and abs(s['lon'] - grid_lon) < 0.001 
                            for s in selected_stations
                        ) if selected_stations else False
                        
                        # Choose color based on selection status
                        node_color = 'blue' if is_selected_station else 'lightgreen'
                        node_size = 15 if is_selected_station else 12
                        node_text = f"Firefighter Station<br>Lat: {grid_lat:.4f}<br>Lon: {grid_lon:.4f}" if is_selected_station else f"Unburnt Node<br>Lat: {grid_lat:.4f}<br>Lon: {grid_lon:.4f}"
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=[grid_lat],
                            lon=[grid_lon],
                            mode='markers',
                            marker={'size': node_size, 'color': node_color, 'symbol': 'circle'},
                            hoverinfo='text',
                            text=[node_text],
                            name='Grid Node',
                            showlegend=False
                        ))
                    
                    # Add grid connections (lines between adjacent grid nodes) - make them light green
                    for i in range(dynamic_grid_size):
                        for j in range(dynamic_grid_size):
                            current_lat = center_lat + (i - half_grid) * grid_spacing
                            current_lon = center_lon + (j - half_grid) * grid_spacing
                            
                            # Connect to right neighbor
                            if j < dynamic_grid_size - 1:
                                right_lat = center_lat + (i - half_grid) * grid_spacing
                                right_lon = center_lon + (j + 1 - half_grid) * grid_spacing
                                fig.add_trace(go.Scattermapbox(
                                    lat=[current_lat, right_lat],
                                    lon=[current_lon, right_lon],
                                    mode='lines',
                                    line={'color': 'lightgreen', 'width': 2},
                                    hoverinfo='skip',
                                    showlegend=False
                                ))
                            
                            # Connect to bottom neighbor
                            if i < dynamic_grid_size - 1:
                                bottom_lat = center_lat + (i + 1 - half_grid) * grid_spacing
                                bottom_lon = center_lon + (j - half_grid) * grid_spacing
                                fig.add_trace(go.Scattermapbox(
                                    lat=[current_lat, bottom_lat],
                                    lon=[current_lon, bottom_lon],
                                    mode='lines',
                                    line={'color': 'lightgreen', 'width': 2},
                                    hoverinfo='skip',
                                    showlegend=False
                                ))
                    
                    # Add fire node (selected point) - keep it red (added last so it appears on top)
                    fig.add_trace(go.Scattermapbox(
                        lat=[center_lat],
                        lon=[center_lon],
                        mode='markers',
                        marker={'size': 15, 'color': 'red', 'symbol': 'circle'},
                        hoverinfo='text',
                        text=[f"Fire Node<br>Lat: {center_lat:.4f}<br>Lon: {center_lon:.4f}"],
                        name='Fire Node',
                        showlegend=False
                    ))
                    
                    # Use manual zoom level from dropdown
                    zoom_level = map_zoom if map_zoom else 12  # Default to 12 if not set
                    
                    # Update map zoom and center
                    fig.update_layout(
                        mapbox_center={'lat': center_lat, 'lon': center_lon},
                        mapbox_zoom=zoom_level
                    )
                    
                    print(f"Created {dynamic_grid_size}x{dynamic_grid_size} grid around center point ({center_lat:.4f}, {center_lon:.4f})")
                    print(f"Grid spacing: {grid_spacing}°, Dynamic size: {dynamic_grid_size}x{dynamic_grid_size}")
                    print(f"Manual zoom level: {zoom_level}")
                    
                    print(f"Created grid graph from selected point ({center_lat:.4f}, {center_lon:.4f})")
                else:
                    print("No fire point selected. Please click on a fire point first, then toggle the grid.")
            else:
                print("No fire points found to create grid graph")
        else:
            # Grid is already filtered out above, so just return the current figure without grid
            pass
            
            # Preserve current map center and zoom
            if 'mapbox' in current_layout:
                fig.update_layout(
                    mapbox_style=current_layout['mapbox'].get('style', 'open-street-map'),
                    mapbox_center=current_layout['mapbox'].get('center', {'lat': 0, 'lon': 0}),
                    mapbox_zoom=current_layout['mapbox'].get('zoom', 4),
                    margin=current_layout.get('margin', {"r":0,"t":0,"l":0,"b":0})
                )
        
        # Reset everything when grid is hidden
        if not new_state:
            return new_state, button_text, fig, [], None, True, True
        else:
            return new_state, button_text, fig, no_update, no_update, no_update, no_update

    # Callback 2: Handle the selection of a single fire point
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
    def select_fire_point(clickData, current_figure, selected_stations, grid_spacing, map_zoom):
        if clickData is None: 
            return no_update, "No point selected.", True, True, no_update, no_update
        
        point = clickData['points'][0]
        lat, lon = point['lat'], point['lon']
        
        # Check if this is a grid node (firefighter station) or a fire point
        if 'text' in point and 'Unburnt Node' in point['text']:
            # This is a grid node - select as firefighter station (limit to 1)
            station_data = {'lat': lat, 'lon': lon}
            
            # Check if station is already selected
            station_already_selected = any(
                abs(s['lat'] - lat) < 0.001 and abs(s['lon'] - lon) < 0.001 
                for s in selected_stations
            )
            
            if station_already_selected:
                # Remove station
                selected_stations = []
                status_message = f"Removed firefighter station at Lat: {lat:.3f}, Lon: {lon:.3f}"
            else:
                # Replace any existing station with the new one (limit to 1)
                selected_stations = [station_data]
                status_message = f"Selected firefighter station at Lat: {lat:.3f}, Lon: {lon:.3f}"
            
            return no_update, status_message, no_update, True, no_update, selected_stations
        else:
            # This is a fire point - select as fire point
            selected_point_data = {'lat': lat, 'lon': lon}
            status_message = "Selected fire point at Lat: {:.3f}, Lon: {:.3f}".format(lat, lon)
            print(f"Fire point selected: {selected_point_data}")
            print("Enabling save button...")
        
        # Create a new figure that zooms in to the selected point
        if current_figure and 'data' in current_figure:
            # Get the existing data from the current figure
            existing_data = current_figure['data']
            
            # Create a new figure with the same data but zoomed to the selected point
            fig = go.Figure(data=existing_data)
            
            # If grid graph is active, create a local grid around the selected point
            grid_active = any(trace.get('name') == 'Grid Nodes' for trace in existing_data)
            if grid_active:
                # Create a local grid around the selected point
                local_bounds = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]  # 0.1 degree radius
                local_grid_data = create_grid_graph(local_bounds, grid_size=6)  # Smaller grid for local view
                
                if local_grid_data:
                    # Add local grid nodes
                    fig.add_trace(go.Scattermapbox(
                        lat=local_grid_data['node_lats'],
                        lon=local_grid_data['node_lons'],
                        mode='markers',
                        marker={'size': 4, 'color': 'green', 'symbol': 'circle'},
                        hoverinfo='text',
                        text=local_grid_data['node_texts'],
                        name='Local Grid Nodes'
                    ))
                    
                    # Add local grid edges
                    for edge in local_grid_data['edges']:
                        fig.add_trace(go.Scattermapbox(
                            lat=edge['lat'],
                            lon=edge['lon'],
                            mode='lines',
                            line={'color': 'lightgreen', 'width': 1},
                            hoverinfo='skip',
                            showlegend=False
                        ))
            
            # Use manual zoom level from dropdown
            zoom_level = map_zoom if map_zoom else 12  # Default to 12 if not set
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={'lat': lat, 'lon': lon},
                mapbox_zoom=zoom_level,
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            
            return fig, status_message, False, False, selected_point_data, selected_stations
        else:
            return no_update, status_message, False, False, selected_point_data, selected_stations

    # Callback 2.7: Enable save button when fire point is selected
    @app.callback(
        Output('save-graph-button', 'disabled', allow_duplicate=True),
        Output('run-mff-button', 'disabled', allow_duplicate=True),
        Input('selected-fire-point', 'data'),
        Input('selected-firefighter-stations', 'data'),
        Input('latest-saved-file', 'data'),
        Input('grid-toggle-state', 'data'),
        prevent_initial_call=True
    )
    def enable_buttons(selected_point, selected_stations, saved_file, grid_toggle_state):
        # Enable save button when we have a fire point, grid is toggled, and at least one firefighter station
        has_fire_point = selected_point is not None
        has_firefighter_station = selected_stations and len(selected_stations) > 0
        has_saved_file = saved_file is not None
        has_grid_toggled = grid_toggle_state is True
        
        save_enabled = has_fire_point and has_grid_toggled and has_firefighter_station
        # Enable MFF button if we have saved file OR if we have grid toggled with firefighter stations
        mff_enabled = has_saved_file or (has_grid_toggled and has_fire_point and has_firefighter_station)
        
        if save_enabled:
            print("Enabling save button - have fire point, grid toggled, and firefighter station")
        else:
            print(f"Save button disabled - fire point: {has_fire_point}, grid toggled: {has_grid_toggled}, firefighter station: {has_firefighter_station}")
            
        if mff_enabled:
            if has_saved_file:
                print("Enabling MFF button - have saved file")
            else:
                print("Enabling MFF button - have grid toggled with firefighter stations")
        else:
            print("MFF button disabled - missing requirements")
        
        return not save_enabled, not mff_enabled  # Return disabled states

    # Callback 2.5: Update grid when firefighter stations change
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Input('selected-firefighter-stations', 'data'),
        State('fire-map', 'figure'),
        State('grid-toggle-state', 'data'),
        prevent_initial_call=True
    )
    def update_grid_on_station_change(selected_stations, current_figure, grid_toggle_state):
        if not grid_toggle_state or not current_figure:
            return no_update
        
        # Get the existing data from the current figure
        existing_data = current_figure.get('data', [])
        current_layout = current_figure.get('layout', {})
        
        # Create new figure with existing data
        fig = go.Figure(data=existing_data)
        
        # Preserve current map center and zoom
        if 'mapbox' in current_layout:
            fig.update_layout(
                mapbox_style=current_layout['mapbox'].get('style', 'open-street-map'),
                mapbox_center=current_layout['mapbox'].get('center', {'lat': 0, 'lon': 0}),
                mapbox_zoom=current_layout['mapbox'].get('zoom', 4),
                margin=current_layout.get('margin', {"r":0,"t":0,"l":0,"b":0})
            )
        
        # Update only the grid nodes based on selection status
        for trace in fig.data:
            if hasattr(trace, 'name') and trace.name == 'Grid Node':
                # Get the coordinates of this node
                if len(trace.lat) > 0 and len(trace.lon) > 0:
                    node_lat = trace.lat[0]
                    node_lon = trace.lon[0]
                    
                    # Check if this node is selected as a firefighter station
                    is_selected_station = any(
                        abs(s['lat'] - node_lat) < 0.001 and abs(s['lon'] - node_lon) < 0.001 
                        for s in selected_stations
                    ) if selected_stations else False
                    
                    # Update the node's appearance
                    if is_selected_station:
                        trace.marker.color = 'blue'
                        trace.marker.size = 15
                        trace.text = [f"Firefighter Station<br>Lat: {node_lat:.4f}<br>Lon: {node_lon:.4f}"]
                    else:
                        trace.marker.color = 'lightgreen'
                        trace.marker.size = 12
                        trace.text = [f"Unburnt Node<br>Lat: {node_lat:.4f}<br>Lon: {node_lon:.4f}"]
        
        # Reorder traces to ensure nodes are on top of edges
        # Move all node traces to the end of the data list
        node_traces = []
        edge_traces = []
        other_traces = []
        
        for trace in fig.data:
            if hasattr(trace, 'name'):
                if trace.name == 'Grid Node' or trace.name == 'Fire Node':
                    node_traces.append(trace)
                elif trace.name == 'Grid Connection':
                    edge_traces.append(trace)
                else:
                    other_traces.append(trace)
            else:
                other_traces.append(trace)
        
        # Reorder: other traces first, then edges, then nodes (on top)
        fig.data = other_traces + edge_traces + node_traces
        
        return fig



    # Callback 3: Run the analysis when the button is clicked
    @app.callback(
        Output('results-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('selected-fire-point', 'data'),
        State('analysis-date-picker', 'date'),
        prevent_initial_call=True
    )
    def run_analysis_and_display(n_clicks, selected_point, selected_date):
        if not selected_point:
            return dbc.Alert("Error: No fire point selected for analysis.", color="danger")
        lat, lon = selected_point['lat'], selected_point['lon']
        try:
            results = run_analysis_pipeline(lat, lon, selected_date, api_key, grid_size=64)
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

    # Callback 4: Save graph data to JSON file
    @app.callback(
        Output('save-graph-button', 'disabled'),
        Output('selection-status', 'children', allow_duplicate=True),
        Output('latest-saved-file', 'data'),
        Output('run-mff-button', 'disabled'),
        Input('save-graph-button', 'n_clicks'),
        State('fire-map', 'figure'),
        State('selected-fire-point', 'data'),
        State('grid-toggle-state', 'data'),
        State('selected-firefighter-stations', 'data'),
        State('grid-graph-size-dropdown', 'value'),
        State('grid-spacing-dropdown', 'value'),
        State('d-value-input', 'value'),
        State('b-value-input', 'value'),
        State('lambda-d-input', 'value'),
        prevent_initial_call=True
    )
    def save_graph_data(n_clicks, current_figure, selected_point, grid_toggle_state, selected_stations, grid_size, grid_spacing, d_value, b_value, lambda_d):
        print(f"Save button clicked! n_clicks: {n_clicks}")
        print(f"Selected point: {selected_point}")
        print(f"Current figure: {current_figure is not None}")
        print(f"Grid size: {grid_size}, Grid spacing: {grid_spacing}")
        
        if not selected_point:
            print("No selected point - returning disabled")
            return True, "No fire point selected.", None, True
        
        if not current_figure:
            print("No current figure - returning disabled")
            return True, "No map data available.", None, True
        
        if not grid_toggle_state:
            print("Grid not toggled - returning disabled")
            return True, "Please toggle the grid graph first.", None, True
        
        if not selected_stations or len(selected_stations) == 0:
            print("No firefighter stations selected - returning disabled")
            return True, "Please select at least one firefighter station.", None, True
        
        try:
            import json
            from datetime import datetime
            import os
            try:
                import networkx as nx
            except ImportError:
                return True, "NetworkX library is required but not installed. Please install it with: pip install networkx", None, True
            import numpy as np
            
            def convert_numpy_types(obj):
                """Convert NumPy types to native Python types for JSON serialization."""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, float) and obj == float('inf'):
                    return "inf"  # Convert infinity to string for JSON compatibility
                return obj
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            total_nodes = grid_size * grid_size
            filename = f"mfp_n{total_nodes}_lambda{grid_spacing}_b1_{timestamp}_problem.json"
            
            # Extract graph data in the exact format of the reference file
            graph_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "description": "Moving Firefighter Problem Instance",
                    "version": "1.0"
                },
                "parameters": {
                    "n": grid_size * grid_size,  # Total number of nodes in grid (e.g., 3x3 = 9 nodes)
                    "lambda_d": lambda_d if lambda_d is not None else 1.0,
                    "burnt_nodes": 1,
                    "instance": 0,
                    "dimension": 2,
                    "D": d_value if d_value is not None else 3,
                    "B": b_value if b_value is not None else 3,
                    "seed": None
                },
                "graph": {
                    "adjacency_matrix": [],
                    "distance_matrix": [],
                    "burnt_nodes": [],
                    "num_vertices": 0,
                    "num_edges": 0,
                    "coordinates": None
                }
            }
            
            # Create grid-based nodes and distance matrix from scratch
            print(f"Creating grid-based distance matrix from scratch...")
            print(f"Grid size: {grid_size}x{grid_size} = {grid_size * grid_size} nodes")
            print(f"Grid spacing: {grid_spacing}°")
            
            # Get the center point (selected fire point)
            center_lat, center_lon = selected_point['lat'], selected_point['lon']
            print(f"Center point: ({center_lat}, {center_lon})")
            
            # Create grid nodes around the center point
            grid_nodes = []
            grid_size_half = grid_size // 2
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Calculate grid position relative to center
                    lat = center_lat + (i - grid_size_half) * grid_spacing
                    lon = center_lon + (j - grid_size_half) * grid_spacing
                    
                    grid_nodes.append({
                        'lat': lat,
                        'lon': lon,
                        'grid_i': i,
                        'grid_j': j,
                        'index': len(grid_nodes)
                    })
            
            print(f"Created {len(grid_nodes)} grid nodes")
            
            # Create adjacency matrix and graph distance matrix (not Haversine)
            num_nodes = len(grid_nodes)
            print(f"Creating grid-based distance and adjacency matrices for {num_nodes} nodes...")
            
            # Initialize adjacency matrix first
            adjacency_matrix = []
            
            for i in range(num_nodes):
                adj_row = []
                
                for j in range(num_nodes):
                    if i == j:
                        adj_row.append(0)
                    else:
                        # Create adjacency matrix (connect only orthogonal neighbors: top, bottom, left, right)
                        grid_i1, grid_j1 = grid_nodes[i]['grid_i'], grid_nodes[i]['grid_j']
                        grid_i2, grid_j2 = grid_nodes[j]['grid_i'], grid_nodes[j]['grid_j']
                        
                        # Check if nodes are orthogonally adjacent (no diagonals)
                        di = abs(grid_i1 - grid_i2)
                        dj = abs(grid_j1 - grid_j2)
                        
                        # Adjacent if: one step in one direction AND zero steps in the other
                        if (di == 1 and dj == 0) or (di == 0 and dj == 1):
                            adj_row.append(1)
                        else:
                            adj_row.append(0)
                
                adjacency_matrix.append(adj_row)
            
            # Create distance matrix using graph shortest paths (not Haversine)
            print(f"Calculating shortest path distances...")
            try:
                import networkx as nx
            except ImportError:
                return True, "NetworkX library is required but not installed. Please install it with: pip install networkx", None, True
            
            # Create NetworkX graph from adjacency matrix
            G = nx.Graph()
            for i in range(num_nodes):
                G.add_node(i)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adjacency_matrix[i][j] == 1:
                        G.add_edge(i, j, weight=1.0)  # Unit weight for grid steps
            
            # Calculate shortest path distance matrix
            distance_matrix = []
            for i in range(num_nodes):
                dist_row = []
                for j in range(num_nodes):
                    if i == j:
                        dist_row.append(0.0)
                    else:
                        try:
                            # Get shortest path length (number of grid steps)
                            path_length = nx.shortest_path_length(G, i, j, weight='weight')
                            dist_row.append(float(path_length))
                        except nx.NetworkXNoPath:
                            # If no path exists, use infinity
                            dist_row.append(float('inf'))
                distance_matrix.append(dist_row)
            
            print(f"Distance matrix created: {len(distance_matrix)}x{len(distance_matrix[0])}")
            print(f"Adjacency matrix created: {len(adjacency_matrix)}x{len(adjacency_matrix[0])}")
            print(f"Sample distances: {distance_matrix[0][:3]}")
            print(f"Sample adjacency: {adjacency_matrix[0][:3]}")
            
            # Find the center node (closest to selected fire point)
            center_node_index = grid_size_half * grid_size + grid_size_half
            print(f"Center node index: {center_node_index}")
            
            # Calculate graph statistics
            num_edges = sum(sum(row) for row in adjacency_matrix) // 2  # Divide by 2 for undirected graph
            print(f"Number of edges: {num_edges}")
            
            if num_nodes > 0:
                # Calculate graph statistics using the adjacency matrix
                avg_degree = sum(sum(row) for row in adjacency_matrix) / num_nodes if num_nodes > 0 else 0
                density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
                
                # Find burnt node (center node)
                burnt_node_index = center_node_index
                print(f"Burnt node index: {burnt_node_index}")
                
                # Create coordinates array for the grid nodes
                coordinates = []
                for node in grid_nodes:
                    coordinates.append([node['lat'], node['lon']])
                
                # Update graph data with both matrices (convert NumPy types)
                graph_data["graph"]["adjacency_matrix"] = convert_numpy_types(adjacency_matrix)
                graph_data["graph"]["distance_matrix"] = convert_numpy_types(distance_matrix)
                graph_data["graph"]["coordinates"] = convert_numpy_types(coordinates)  # Add actual node positions
                graph_data["graph"]["num_vertices"] = convert_numpy_types(num_nodes)
                graph_data["graph"]["num_edges"] = convert_numpy_types(num_edges)
                graph_data["graph"]["burnt_nodes"] = convert_numpy_types([burnt_node_index])
                
                print(f"Saved adjacency matrix: {len(graph_data['graph']['adjacency_matrix'])}x{len(graph_data['graph']['adjacency_matrix'][0]) if graph_data['graph']['adjacency_matrix'] else 0}")
                print(f"Saved distance matrix: {len(graph_data['graph']['distance_matrix'])}x{len(graph_data['graph']['distance_matrix'][0]) if graph_data['graph']['distance_matrix'] else 0}")
                print(f"Distance matrix sample: {graph_data['graph']['distance_matrix'][0][:3] if graph_data['graph']['distance_matrix'] else 'None'}")
                print(f"🔥 Fire node (center): index {burnt_node_index}, coords {coordinates[burnt_node_index]}")
                print(f"📊 Grid size: {grid_size}x{grid_size}, total nodes: {num_nodes}")
                print(f"📐 Grid indexing: center should be at ({grid_size_half}, {grid_size_half}) = index {center_node_index}")
                
                # Debug: Show grid node layout
                print(f"📍 Grid node layout (first few nodes):")
                for i in range(min(5, len(grid_nodes))):
                    node = grid_nodes[i]
                    print(f"   Node {i}: grid_pos({node['grid_i']}, {node['grid_j']}) -> coords({node['lat']:.6f}, {node['lon']:.6f})")
                
                # Add firefighter stations if any are selected
                if selected_stations and len(selected_stations) > 0:
                    firefighter_indices = []
                    for station in selected_stations:
                        # Find the grid node closest to this firefighter station
                        min_distance = float('inf')
                        closest_index = 0
                        for i, node in enumerate(grid_nodes):
                            lat1, lon1 = station['lat'], station['lon']
                            lat2, lon2 = node['lat'], node['lon']
                            
                            # Calculate simple Euclidean distance (sufficient for small grids)
                            distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_index = i
                        
                        firefighter_indices.append(closest_index)
                        print(f"Firefighter station at grid node {closest_index}")
                    
                    # Add firefighter stations to graph data
                    graph_data["graph"]["firefighter_stations"] = convert_numpy_types(firefighter_indices)
                    graph_data["parameters"]["firefighters"] = len(firefighter_indices)
                else:
                    graph_data["parameters"]["firefighters"] = 0
                
                # Add additional metadata
                graph_data["metadata"]["graph_stats"] = {
                    "average_degree": round(avg_degree, 3),
                    "density": round(density, 3),
                    "diameter": "Grid-based (connected)",
                    "average_clustering": "N/A (grid structure)",
                    "average_shortest_path": "N/A (grid structure)"
                }
            else:
                print(f"⚠️  No nodes found in the graph!")
                print(f"Fire points: {len(fire_points)}")
                print(f"Grid nodes: {len(grid_nodes)}")
                print(f"Current figure data: {current_figure.keys() if current_figure else 'None'}")
                return False, "No nodes found in the graph. Please select a fire point and toggle the grid graph first.", None, True
            
            # Save to file
            filepath = os.path.join(os.getcwd(), filename)
            
            # Debug: Check what's in the JSON before saving
            print(f"Final JSON structure:")
            print(f"- Adjacency matrix: {len(graph_data['graph']['adjacency_matrix'])}x{len(graph_data['graph']['adjacency_matrix'][0]) if graph_data['graph']['adjacency_matrix'] else 0}")
            print(f"- Distance matrix: {len(graph_data['graph']['distance_matrix'])}x{len(graph_data['graph']['distance_matrix'][0]) if graph_data['graph']['distance_matrix'] else 0}")
            print(f"- Distance matrix type: {type(graph_data['graph']['distance_matrix'])}")
            print(f"- Distance matrix sample: {graph_data['graph']['distance_matrix'][0][:3] if graph_data['graph']['distance_matrix'] else 'None'}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, default=str)
            
            # Verify the saved file
            with open(filepath, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                print(f"Verified saved file:")
                print(f"- Adjacency matrix in file: {len(saved_data['graph']['adjacency_matrix'])}x{len(saved_data['graph']['adjacency_matrix'][0]) if saved_data['graph']['adjacency_matrix'] else 0}")
                print(f"- Distance matrix in file: {len(saved_data['graph']['distance_matrix'])}x{len(saved_data['graph']['distance_matrix'][0]) if saved_data['graph']['distance_matrix'] else 0}")
                print(f"- Distance matrix sample from file: {saved_data['graph']['distance_matrix'][0][:3] if saved_data['graph']['distance_matrix'] else 'None'}")
            
            status_message = f"✅ Graph data saved successfully to {filename}"
            return False, status_message, filename, False  # Enable MFF button
            
        except Exception as e:
            error_message = f"Failed to save graph data: {str(e)}"
            return False, error_message, None, True  # Keep MFF button disabled

    # Callback 5: Run MFF Solver
    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Output('mff-solution-data', 'data'),
        Output('run-mff-button', 'disabled', allow_duplicate=True),
        Output('latest-saved-file', 'data', allow_duplicate=True),
        Input('run-mff-button', 'n_clicks'),
        State('latest-saved-file', 'data'),
        State('fire-map', 'figure'),
        State('selected-fire-point', 'data'),
        State('grid-toggle-state', 'data'),
        State('selected-firefighter-stations', 'data'),
        State('grid-graph-size-dropdown', 'value'),
        State('grid-spacing-dropdown', 'value'),
        State('d-value-input', 'value'),
        State('b-value-input', 'value'),
        State('lambda-d-input', 'value'),
        prevent_initial_call=True
    )
    def run_mff_solver(n_clicks, saved_file, current_figure, selected_point, grid_toggle_state, selected_stations, grid_size, grid_spacing, d_value, b_value, lambda_d):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # If no saved file, we need to save the graph data first
        if not saved_file:
            print("No saved file found, saving graph data first...")
            try:
                # Call the save_graph_data function logic inline
                import json
                from datetime import datetime
                import os
                try:
                    import networkx as nx
                except ImportError:
                    error_content = [
                        dbc.Alert("NetworkX library is required but not installed. Please install it with: pip install networkx", color="danger", className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader(html.H4("❌ Missing Dependency", className="mb-0")),
                            dbc.CardBody([
                                html.P("The NetworkX library is required for graph operations."),
                                html.P("Please install it with: pip install networkx")
                            ])
                        ])
                    ]
                    return error_content, None, True, None
                import numpy as np
                
                def convert_numpy_types(obj):
                    """Convert NumPy types to native Python types for JSON serialization."""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, float) and obj == float('inf'):
                        return "inf"
                    return obj
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                total_nodes = grid_size * grid_size
                filename = f"mfp_n{total_nodes}_lambda{grid_spacing}_b1_{timestamp}_problem.json"
                
                # Create the graph data (simplified version of save_graph_data logic)
                graph_data = {
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "description": "Moving Firefighter Problem Instance",
                        "version": "1.0"
                    },
                    "parameters": {
                        "n": grid_size * grid_size,
                        "lambda_d": lambda_d if lambda_d is not None else 1.0,
                        "burnt_nodes": 1,
                        "instance": 0,
                        "dimension": 2,
                        "D": d_value if d_value is not None else 3,
                        "B": b_value if b_value is not None else 3,
                        "seed": None
                    },
                    "graph": {
                        "adjacency_matrix": [],
                        "distance_matrix": [],
                        "burnt_nodes": [],
                        "num_vertices": 0,
                        "num_edges": 0,
                        "coordinates": None
                    }
                }
                
                # Create grid nodes and matrices
                center_lat, center_lon = selected_point['lat'], selected_point['lon']
                grid_nodes = []
                grid_size_half = grid_size // 2
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        lat = center_lat + (i - grid_size_half) * grid_spacing
                        lon = center_lon + (j - grid_size_half) * grid_spacing
                        grid_nodes.append({
                            'lat': lat,
                            'lon': lon,
                            'grid_i': i,
                            'grid_j': j,
                            'index': len(grid_nodes)
                        })
                
                # Create adjacency and distance matrices
                num_nodes = len(grid_nodes)
                adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
                distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
                
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            # Check if nodes are adjacent (Manhattan distance = 1)
                            node_i = grid_nodes[i]
                            node_j = grid_nodes[j]
                            manhattan_dist = abs(node_i['grid_i'] - node_j['grid_i']) + abs(node_i['grid_j'] - node_j['grid_j'])
                            
                            if manhattan_dist == 1:
                                adjacency_matrix[i][j] = 1
                                distance_matrix[i][j] = 1
                            else:
                                # Calculate shortest path distance
                                try:
                                    G = nx.Graph()
                                    for k in range(num_nodes):
                                        G.add_node(k)
                                    for k in range(num_nodes):
                                        for l in range(num_nodes):
                                            if adjacency_matrix[k][l] == 1:
                                                G.add_edge(k, l)
                                    path_length = nx.shortest_path_length(G, i, j, weight='weight')
                                    distance_matrix[i][j] = float(path_length)
                                except:
                                    distance_matrix[i][j] = float('inf')
                
                # Find center node and burnt node
                center_node_index = grid_size_half * grid_size + grid_size_half
                coordinates = [[node['lat'], node['lon']] for node in grid_nodes]
                
                # Update graph data
                graph_data["graph"]["adjacency_matrix"] = convert_numpy_types(adjacency_matrix)
                graph_data["graph"]["distance_matrix"] = convert_numpy_types(distance_matrix)
                graph_data["graph"]["coordinates"] = convert_numpy_types(coordinates)
                graph_data["graph"]["num_vertices"] = convert_numpy_types(num_nodes)
                graph_data["graph"]["num_edges"] = convert_numpy_types(sum(sum(row) for row in adjacency_matrix) // 2)
                graph_data["graph"]["burnt_nodes"] = convert_numpy_types([center_node_index])
                
                # Add firefighter stations
                if selected_stations and len(selected_stations) > 0:
                    firefighter_indices = []
                    for station in selected_stations:
                        min_distance = float('inf')
                        closest_index = 0
                        for i, node in enumerate(grid_nodes):
                            lat1, lon1 = station['lat'], station['lon']
                            lat2, lon2 = node['lat'], node['lon']
                            distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_index = i
                        firefighter_indices.append(closest_index)
                    
                    graph_data["graph"]["firefighter_stations"] = convert_numpy_types(firefighter_indices)
                    graph_data["parameters"]["firefighters"] = len(firefighter_indices)
                else:
                    graph_data["parameters"]["firefighters"] = 0
                
                # Save to file
                filepath = os.path.join(os.getcwd(), filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                
                saved_file = filename
                print(f"Auto-saved graph data to: {filename}")
                
            except Exception as e:
                error_content = [
                    dbc.Alert(f"Failed to auto-save graph data: {str(e)}", color="danger", className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader(html.H4("❌ Auto-Save Error", className="mb-0")),
                        dbc.CardBody([
                            html.P("Please manually save the graph data first using the 'Save Graph Data' button."),
                            html.P(f"Error: {str(e)}")
                        ])
                    ])
                ]
                return error_content, None, True, None
        
        try:
            # Import MFF integration module
            from modules.mff_integration import run_mff_inference, create_timeline_visualization_plotly
            
            print(f"🔄 Running MFF solver on: {saved_file}")
            
            # Run the MFF inference
            result = run_mff_inference(saved_file, time_limit=300, verbose=True)
            
            if result['success']:
                solution_data = result['solution_data']
                problem_data = result['problem_data']
                
                # Create timeline visualization
                timeline_fig, timeline_msg = create_timeline_visualization_plotly(problem_data, solution_data)
                
                # Create results display
                results_content = [
                    dbc.Alert(result['message'], color="success", className="mb-3"),
                    
                    # Solution Summary Card
                    dbc.Card([
                        dbc.CardHeader(html.H4("🎯 MFF Solution Summary", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Problem Details", className="text-muted mb-2"),
                                    html.P(f"• Vertices: {problem_data['parameters']['n']}", className="mb-1"),
                                    html.P(f"• Initial fires: {len(problem_data['graph']['burnt_nodes'])}", className="mb-1"),
                                    html.P(f"• Defense rounds (D): {problem_data['parameters'].get('D', 3)}", className="mb-1"),
                                    html.P(f"• Burning rounds (B): {problem_data['parameters'].get('B', 3)}", className="mb-1"),
                                ], md=6),
                                dbc.Col([
                                    html.H6("Solution Results", className="text-muted mb-2"),
                                    html.P(f"• Feasible: {'✅ Yes' if solution_data['feasible'] else '❌ No'}", className="mb-1"),
                                    html.P(f"• Objective (burned): {solution_data['objective']}", className="mb-1"),
                                    html.P(f"• Runtime: {solution_data['runtime']:.2f}s", className="mb-1"),
                                    html.P(f"• Solver: {solution_data['solver']}", className="mb-1"),
                                ], md=6)
                            ]),
                            html.Hr(),
                            html.H6("Defense Sequence", className="text-muted mb-2"),
                            html.P(f"Firefighter path: {[v for v, _, _ in solution_data['defense_sequence']]}" if solution_data['defense_sequence'] else "No defense sequence available")
                        ])
                    ], className="mb-3"),
                ]
                
                # Add timeline visualization if available
                if timeline_fig:
                    results_content.append(
                        dbc.Card([
                            dbc.CardHeader(html.H4("🎬 Solution Timeline", className="mb-0")),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='timeline-graph',
                                    figure=timeline_fig, 
                                    style={"height": "600px"}
                                ),
                                html.P(timeline_msg, className="text-muted mt-2")
                            ])
                        ], className="mb-3")
                    )
                else:
                    results_content.append(
                        dbc.Alert(f"Timeline visualization: {timeline_msg}", color="warning", className="mb-3")
                    )
                
                return results_content, result, False, saved_file
                
            else:
                error_content = [
                    dbc.Alert(result['message'], color="danger", className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader(html.H4("❌ MFF Solver Error", className="mb-0")),
                        dbc.CardBody([
                            html.P(f"Error: {result['error']}"),
                            html.P("Possible solutions:"),
                            html.Ul([
                                html.Li("Check that the MFF dependencies are installed"),
                                html.Li("Verify the problem file format is correct"),
                                html.Li("Try reducing the problem size"),
                                html.Li("Check that SCIP solver is available")
                            ])
                        ])
                    ])
                ]
                return error_content, None, False, saved_file
                
        except Exception as e:
            error_content = [
                dbc.Alert(f"MFF solver failed: {str(e)}", color="danger", className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H4("❌ System Error", className="mb-0")),
                    dbc.CardBody([
                        html.P(f"System error: {str(e)}"),
                        html.P("This usually indicates missing dependencies. Please install:"),
                        html.Pre([
                            "pip install pyscipopt\n",
                            "pip install moving-firefighter-problem-generator\n",
                            "# And copy the movingff_paper directory to your project"
                        ], style={'background-color': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'})
                    ])
                ])
            ]
            return error_content, None, False, saved_file

    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Output('mff-solution-data', 'data', allow_duplicate=True),
        Input('load-example-1-button', 'n_clicks'),
        Input('load-example-2-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def load_example_solution(n_clicks_1, n_clicks_2):
        # Debug: Print the actual click counts
        print(f"🔍 Button click detection:")
        print(f"   n_clicks_1: {n_clicks_1} (type: {type(n_clicks_1)})")
        print(f"   n_clicks_2: {n_clicks_2} (type: {type(n_clicks_2)})")
        
        # Determine which button was clicked
        if n_clicks_1 and n_clicks_1 > 0:
            example_num = 1
            print(f"   ✅ Detected Example 1 button clicked")
        elif n_clicks_2 and n_clicks_2 > 0:
            example_num = 2
            print(f"   ✅ Detected Example 2 button clicked")
        else:
            print(f"   ❌ No button detected as clicked")
            return dash.no_update, dash.no_update
        
        try:
            # Import MFF integration module
            from modules.mff_integration import create_timeline_visualization_plotly
            import json
            import os
            
            # Define the example file paths based on which button was clicked
            if example_num == 1:
                problem_file = "mfp_n25_lambda0.05_b1_20250803_171230_problem.json"
                solution_file = "mff_solution_20250803_171230.json"
                example_name = "Example 1"
            else:  # example_num == 2
                problem_file = "mfp_n25_lambda0.05_b1_20250803_175119_problem.json"
                solution_file = "mff_solution_20250803_175120.json"
                example_name = "Example 2"
            
            print(f"🔄 Loading {example_name}...")
            print(f"   Problem file: {problem_file}")
            print(f"   Solution file: {solution_file}")
            print(f"   Example number: {example_num}")
            
            # Check if files exist
            if not os.path.exists(problem_file):
                error_content = [
                    dbc.Alert(f"Example problem file not found: {problem_file}", color="danger", className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader(html.H4("❌ File Not Found", className="mb-0")),
                        dbc.CardBody([
                            html.P(f"Please ensure the example files are in the project root directory:"),
                            html.Ul([
                                html.Li(f"• {problem_file}"),
                                html.Li(f"• {solution_file}")
                            ])
                        ])
                    ])
                ]
                return error_content, None
            
            if not os.path.exists(solution_file):
                error_content = [
                    dbc.Alert(f"Example solution file not found: {solution_file}", color="danger", className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader(html.H4("❌ File Not Found", className="mb-0")),
                        dbc.CardBody([
                            html.P(f"Please ensure the example files are in the project root directory:"),
                            html.Ul([
                                html.Li(f"• {problem_file}"),
                                html.Li(f"• {solution_file}")
                            ])
                        ])
                    ])
                ]
                return error_content, None
            
            # Load the problem and solution data
            with open(problem_file, 'r') as f:
                problem_data = json.load(f)
            
            with open(solution_file, 'r') as f:
                solution_data = json.load(f)
            
            print(f"   ✅ Loaded problem: {problem_data['parameters']['n']} nodes")
            print(f"   ✅ Loaded solution: feasible={solution_data.get('feasible', False)}")
            print(f"   🔍 Problem data keys: {list(problem_data.keys())}")
            print(f"   🔍 Solution data keys: {list(solution_data.keys())}")
            
            # Check if solution has nested structure
            if 'solution' in solution_data:
                print(f"   🔍 Solution nested keys: {list(solution_data['solution'].keys())}")
                if 'defense_sequence' in solution_data['solution']:
                    print(f"   🔍 Defense sequence found in nested solution: {len(solution_data['solution']['defense_sequence'])} items")
            elif 'defense_sequence' in solution_data:
                print(f"   🔍 Defense sequence found in root: {len(solution_data['defense_sequence'])} items")
            else:
                print(f"   ⚠️  No defense sequence found in solution data")
            
            # Create timeline visualization
            print(f"   🎬 Creating timeline visualization for {example_name}...")
            print(f"   🎬 Problem data type: {type(problem_data)}")
            print(f"   🎬 Solution data type: {type(solution_data)}")
            timeline_fig, timeline_msg = create_timeline_visualization_plotly(problem_data, solution_data)
            print(f"   🎬 Timeline created: {timeline_fig is not None}")
            print(f"   🎬 Timeline message: {timeline_msg}")
            
            if timeline_fig:
                print(f"   🎬 Timeline figure has {len(timeline_fig.data)} traces")
                print(f"   🎬 Timeline figure layout type: {type(timeline_fig.layout)}")
            else:
                print(f"   ❌ Timeline figure is None")
            
            # Create results display
            results_content = [
                dbc.Alert(f"✅ {example_name} loaded successfully!", color="success", className="mb-3"),
                
                # Solution Summary Card
                dbc.Card([
                    dbc.CardHeader(html.H4(f"🎯 {example_name} MFF Solution", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Problem Details", className="text-muted mb-2"),
                                html.P(f"• Vertices: {problem_data['parameters']['n']}", className="mb-1"),
                                html.P(f"• Initial fires: {len(problem_data['graph']['burnt_nodes'])}", className="mb-1"),
                                html.P(f"• Lambda: {problem_data['parameters'].get('lambda_d', 'N/A')}", className="mb-1"),
                                html.P(f"• Defense rounds (D): {problem_data['parameters'].get('D', 3)}", className="mb-1"),
                                html.P(f"• Burning rounds (B): {problem_data['parameters'].get('B', 3)}", className="mb-1"),
                            ], md=6),
                            dbc.Col([
                                html.H6("Solution Results", className="text-muted mb-2"),
                                html.P(f"• Feasible: {'✅ Yes' if solution_data.get('feasible', False) else '❌ No'}", className="mb-1"),
                                html.P(f"• Objective (burned): {solution_data.get('objective', 'N/A')}", className="mb-1"),
                                html.P(f"• Runtime: {solution_data.get('runtime', 'N/A')}s", className="mb-1"),
                                html.P(f"• Solver: {solution_data.get('solver', 'N/A')}", className="mb-1"),
                            ], md=6)
                        ]),
                        html.Hr(),
                        html.H6("Defense Sequence", className="text-muted mb-2"),
                        html.P(f"Firefighter path: {[v for v, _, _ in solution_data.get('defense_sequence', [])]}" if solution_data.get('defense_sequence') else "No defense sequence available")
                    ])
                ], className="mb-3"),
            ]
            
            # Add timeline visualization if available
            if timeline_fig:
                results_content.append(
                    dbc.Card([
                        dbc.CardHeader(html.H4("🎬 Solution Timeline", className="mb-0")),
                        dbc.CardBody([
                            dcc.Graph(figure=timeline_fig, style={"height": "600px"}),
                            html.P(timeline_msg, className="text-muted mt-2")
                        ])
                    ], className="mb-3")
                )
            else:
                results_content.append(
                    dbc.Alert(f"Timeline visualization: {timeline_msg}", color="warning", className="mb-3")
                )
            
            return results_content, solution_data
            
        except Exception as e:
            import traceback
            print(f"   ❌ Error loading {example_name}: {str(e)}")
            print(f"   ❌ Full traceback:")
            traceback.print_exc()
            
            error_content = [
                dbc.Alert(f"Failed to load {example_name}: {str(e)}", color="danger", className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H4("❌ Load Error", className="mb-0")),
                    dbc.CardBody([
                        html.P(f"Error: {str(e)}"),
                        html.P("Please ensure the example files are in the correct location and format."),
                        html.P("Check the console for detailed error information.")
                    ])
                ])
            ]
            return error_content, None

    # Add new callback for grid options changes
    @app.callback(
        Output('fire-map', 'figure', allow_duplicate=True),
        Input('grid-graph-size-dropdown', 'value'),
        Input('grid-spacing-dropdown', 'value'),
        Input('map-zoom-dropdown', 'value'),
        State('fire-map', 'figure'),
        State('grid-toggle-state', 'data'),
        State('selected-fire-point', 'data'),
        State('selected-firefighter-stations', 'data'),
        prevent_initial_call=True
    )
    def update_grid_on_option_change(grid_graph_size, grid_spacing, map_zoom, current_figure, grid_toggle_state, selected_point, selected_stations):
        """Update the grid when grid options change."""
        if not current_figure or not grid_toggle_state:
            return no_update
        
        # Get the current map data and layout
        existing_data = current_figure.get('data', [])
        current_layout = current_figure.get('layout', {})
        
        # Filter out existing grid lines and firefighter stations to avoid duplication
        filtered_data = []
        for trace in existing_data:
            # Keep only non-grid traces (fire points, etc.)
            if not (trace.get('name', '').startswith('Grid Line') or 
                   trace.get('name', '').startswith('Firefighter Station')):
                filtered_data.append(trace)
        
        # Create new figure with filtered data (no grid lines)
        fig = go.Figure(data=filtered_data)
        
        # Preserve current map center and zoom
        if 'mapbox' in current_layout:
            fig.update_layout(
                mapbox_style=current_layout['mapbox'].get('style', 'open-street-map'),
                mapbox_center=current_layout['mapbox'].get('center', {'lat': 0, 'lon': 0}),
                mapbox_zoom=map_zoom if map_zoom else current_layout['mapbox'].get('zoom', 4),
                margin=current_layout.get('margin', {"r":0,"t":0,"l":0,"b":0})
            )
        
        # If grid is currently shown, update it with new parameters
        if grid_toggle_state and selected_point:
            try:
                # Extract coordinates from selected point
                center_lat = selected_point.get('lat')
                center_lon = selected_point.get('lon')
                
                if center_lat is not None and center_lon is not None:
                    # Create grid around the selected point
                    grid_spacing_deg = grid_spacing if grid_spacing else 0.005
                    grid_size = grid_graph_size if grid_graph_size else 7
                    
                    # Calculate grid bounds
                    half_size = (grid_size - 1) / 2
                    min_lat = center_lat - (half_size * grid_spacing_deg)
                    max_lat = center_lat + (half_size * grid_spacing_deg)
                    min_lon = center_lon - (half_size * grid_spacing_deg)
                    max_lon = center_lon + (half_size * grid_spacing_deg)
                    
                    # Create grid lines
                    grid_lines = []
                    
                    # Vertical lines
                    for i in range(grid_size):
                        lon = min_lon + (i * grid_spacing_deg)
                        grid_lines.append(go.Scattermapbox(
                            lat=[min_lat, max_lat],
                            lon=[lon, lon],
                            mode='lines',
                            line=dict(color='rgba(0, 0, 255, 0.3)', width=1),
                            name=f'Grid Line V{i}',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Horizontal lines
                    for i in range(grid_size):
                        lat = min_lat + (i * grid_spacing_deg)
                        grid_lines.append(go.Scattermapbox(
                            lat=[lat, lat],
                            lon=[min_lon, max_lon],
                            mode='lines',
                            line=dict(color='rgba(0, 0, 255, 0.3)', width=1),
                            name=f'Grid Line H{i}',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Add grid lines to figure
                    for line in grid_lines:
                        fig.add_trace(line)
                    
                    # Update firefighter stations if any are selected
                    if selected_stations:
                        station_traces = []
                        for i, station in enumerate(selected_stations):
                            station_traces.append(go.Scattermapbox(
                                lat=[station['lat']],
                                lon=[station['lon']],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color='green',
                                    symbol='star'
                                ),
                                name=f'Firefighter Station {i+1}',
                                text=f"Station {i+1}<br>Lat: {station['lat']:.4f}<br>Lon: {station['lon']:.4f}",
                                hoverinfo='text'
                            ))
                        
                        for trace in station_traces:
                            fig.add_trace(trace)
                    
                    print(f"Updated grid with size={grid_size}, spacing={grid_spacing_deg}, zoom={map_zoom}")
                    
            except Exception as e:
                print(f"Error updating grid on option change: {e}")
        
        return fig