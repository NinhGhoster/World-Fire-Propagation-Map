# modules/callbacks.py
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import no_update

from .data_fetcher import get_fire_data, get_country_list
from .simulation import FireSpreadSimulator, SimulationConfig
from .fire_stations import get_coverage_status, get_nearest_stations

def register_callbacks(app, api_key):
    """Register all Dash callbacks."""
    
    # 1. Update fire map
    @app.callback(
        [Output('fire-map', 'figure'),
         Output('map-status-message', 'children'),
         Output('analyze-button', 'disabled')],
        [Input('country-dropdown', 'value'),
         Input('analysis-date-picker', 'date')]
    )
    def update_fire_map(country, date):
        if not country:
            return go.Figure(), "Select a country to view fires.", True
        
        try:
            country_df = get_country_list()
            bbox = country_df[country_df['abreviation'] == country]['bbox_coords'].values[0]
            west, south, east, north = map(float, bbox.split(','))
            
            boundary_str = f"{west},{south},{east},{north}"
            df = get_fire_data(api_key, boundary_str, start_date=date)
            
            fig = go.Figure()
            center_lat = (south + north) / 2
            center_lon = (west + east) / 2
            
            if not df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=0.7),
                    text=[f"🔥 {row['latitude']:.2f}, {row['longitude']:.2f}<br>Brightness: {row.get('brightness', 'N/A')}K" 
                         for _, row in df.iterrows()],
                    hoverinfo='text',
                    name='Active Fires'
                ))
                status = f"✅ Found {len(df)} active fires in {country}"
            else:
                status = f"⚠️ No fires detected in {country}"
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=country == 'AU' and 3 or 4),
                margin=dict(l=10, r=10, t=10, b=10),
                height=480,
                showlegend=True
            )
            
            return fig, status, False
            
        except Exception as e:
            return go.Figure(), f"❌ Error: {str(e)}", True
    
    # 2. Handle fire point click
    @app.callback(
        [Output('selection-status', 'children'),
         Output('selected-fire-point', 'data')],
        [Input('fire-map', 'clickData')]
    )
    def handle_click(clickData):
        if clickData is None:
            return "👆 Click a fire point to select it", None
        
        point = clickData['points'][0]
        lat, lon = point['lat'], point['lon']
        
        return f"📍 Selected: {lat:.4f}, {lon:.4f}", {'lat': lat, 'lon': lon}
    
    # 3. Run Analysis with Fire Stations
    @app.callback(
        Output('results-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('selected-fire-point', 'data'),
        State('country-dropdown', 'value'),
        prevent_initial_call=True
    )
    def run_analysis(n_clicks, selected_point, country):
        if not selected_point:
            return dbc.Alert("⚠️ Please select a fire point first!", color="warning")
        
        lat, lon = selected_point['lat'], selected_point['lon']
        
        try:
            country_df = get_country_list()
            country_name = country
            for _, row in country_df.iterrows():
                if row['abreviation'] == country:
                    country_name = row['name']
                    break
            
            bbox = country_df[country_df['abreviation'] == country]['bbox_coords'].values[0]
            west, south, east, north = map(float, bbox.split(','))
            boundary_str = f"{west},{south},{east},{north}"
            df = get_fire_data(api_key, boundary_str)
            
            total_fires = len(df) if not df.empty else 0
            region_area = (east - west) * (north - south) * 111 * 111
            
            if not df.empty:
                df['distance'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)**0.5
                nearby = df[df['distance'] < 2.0]
                nearby_fires = len(nearby)
                avg_brightness = nearby['brightness'].mean() if not nearby.empty else df['brightness'].mean()
                total_frp = nearby['frp'].sum() if not nearby.empty and 'frp' in nearby else df['frp'].sum()
            else:
                nearby_fires = 0
                avg_brightness = 0
                total_frp = 0
            
            # Get fire station coverage
            coverage = get_coverage_status(lat, lon, country)
            
            return dbc.Card([
                dbc.CardHeader([
                    html.H5("🔥 Fire Analysis Results", className="mb-0"),
                    html.Span(f" {country_name}", className="badge bg-secondary ms-2")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.H3(str(total_fires), className="text-danger mb-0"),
                                html.P("Total Fires", className="small text-muted mb-0")], width=3, className="text-center"),
                        dbc.Col([html.H3(str(nearby_fires), className="text-warning mb-0"),
                                html.P("Nearby", className="small text-muted mb-0")], width=3, className="text-center"),
                        dbc.Col([html.H3(f"{avg_brightness:.0f}K", className="text-info mb-0"),
                                html.P("Avg Brightness", className="small text-muted mb-0")], width=3, className="text-center"),
                        dbc.Col([html.H3(f"{total_frp:.0f} MW", className="text-success mb-0"),
                                html.P("Total FRP", className="small text-muted mb-0")], width=3, className="text-center"),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("🚒 Fire Station Coverage", className="mb-2"),
                    coverage['nearest_station'] and dbc.Alert([
                        html.Strong(f"Nearest: {coverage['nearest_station']}"),
                        html.Br(),
                        f"Response time: {coverage['best_response_time']} min",
                        html.Br(),
                        f"Available trucks: {coverage['total_available_trucks']}"
                    ], color="info", className="small mb-2") or dbc.Alert("No stations in coverage area", color="warning"),
                    html.Hr(),
                    html.P([
                        html.Strong("Selected: "), f"{lat:.4f}°, {abs(lon):.4f}°"
                    ]),
                ])
            ], className="shadow-sm mt-3")
            
        except Exception as e:
            return dbc.Alert(f"❌ Analysis failed: {e}", color="danger")
    
    # 4. Run Simulation
    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Input('simulate-button', 'n_clicks'),
        State('grid-graph-size-dropdown', 'value'),
        State('lambda-dropdown', 'value'),
        State('firefighters-dropdown', 'value'),
        State('wind-speed-dropdown', 'value'),
        State('wind-direction-dropdown', 'value'),
        State('response-time-dropdown', 'value'),
        prevent_initial_call=True
    )
    def run_simulation(n_clicks, grid_size, lambda_val, firefighters, wind_speed, wind_dir, response_time):
        if not n_clicks:
            return no_update
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_val,
            num_firefighters=firefighters,
            fire_start_nodes=[grid_size**2 // 2],
            seed=42,
            wind_speed=wind_speed,
            wind_direction=wind_dir
        )
        
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy="greedy")
        
        total_nodes = grid_size ** 2
        protection_pct = (result.total_protected / total_nodes) * 100
        burn_pct = (result.total_burned / total_nodes) * 100
        
        direction_arrows = {"N": "⬆️", "NE": "↗️", "E": "➡️", "SE": "↘️", "S": "⬇️", "SW": "↙️", "W": "⬅️", "NW": "↖️"}
        
        ascii_lines = []
        for row in range(grid_size):
            line = ""
            for col in range(grid_size):
                idx = row * grid_size + col
                if idx in result.firefighter_placements.values():
                    line += "🛡️"
                elif idx in result.burned_nodes:
                    line += "⬛"
                elif idx in simulator.burning:
                    line += "🔥"
                else:
                    line += "░"
            ascii_lines.append(line)
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🧯 Fire Spread Simulation", className="mb-0 d-inline"),
                html.Span(f" {grid_size}×{grid_size}", className="badge bg-secondary ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span(direction_arrows.get(wind_dir, "➡️"), className="display-6"),
                            html.P(f"{wind_dir} {wind_speed}km/h", className="mb-0 small")
                        ], className="text-center")
                    ], width=2),
                    dbc.Col([
                        html.H6(f"🌬️ Wind Spread: ×{(1 + wind_speed/25):.1f}", className="mb-1"),
                        html.Progress([html.Bar(min=0, max=100, value=wind_speed, striped=True, animated=True)], className="mb-1"),
                    ], width=10)
                ], className="mb-3"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([html.H3(str(result.total_burned), className="text-danger"),
                            html.P("Burned", className="small text-muted")], className="text-center"),
                    dbc.Col([html.H3(str(result.total_protected), className="text-success"),
                            html.P("Protected", className="small text-muted")], className="text-center"),
                    dbc.Col([html.H3(f"{burn_pct:.0f}%", className="text-danger"),
                            html.P("Burned %", className="small text-muted")], className="text-center"),
                    dbc.Col([html.H3(f"{protection_pct:.0f}%", className="text-success"),
                            html.P("Protected %", className="small text-muted")], className="text-center"),
                ], className="mb-3"),
                html.Pre("\n".join(ascii_lines), className="text-center font-monospace small mb-2"),
                html.P([
                    "Legend: 🔥=Burning ⬛=Burned 🛡️=Protected ░=Safe"
                ], className="small text-center"),
            ])
        ], className="shadow-sm mt-3")
    
    # 5. Compare Strategies
    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Input('compare-button', 'n_clicks'),
        State('grid-graph-size-dropdown', 'value'),
        State('lambda-dropdown', 'value'),
        State('wind-speed-dropdown', 'value'),
        State('wind-direction-dropdown', 'value'),
        prevent_initial_call=True
    )
    def compare_strategies(n_clicks, grid_size, lambda_val, wind_speed, wind_dir):
        if not n_clicks:
            return no_update
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_val,
            num_firefighters=2,
            fire_start_nodes=[grid_size**2 // 2],
            seed=42,
            wind_speed=wind_speed,
            wind_direction=wind_dir
        )
        
        simulator = FireSpreadSimulator(config)
        comparison = simulator.compare_strategies()
        
        best = min(comparison.keys(), key=lambda s: comparison[s].total_burned)
        
        rows = []
        for strategy, res in comparison.items():
            is_best = strategy == best
            rows.append(dbc.Row([
                dbc.Col([html.Strong(strategy.capitalize()), is_best and html.Badge(" BEST", "bg-success ms-2") or ""], width=4),
                dbc.Col(f"{res.total_burned} burned", width=3),
                dbc.Col(f"{res.total_protected} protected", width=3),
                dbc.Col(f"{res.time_steps} steps", width=2),
            ], className=f"py-2 {'bg-success bg-opacity-10' if is_best else ''}"))
        
        return dbc.Card([
            dbc.CardHeader("📊 Strategy Comparison"),
            dbc.CardBody([
                html.P(f"Grid: {grid_size}×{grid_size}, λ={lambda_val}, Wind: {wind_speed}km/h {wind_dir}", className="small text-muted"),
                html.Hr(),
                html.Div(rows, className="mt-2"),
                dbc.Alert(f"💡 {best.capitalize()} strategy burns fewest nodes ({comparison[best].total_burned})", color="success", className="mt-2 mb-0")
            ])
        ], className="shadow-sm mt-3")
    
    # 6. Evacuation Plan
    @app.callback(
        Output('results-output', 'children', allow_duplicate=True),
        Input('evacuate-button', 'n_clicks'),
        State('selected-fire-point', 'data'),
        State('wind-speed-dropdown', 'value'),
        State('wind-direction-dropdown', 'value'),
        prevent_initial_call=True
    )
    def get_evacuation_plan(n_clicks, selected_point, wind_speed, wind_dir):
        if not n_clicks or not selected_point:
            return dbc.Alert("⚠️ Select a fire point first!", color="warning")
        
        lat, lon = selected_point['lat'], selected_point['lon']
        
        # Get fire station coverage
        coverage = get_coverage_status(lat, lon)
        
        # Evacuation zones based on wind
        downwind_dist = min(wind_speed * 0.5, 30)  # km
        
        zones = [
            {"zone": "CRITICAL", "dist": downwind_dist, "action": "IMMEDIATE EVACUATE"},
            {"zone": "WARNING", "dist": downwind_dist * 1.5, "action": "PREPARE TO LEAVE"},
            {"zone": "ALERT", "dist": downwind_dist * 2, "action": "STAY INFORMED"},
        ]
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🚨 Evacuation Plan", className="mb-0 d-inline"),
                html.Span(f" 🌬️ {wind_dir} {wind_speed}km/h", className="badge bg-danger ms-2")
            ]),
            dbc.CardBody([
                html.H6(f"📍 Fire at {lat:.4f}°, {abs(lon):.4f}°", className="mb-3"),
                html.Hr(),
                html.H6("🚒 Dispatch Recommendations"),
                coverage['nearest_station'] and dbc.Alert([
                    html.Strong(f"Nearest: {coverage['nearest_station']}"),
                    html.Br(),
                    f"Response: {coverage['best_response_time']} min | ",
                    f"Trucks: {coverage['total_available_trucks']}"
                ], color="info", className="mb-3") or dbc.Alert("No stations nearby!", color="warning"),
                html.Hr(),
                html.H6("📍 Evacuation Zones (downwind)"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong(z["zone"]),
                        html.Br(),
                        f"Within {z['dist']:.0f} km",
                        html.Br(),
                        html.Strong(z["action"], className="text-danger")
                    ], color={"CRITICAL": "danger", "WARNING": "warning", "ALERT": "secondary"}.get(z["zone"]))
                    for z in zones
                ], className="mb-3"),
                html.Hr(),
                html.P([
                    "💡 Fire will spread ", html.Strong(f"{downwind_dist:.0f} km downwind"),
                    " in the next few hours based on current wind conditions."
                ], className="small text-muted")
            ])
        ], className="shadow-sm mt-3 border-danger")
    
    # 7. Demo Button
    @app.callback(
        [Output('fire-map', 'figure', allow_duplicate=True),
         Output('selection-status', 'children', allow_duplicate=True),
         Output('selected-fire-point', 'data', allow_duplicate=True),
         Output('country-dropdown', 'value', allow_duplicate=True)],
        Input('demo-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def load_demo(n_clicks):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        
        try:
            df = get_fire_data(api_key, "110,-40,160,-10", day_range=3)
            
            if df.empty:
                sample_data = [
                    {"latitude": -21.0, "longitude": 116.8, "brightness": 326, "frp": 50},
                    {"latitude": -35.6, "longitude": 138.1, "brightness": 355, "frp": 135},
                    {"latitude": -26.4, "longitude": 126.3, "brightness": 397, "frp": 75},
                ]
                df = pd.DataFrame(sample_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='markers',
                marker=dict(size=10, color='red', opacity=0.7),
                text=[f"🔥 {row['latitude']:.2f}°, {row['longitude']:.2f}°" for _, row in df.iterrows()],
                hoverinfo='text',
                name=f'Active Fires ({len(df)})'
            ))
            
            selected_lat = df.iloc[0]['latitude']
            selected_lon = df.iloc[0]['longitude']
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=-25, lon=133), zoom=4),
                margin=dict(l=10, r=10, t=10, b=10),
                height=480
            )
            
            return fig, f"📍 Demo: (-21.0, 116.8)", {'lat': selected_lat, 'lon': selected_lon}, 'AU'
            
        except Exception as e:
            return no_update, f"Error: {e}", no_update, no_update
