"""
World Fire Propagation Map - Main Application Entry Point

Production-ready Dash application with REST API support.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc
from flask import Flask, jsonify, request, Response

from config import Config, get_config
from modules.logger import setup_logging
from modules.layout import create_layout
from modules.callbacks import register_callbacks
from modules.data_fetcher import DataFetcher, FALLBACK_COUNTRIES, FIRMSAPIError
from modules.simulation import FireSpreadSimulator, SimulationConfig

logger = setup_logging(__name__)


def create_app(debug: bool = False) -> dash.Dash:
    config = get_config()
    is_valid, errors = config.validate()
    
    if not is_valid:
        for error in errors:
            logger.warning(f"Configuration warning: {error}")
    
    server = Flask(__name__)
    server.config["DEBUG"] = debug
    
    @server.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
        return response
    
    @server.route("/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "version": Config.VERSION,
            "name": Config.APP_NAME
        })
    
    @server.route("/ready")
    def readiness_check():
        return jsonify({"status": "ready"})
    
    @server.route("/version")
    def version_check():
        return jsonify({
            "version": Config.VERSION,
            "name": Config.APP_NAME,
            "debug": Config.DEBUG
        })
    
    # ========== API ROUTES ==========
    
    @server.route("/api/v1/fires", methods=["GET", "OPTIONS"])
    def get_fires():
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        radius = request.args.get("radius", 200, type=float)
        source = request.args.get("source", "MODIS_NRT")
        days = request.args.get("days", 3, type=int)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing required parameters", "required": ["lat", "lon"]}), 400
        
        from modules.analysis_pipeline import point_to_boundary
        boundary_str, bounds = point_to_boundary(lat, lon, radius_km=radius)
        
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str, source=source, day_range=days)
            
            if df.empty:
                return jsonify({
                    "message": "No fires found in the specified area",
                    "count": 0,
                    "data": [],
                    "bounds": bounds,
                    "source": source,
                    "tip": "Try selecting a different location or increasing the radius"
                })
            
            fires = df.to_dict(orient="records")
            return jsonify({
                "count": len(fires),
                "bounds": bounds,
                "source": source,
                "data": fires
            })
        except FIRMSAPIError as e:
            return jsonify({"error": str(e)}), 400
    
    @server.route("/api/v1/countries", methods=["GET", "OPTIONS"])
    def get_countries():
        fetcher = DataFetcher(Config.FIRMS_API_KEY)
        try:
            df = fetcher.get_country_list()
            countries = df[["abreviation", "name", "bbox_coords"]].to_dict(orient="records")
            return jsonify({"count": len(countries), "data": countries})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @server.route("/api/v1/analyze", methods=["POST", "OPTIONS"])
    def analyze_location():
        from modules.analysis_pipeline import run_analysis_pipeline
        
        data = request.get_json() or {}
        
        lat = data.get("lat")
        lon = data.get("lon")
        date_param = data.get("date")
        radius = data.get("radius", 100)
        grid_size = data.get("grid_size", 64)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing required parameters", "required": ["lat", "lon"]}), 400
        
        try:
            result = run_analysis_pipeline(
                lat=lat, lon=lon,
                selected_date=date_param or "2026-02-03",
                api_key=Config.FIRMS_API_KEY,
                grid_size=grid_size
            )
            
            if result['stats']['total_fires'] == 0:
                result['message'] = "No fires detected in this area. Try a different location or use Load Example 1."
            
            return jsonify({
                "status": "success",
                "location": {"lat": lat, "lon": lon},
                "stats": result['stats'],
                "message": result.get('message', '')
            })
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/simulate", methods=["POST", "OPTIONS"])
    def simulate_fire():
        """Run fire spread simulation with wind."""
        data = request.get_json() or {}
        
        grid_size = data.get("grid_size", 7)
        lambda_spread = data.get("lambda_spread", 0.1)
        firefighters = data.get("firefighters", 2)
        strategy = data.get("strategy", "greedy")
        start_node = data.get("start_node", grid_size ** 2 // 2)
        
        # Wind parameters
        wind_speed = data.get("wind_speed", 30.0)
        wind_direction = data.get("wind_direction", "NE")
        
        # Weather parameters
        temperature = data.get("temperature", 25.0)
        humidity = data.get("humidity", 30.0)
        vegetation_dryness = data.get("vegetation_dryness", 0.8)
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_spread,
            num_firefighters=firefighters,
            fire_start_nodes=[start_node],
            seed=data.get("seed", 42),
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            temperature=temperature,
            humidity=humidity,
            vegetation_dryness=vegetation_dryness
        )
        
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy=strategy)
        
        return jsonify({
            "status": "success",
            "configuration": {
                "grid_size": grid_size,
                "lambda": lambda_spread,
                "firefighters": firefighters,
                "strategy": strategy,
                "wind": {
                    "speed_kmh": wind_speed,
                    "direction": wind_direction
                },
                "weather": {
                    "temperature_c": temperature,
                    "humidity_percent": humidity,
                    "vegetation_dryness": vegetation_dryness
                }
            },
            "results": {
                "total_burned": result.total_burned,
                "total_protected": result.total_protected,
                "time_steps": result.time_steps,
                "burned_nodes": result.burned_nodes,
                "protected_nodes": result.protected_nodes,
                "firefighter_placements": result.firefighter_placements
            }
        })
    
    @server.route("/api/v1/compare", methods=["POST", "OPTIONS"])
    def compare_strategies():
        """Compare all firefighter placement strategies with wind."""
        data = request.get_json() or {}
        
        grid_size = data.get("grid_size", 7)
        lambda_spread = data.get("lambda_spread", 0.1)
        firefighters = data.get("firefighters", 2)
        start_node = data.get("start_node", grid_size ** 2 // 2)
        
        # Wind parameters
        wind_speed = data.get("wind_speed", 30.0)
        wind_direction = data.get("wind_direction", "NE")
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_spread,
            num_firefighters=firefighters,
            fire_start_nodes=[start_node],
            seed=42,
            wind_speed=wind_speed,
            wind_direction=wind_direction
        )
        
        simulator = FireSpreadSimulator(config)
        comparison = simulator.compare_strategies()
        
        results = {}
        for strategy, res in comparison.items():
            results[strategy] = {
                "burned": res.total_burned,
                "protected": res.total_protected,
                "time_steps": res.time_steps
            }
        
        return jsonify({
            "status": "success",
            "configuration": {
                "grid_size": grid_size,
                "lambda": lambda_spread,
                "firefighters": firefighters,
                "start_node": start_node,
                "wind": {"speed_kmh": wind_speed, "direction": wind_direction}
            },
            "results": results
        })
    
    @server.route("/api/v1/forecast", methods=["POST", "OPTIONS"])
    def forecast_spread():
        """
        Predict fire spread given weather conditions.
        Returns expected burn area and recommended firefighter positions.
        """
        data = request.get_json() or {}
        
        lat = data.get("lat")
        lon = data.get("lon")
        hours = data.get("hours", 24)
        
        wind_speed = data.get("wind_speed", 30.0)
        wind_direction = data.get("wind_direction", "NE")
        temperature = data.get("temperature", 25.0)
        humidity = data.get("humidity", 30.0)
        
        # Convert hours to approximate time steps (1 step ≈ 15 min)
        time_steps = int(hours * 4)
        grid_size = min(15, max(7, time_steps // 4))
        
        # Estimate spread based on conditions
        base_lambda = 0.1
        wind_factor = 1.0 + (wind_speed / 25.0)
        temp_factor = 1.0 + (temperature - 25.0) / 50.0
        humidity_factor = 1.0 - (humidity - 30.0) / 200.0
        
        estimated_lambda = base_lambda * wind_factor * temp_factor * humidity_factor
        
        # Run simulation
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=estimated_lambda,
            num_firefighters=3,
            fire_start_nodes=[grid_size ** 2 // 2],
            seed=42,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            temperature=temperature,
            humidity=humidity
        )
        
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy="greedy")
        
        # Calculate burn area
        cell_size_km = 1.0  # Approximate
        burn_area_km2 = result.total_burned * (cell_size_km ** 2)
        
        return jsonify({
            "status": "success",
            "location": {"lat": lat, "lon": lon},
            "forecast": {
                "hours": hours,
                "time_steps": time_steps,
                "estimated_burn_area_km2": burn_area_km2,
                "nodes_burned": result.total_burned,
                "nodes_protected": result.total_protected,
                "containment_time_steps": result.time_steps
            },
            "conditions": {
                "wind_speed_kmh": wind_speed,
                "wind_direction": wind_direction,
                "temperature_c": temperature,
                "humidity_percent": humidity,
                "effective_spread_rate": round(estimated_lambda, 3)
            },
            "recommendations": {
                "firefighters_needed": max(2, result.total_burned // 10),
                "priority_directions": [wind_direction, 
                    {"N": "S", "NE": "SW", "E": "W", "SE": "NW",
                     "S": "N", "SW": "NE", "W": "E", "NW": "SE"}[wind_direction]],
                "evacuation_zones": ["within 5km downwind"]
            }
        })
    
    @server.route("/api/v1/parameters", methods=["GET", "OPTIONS"])
    def get_parameters():
        return jsonify({
            "grid_sizes": [3, 5, 7, 9, 11, 13, 15],
            "lambda_values": [0.05, 0.1, 0.2, 0.3, 0.5],
            "firefighters": list(range(1, 8)),
            "strategies": ["greedy", "random", "central"],
            "sources": ["MODIS_NRT", "VIIRS_NRT"],
            "wind_speeds": [0, 15, 30, 50, 70, 100],
            "wind_directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        })
    
    @server.route("/api/v1/demo", methods=["GET", "OPTIONS"])
    def get_demo():
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data("110,-40,160,-10", source="MODIS_NRT", day_range=3)
            
            if df.empty:
                return jsonify({
                    "status": "demo",
                    "message": "No live data available, showing demo data",
                    "data": [
                        {"latitude": -21.0, "longitude": 116.8, "brightness": 326, "frp": 50},
                        {"latitude": -35.6, "longitude": 138.1, "brightness": 355, "frp": 135},
                        {"latitude": -26.4, "longitude": 126.3, "brightness": 397, "frp": 75},
                    ]
                })
            
            sample = df.head(20).to_dict(orient="records")
            return jsonify({
                "status": "demo",
                "message": f"Showing {len(sample)} of {len(df)} total fires",
                "data": sample,
                "total_fires": len(df)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # ========== DASH APP ==========
    
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP
        ],
        title=Config.APP_NAME,
        suppress_callback_exceptions=True,
        assets_folder=str(PROJECT_ROOT / "assets")
    )
    
    logger.info("Creating application layout...")
    app.layout = create_layout(app)
    
    logger.info("Registering application callbacks...")
    try:
        api_key = config.FIRMS_API_KEY
        register_callbacks(app, api_key)
        logger.info("Callbacks registered successfully")
    except Exception as e:
        logger.error(f"Error registering callbacks: {e}")
        raise
    
    logger.info(f"{Config.APP_NAME} v{Config.VERSION} initialized")
    
    return app


def main():
    config = get_config()
    is_valid, errors = config.validate()
    
    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    debug = config.DEBUG
    app = create_app(debug=debug)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8050))
    
    logger.info(f"Starting {Config.APP_NAME} v{Config.VERSION}")
    logger.info(f"Dashboard: http://{host}:{port}")
    logger.info(f"API:      http://{host}:{port}/api/v1/")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug
    )


if __name__ == "__main__":
    main()
