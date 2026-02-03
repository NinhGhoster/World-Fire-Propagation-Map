"""
World Fire Propagation Map - Main Application Entry Point

Production-ready Dash application with REST API support.
"""
import os
import sys
from pathlib import Path

# Add project root to path
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

# Initialize logging
logger = setup_logging(__name__)


def create_app(debug: bool = False) -> dash.Dash:
    """
    Create and configure the Dash application with integrated API.
    """
    config = get_config()
    is_valid, errors = config.validate()
    
    if not is_valid:
        for error in errors:
            logger.warning(f"Configuration warning: {error}")
    
    # Create Flask server
    server = Flask(__name__)
    server.config["DEBUG"] = debug
    
    # CORS headers for API
    @server.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
        return response
    
    # Health endpoints
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
        """Get fire data for a given area."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        radius = request.args.get("radius", 40, type=float)
        source = request.args.get("source", "MODIS_NRT")
        days = request.args.get("days", 3, type=int)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing required parameters", "required": ["lat", "lon"]}), 400
        
        # Calculate bounding box
        from modules.analysis_pipeline import point_to_boundary
        boundary_str, bounds = point_to_boundary(lat, lon, radius_km=radius)
        
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str, source=source, day_range=days)
            
            if df.empty:
                return jsonify({"message": "No fires found", "count": 0, "data": []})
            
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
        """Get list of supported countries."""
        fetcher = DataFetcher(Config.FIRMS_API_KEY)
        try:
            df = fetcher.get_country_list()
            countries = df[["abreviation", "name", "bbox_coords"]].to_dict(orient="records")
            return jsonify({"count": len(countries), "data": countries})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @server.route("/api/v1/analyze", methods=["POST", "OPTIONS"])
    def analyze_fire():
        """Analyze fire data for a given location and date."""
        from modules.analysis_pipeline import run_analysis_pipeline
        
        data = request.get_json() or {}
        
        lat = data.get("lat")
        lon = data.get("lon")
        date = data.get("date")
        grid_size = data.get("grid_size", 128)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing required parameters", "required": ["lat", "lon"]}), 400
        
        try:
            result = run_analysis_pipeline(
                lat=lat, lon=lon,
                selected_date=date or "2026-02-03",
                api_key=Config.FIRMS_API_KEY,
                grid_size=grid_size
            )
            return jsonify({"status": "success", "stats": result["stats"]})
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/simulate", methods=["POST", "OPTIONS"])
    def simulate_fire():
        """Run fire spread simulation."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        data = request.get_json() or {}
        
        grid_size = data.get("grid_size", 7)
        lambda_spread = data.get("lambda_spread", 0.005)
        firefighters = data.get("firefighters", 1)
        strategy = data.get("strategy", "greedy")
        start_node = data.get("start_node", grid_size ** 2 // 2)
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_spread,
            num_firefighters=firefighters,
            fire_start_nodes=[start_node],
            seed=data.get("seed")
        )
        
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy=strategy)
        
        return jsonify({
            "status": "success",
            "configuration": {
                "grid_size": grid_size,
                "lambda": lambda_spread,
                "firefighters": firefighters,
                "strategy": strategy
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
        """Compare all firefighter placement strategies."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        data = request.get_json() or {}
        
        grid_size = data.get("grid_size", 7)
        lambda_spread = data.get("lambda_spread", 0.005)
        firefighters = data.get("firefighters", 1)
        start_node = data.get("start_node", grid_size ** 2 // 2)
        
        config = SimulationConfig(
            grid_size=grid_size,
            lambda_spread=lambda_spread,
            num_firefighters=firefighters,
            fire_start_nodes=[start_node],
            seed=data.get("seed")
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
            "configuration": {
                "grid_size": grid_size,
                "lambda": lambda_spread,
                "firefighters": firefighters,
                "start_node": start_node
            },
            "results": results
        })
    
    @server.route("/api/v1/parameters", methods=["GET", "OPTIONS"])
    def get_parameters():
        """Get available simulation parameters."""
        return jsonify({
            "grid_sizes": [3, 5, 7, 9],
            "lambda_values": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "firefighters": list(range(1, 11)),
            "strategies": ["greedy", "random", "central"],
            "sources": ["MODIS_NRT", "VIIRS_NRT"]
        })
    
    # ========== DASH APP ==========
    
    # Create Dash app
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
    
    # Create layout
    logger.info("Creating application layout...")
    app.layout = create_layout(app)
    
    # Register callbacks
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
    """Main entry point."""
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
