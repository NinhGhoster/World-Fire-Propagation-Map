"""
World Fire Propagation Map - REST API Module

Production-ready REST API for the fire propagation dashboard.
"""
from flask import Flask, jsonify, request, Response
from functools import wraps
import json
from datetime import datetime
from typing import Callable, Any
from config import Config
from modules.logger import get_logger


logger = get_logger(__name__)


def create_api_app() -> Flask:
    """Create and configure the Flask API application."""
    app = Flask(__name__)
    
    # Add CORS headers
    @app.after_request
    def after_request(response: Response) -> Response:
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    # Request logging middleware
    @app.before_request
    def log_request():
        logger.debug(f"API Request: {request.method} {request.path}")
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error) -> tuple:
        return jsonify({"error": "Bad Request", "message": str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error) -> tuple:
        return jsonify({"error": "Not Found", "message": str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error) -> tuple:
        logger.error(f"Internal error: {error}")
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500
    
    # Health endpoints
    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "version": Config.VERSION,
            "name": Config.APP_NAME
        })
    
    @app.route("/ready")
    def ready():
        return jsonify({"status": "ready"})
    
    @app.route("/version")
    def version():
        return jsonify({
            "version": Config.VERSION,
            "name": Config.APP_NAME,
            "debug": Config.DEBUG
        })
    
    # API routes
    @app.route("/api/v1/fires", methods=["GET"])
    def get_fires():
        """
        Get fire data for a given area.
        
        Query parameters:
            lat: Latitude of center point
            lon: Longitude of center point
            radius: Search radius in km (default: 40)
            source: Data source (MODIS_NRT or VIIRS_NRT, default: MODIS_NRT)
            days: Number of days to look back (default: 1)
        """
        # Import here to avoid circular imports
        from modules.data_fetcher import DataFetcher, FIRMSAPIError
        
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        radius = request.args.get("radius", 40, type=float)
        source = request.args.get("source", "MODIS_NRT")
        days = request.args.get("days", 1, type=int)
        
        if lat is None or lon is None:
            return jsonify({
                "error": "Missing required parameters",
                "required": ["lat", "lon"]
            }), 400
        
        # Calculate bounding box
        from modules.analysis_pipeline import point_to_boundary
        boundary_str, bounds = point_to_boundary(lat, lon, radius_km=radius)
        
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str, source=source, day_range=days)
            
            if df.empty:
                return jsonify({
                    "message": "No fires found in the specified area",
                    "count": 0,
                    "data": []
                })
            
            # Convert to list of dicts
            fires = df.to_dict(orient="records")
            
            return jsonify({
                "count": len(fires),
                "bounds": bounds,
                "source": source,
                "data": fires
            })
            
        except FIRMSAPIError as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/v1/countries", methods=["GET"])
    def get_countries():
        """Get list of supported countries."""
        from modules.data_fetcher import DataFetcher
        
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_country_list()
            
            countries = df[["abreviation", "name", "bbox_coords"]].to_dict(orient="records")
            
            return jsonify({
                "count": len(countries),
                "data": countries
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/v1/analyze", methods=["POST"])
    def analyze_fire():
        """
        Analyze fire data for a given location and date.
        
        JSON body:
            lat: Latitude
            lon: Longitude
            date: Date in YYYY-MM-DD format
            grid_size: Grid size (default: 128)
        """
        from modules.analysis_pipeline import run_analysis_pipeline
        
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "Missing request body"}), 400
        
        lat = data.get("lat")
        lon = data.get("lon")
        date = data.get("date")
        grid_size = data.get("grid_size", 128)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing required parameters", "required": ["lat", "lon"]}), 400
        
        try:
            result = run_analysis_pipeline(
                lat=lat,
                lon=lon,
                selected_date=date or datetime.now().strftime("%Y-%m-%d"),
                api_key=Config.FIRMS_API_KEY,
                grid_size=grid_size
            )
            
            return jsonify({
                "status": "success",
                "stats": result["stats"],
                "message": "Analysis complete"
            })
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/v1/simulate", methods=["POST"])
    def simulate_fire():
        """
        Run fire spread simulation.
        
        JSON body:
            grid_size: Size of grid (default: 7)
            lambda_spread: Fire spread probability (default: 0.005)
            firefighters: Number of firefighters (default: 1)
            strategy: Strategy (greedy, random, central)
            start_node: Starting node index (default: center)
        """
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
    
    @app.route("/api/v1/compare", methods=["POST"])
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
    
    @app.route("/api/v1/parameters", methods=["GET"])
    def get_parameters():
        """Get available simulation parameters."""
        return jsonify({
            "grid_sizes": [3, 5, 7, 9],
            "lambda_values": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "firefighters": list(range(1, 11)),
            "strategies": ["greedy", "random", "central"],
            "sources": ["MODIS_NRT", "VIIRS_NRT"]
        })
    
    return app


# Create the API app instance
api_app = create_api_app()


if __name__ == "__main__":
    api_app.run(host="0.0.0.0", port=8051, debug=True)
