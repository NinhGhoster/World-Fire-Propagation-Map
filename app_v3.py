"""
World Fire Propagation Map v3.0 - 100x Better

Production-ready Dash application with:
- Real-time fire tracking
- Weather integration
- Fire analytics (hotspots, seasonal patterns, risk assessment)
- Enhanced simulation with wind
- Modern UI with tabs
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc
from flask import Flask, jsonify, request

from config import Config, get_config
from modules.logger import setup_logging
from modules.layout import create_layout
from modules.callbacks import register_callbacks
from modules.data_fetcher import DataFetcher, FIRMSAPIError
from modules.simulation import FireSpreadSimulator, SimulationConfig
from modules.weather import WeatherAPI, get_fire_danger
from modules.analytics import FireAnalytics, analyze_fires

logger = setup_logging(__name__)


def create_app(debug: bool = False) -> dash.Dash:
    config = get_config()
    server = Flask(__name__)
    server.config["DEBUG"] = debug
    
    @server.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return response
    
    @server.route("/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "version": "3.0.0",
            "name": "World Fire Propagation Map",
            "features": ["fire_tracking", "weather", "analytics", "simulation"]
        })
    
    # ========== API ROUTES ==========
    
    @server.route("/api/v1/fires", methods=["GET"])
    def get_fires():
        """Get active fires for an area."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        radius = request.args.get("radius", 200, type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing lat, lon"}), 400
        
        from modules.analysis_pipeline import point_to_boundary
        boundary_str, bounds = point_to_boundary(lat, lon, radius_km=radius)
        
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str)
            
            return jsonify({
                "count": len(df),
                "bounds": bounds,
                "data": df.to_dict(orient="records") if not df.empty else []
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/weather", methods=["GET"])
    def get_weather():
        """Get current weather and fire danger."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing lat, lon"}), 400
        
        try:
            weather_api = WeatherAPI()
            weather = weather_api.get_current(lat, lon)
            danger = weather_api.calculate_fire_danger(weather)
            
            return jsonify({
                "weather": weather.to_dict(),
                "fire_danger": danger.to_dict()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/forecast", methods=["POST"])
    def get_forecast():
        """Get fire spread forecast."""
        data = request.get_json() or {}
        
        lat = data.get("lat")
        lon = data.get("lon")
        hours = data.get("hours", 24)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing lat, lon"}), 400
        
        try:
            weather_api = WeatherAPI()
            forecast = weather_api.get_forecast(lat, lon, hours)
            
            # Calculate fire danger for each forecast hour
            dangers = []
            for w in forecast:
                danger = weather_api.calculate_fire_danger(w)
                dangers.append({
                    "time": w.timestamp.isoformat(),
                    "danger": danger.rating,
                    "score": danger.score,
                    "fwi": danger.fwi
                })
            
            return jsonify({
                "location": {"lat": lat, "lon": lon},
                "hours": hours,
                "weather_forecast": [w.to_dict() for w in forecast],
                "fire_danger_forecast": dangers
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/analytics/hotspots", methods=["GET"])
    def get_hotspots():
        """Identify fire hotspots."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        radius = request.args.get("radius", 500, type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing lat, lon"}), 400
        
        try:
            from modules.analysis_pipeline import point_to_boundary
            boundary_str, _ = point_to_boundary(lat, lon, radius_km=radius)
            
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str)
            
            analytics = FireAnalytics()
            hotspots = analytics.identify_hotspots(
                df.to_dict(orient="records") if not df.empty else []
            )
            
            return jsonify({
                "hotspot_count": len(hotspots),
                "hotspots": [h.to_dict() for h in hotspots],
                "summary": analytics.get_summary()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/analytics/seasonal", methods=["GET"])
    def get_seasonal():
        """Get seasonal fire patterns."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        
        # For now, return regional patterns
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data("-180,-90,180,90")
            
            analytics = FireAnalytics()
            patterns = analytics.analyze_seasonal_patterns(
                df.to_dict(orient="records") if not df.empty else []
            )
            
            return jsonify({
                "patterns": [p.to_dict() for p in patterns]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/risk", methods=["GET"])
    def get_risk():
        """Get fire risk assessment."""
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Missing lat, lon"}), 400
        
        try:
            # Get fires
            from modules.analysis_pipeline import point_to_boundary
            boundary_str, _ = point_to_boundary(lat, lon, radius_km=100)
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data(boundary_str)
            fires = df.to_dict(orient="records") if not df.empty else []
            
            # Get weather
            weather_api = WeatherAPI()
            weather = weather_api.get_current(lat, lon)
            weather_data = weather.to_dict() if weather else {}
            
            # Assess risk
            analytics = FireAnalytics()
            risk = analytics.assess_risk(lat, lon, fires, weather_data)
            
            return jsonify(risk.to_dict())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @server.route("/api/v1/simulate", methods=["POST"])
    def simulate_fire():
        """Run fire spread simulation."""
        data = request.get_json() or {}
        
        config = SimulationConfig(
            grid_size=data.get("grid_size", 7),
            lambda_spread=data.get("lambda_spread", 0.1),
            num_firefighters=data.get("firefighters", 2),
            fire_start_nodes=[data.get("start_node", 24)],
            seed=data.get("seed", 42),
            wind_speed=data.get("wind_speed", 30),
            wind_direction=data.get("wind_direction", "NE")
        )
        
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy=data.get("strategy", "greedy"))
        
        return jsonify({
            "status": "success",
            "configuration": {
                "grid_size": config.grid_size,
                "lambda": config.lambda_spread,
                "firefighters": config.num_firefighters,
                "wind": {"speed": config.wind_speed, "direction": config.wind_direction}
            },
            "results": {
                "total_burned": result.total_burned,
                "total_protected": result.total_protected,
                "time_steps": result.time_steps
            }
        })
    
    @server.route("/api/v1/demo")
    def get_demo():
        """Get demo data for Australia."""
        try:
            fetcher = DataFetcher(Config.FIRMS_API_KEY)
            df = fetcher.get_fire_data("110,-40,160,-10")
            
            if df.empty:
                return jsonify({
                    "status": "demo",
                    "message": "No live data",
                    "data": [
                        {"latitude": -21.0, "longitude": 116.8, "brightness": 326, "frp": 50},
                        {"latitude": -35.6, "longitude": 138.1, "brightness": 355, "frp": 135},
                    ]
                })
            
            return jsonify({
                "status": "demo",
                "total_fires": len(df),
                "data": df.head(50).to_dict(orient="records")
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
        title="World Fire Propagation Map v3.0",
        suppress_callback_exceptions=True
    )
    
    logger.info("Creating v3.0 application layout...")
    app.layout = create_layout(app)
    
    logger.info("Registering callbacks...")
    try:
        register_callbacks(app, Config.FIRMS_API_KEY)
        logger.info("Callbacks registered")
    except Exception as e:
        logger.error(f"Callback error: {e}")
    
    logger.info("World Fire Propagation Map v3.0 initialized")
    
    return app


def main():
    config = get_config()
    app = create_app(debug=config.DEBUG)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8050))
    
    logger.info(f"Starting World Fire Propagation Map v3.0")
    logger.info(f"URL: http://{host}:{port}")
    logger.info(f"API: http://{host}:{port}/api/v1/")
    
    app.run(host=host, port=port, debug=config.DEBUG)


if __name__ == "__main__":
    main()
