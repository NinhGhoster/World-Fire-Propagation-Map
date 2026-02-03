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
from flask import Flask

from config import Config, get_config
from modules.logger import setup_logging
from modules.layout import create_layout
from modules.callbacks import register_callbacks
from modules.api import create_api_app

# Initialize logging
logger = setup_logging(__name__)


def create_app(debug: bool = False) -> dash.Dash:
    """
    Create and configure the Dash application.
    
    Args:
        debug: Enable debug mode
    
    Returns:
        Configured Dash application
    """
    # Validate configuration
    config = get_config()
    is_valid, errors = config.validate()
    
    if not is_valid:
        for error in errors:
            logger.warning(f"Configuration warning: {error}")
    
    # Create Flask server
    server = Flask(__name__)
    server.config["DEBUG"] = debug
    
    # Health check endpoints
    @server.route("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": Config.VERSION,
            "name": Config.APP_NAME
        }, 200
    
    @server.route("/ready")
    def readiness_check():
        return {"status": "ready"}, 200
    
    @server.route("/version")
    def version_check():
        return {
            "version": Config.VERSION,
            "name": Config.APP_NAME,
            "debug": Config.DEBUG
        }, 200
    
    # Mount REST API on /api/*
    api_app = create_api_app()
    server.wsgi_app = api_app.wsgi_app  # Mount the API app
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP
        ],
        title=Config.APP_NAME,
        suppress_callback_exceptions=True
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
    # Validate configuration first
    config = get_config()
    is_valid, errors = config.validate()
    
    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Create and run app
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
