"""
World Fire Propagation Map - Main Application Entry Point

Production-ready Dash application with proper error handling and logging.
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
    
    # Configure Flask
    server.config["DEBUG"] = debug
    
    # Add health check endpoint
    @server.route("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": Config.VERSION,
            "name": Config.APP_NAME
        }, 200
    
    @server.route("/ready")
    def readiness_check():
        """Kubernetes readiness probe."""
        # Could add more checks here (API connectivity, etc.)
        return {"status": "ready"}, 200
    
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
    
    # Set up routes
    @server.route("/")
    def index():
        """Serve the main application."""
        return app.index()
    
    # Create layout
    logger.info("Creating application layout...")
    app.layout = create_layout(app)
    
    # Register callbacks
    logger.info("Registering application callbacks...")
    try:
        # Get API key from config for callbacks
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
    logger.info(f"Listening on http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug
    )


if __name__ == "__main__":
    main()
