"""
World Fire Propagation Map - Configuration Management

Production-ready configuration with environment variable support.
Copy .env.example to .env and configure your values.
"""
import os
from pathlib import Path


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = None) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float = None) -> float:
    """Get environment variable as float."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Config:
    """Application configuration class."""

    # Application
    APP_NAME = "World Fire Propagation Map"
    VERSION = "2.0.0"
    DEBUG = get_env_bool("DEBUG", False)
    LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
    
    # NASA FIRMS API
    FIRMS_API_KEY = get_env("FIRMS_API_KEY", "")
    FIRMS_DEFAULT_SOURCE = "MODIS_NRT"  # or "VIIRS_NRT"
    FIRMS_DAY_RANGE = 1
    
    # Map Settings
    DEFAULT_GRID_SIZE = 128
    DEFAULT_D_VALUE = 3
    DEFAULT_B_VALUE = 3
    DEFAULT_LAMBDA = 1.0
    
    # Map Display
    DEFAULT_MAP_ZOOM = 12
    GRAPH_GRID_SIZES = [3, 5, 7, 9]
    GRAPH_SPACING_OPTIONS = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    # MFF Solver Settings
    MFF_TIMEOUT_SECONDS = 60
    MFF_MAX_ITERATIONS = 1000
    
    # Logging
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Paths
    ASSETS_DIR = Path(__file__).parent / "assets"
    DATA_DIR = Path(__file__).parent / "data"
    LOGS_DIR = Path(__file__).parent / "logs"
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """Validate configuration. Returns (is_valid, errors)."""
        errors = []
        
        if not cls.FIRMS_API_KEY:
            errors.append("FIRMS_API_KEY is not set. Get one from NASA FIRMS API.")
        
        if cls.DEBUG and cls.LOG_LEVEL == "INFO":
            cls.LOG_LEVEL = "DEBUG"
        
        return len(errors) == 0, errors
    
    @classmethod
    def get_firms_url(cls, area_coords: str, source: str = None, day_range: int = None, start_date: str = None) -> str:
        """Build FIRMS API URL."""
        source = source or cls.FIRMS_DEFAULT_SOURCE
        day_range = day_range or cls.FIRMS_DAY_RANGE
        
        base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{cls.FIRMS_API_KEY}/{source}/{area_coords}/{day_range}"
        
        if start_date:
            base_url = f"{base_url}/{start_date}"
        
        return base_url


# Create instance for import
config = Config()


def get_config() -> Config:
    """Get configuration instance."""
    return config
