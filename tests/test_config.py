"""
Unit tests for configuration module.
"""
import os
import pytest
from pathlib import Path


def test_config_defaults():
    """Test that configuration has sensible defaults."""
    # Clear environment variables
    for key in ["FIRMS_API_KEY", "DEBUG", "LOG_LEVEL"]:
        os.environ.pop(key, None)
    
    # Import after clearing env
    from config import Config
    
    assert Config.APP_NAME == "World Fire Propagation Map"
    assert Config.VERSION == "2.0.0"
    assert Config.FIRMS_API_KEY == ""
    assert Config.DEFAULT_GRID_SIZE == 128


def test_config_from_env():
    """Test configuration loads from environment variables."""
    os.environ["FIRMS_API_KEY"] = "test_api_key"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    from importlib import reload
    import config
    reload(config)
    
    assert config.Config.FIRMS_API_KEY == "test_api_key"
    assert config.Config.DEBUG is True
    assert config.Config.LOG_LEVEL == "DEBUG"
    
    # Cleanup
    os.environ.pop("FIRMS_API_KEY")
    os.environ.pop("DEBUG")
    os.environ.pop("LOG_LEVEL")


def test_config_validate():
    """Test configuration validation."""
    from config import Config
    
    # Without API key, should have error
    Config.FIRMS_API_KEY = ""
    is_valid, errors = Config.validate()
    assert is_valid is False
    assert len(errors) > 0
    assert "FIRMS_API_KEY" in errors[0]
    
    # With API key, should be valid
    Config.FIRMS_API_KEY = "valid_key"
    is_valid, errors = Config.validate()
    assert is_valid is True
    assert len(errors) == 0


def test_firms_url_builder():
    """Test FIRMS URL builder."""
    from config import Config
    
    url = Config.get_firms_url("10,20,30,40")
    assert "area/csv/" in url
    assert "10,20,30,40" in url
    
    # With date
    url_with_date = Config.get_firms_url("10,20,30,40", start_date="2024-01-01")
    assert "2024-01-01" in url_with_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
