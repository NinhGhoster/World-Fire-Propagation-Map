"""
Unit tests for analysis pipeline module.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


class TestAnalysisPipeline:
    """Tests for analysis pipeline functions."""
    
    def test_point_to_boundary(self):
        """Test point to boundary conversion."""
        from modules.analysis_pipeline import point_to_boundary
        
        boundary_str, bounds = point_to_boundary(-35.0, 140.0)
        
        assert isinstance(boundary_str, str)
        assert len(bounds) == 4
        west, south, east, north = bounds
        
        assert west < east
        assert south < north
        assert -180 <= west <= 180
        assert -90 <= south <= 90
    
    def test_fire_df_to_grid_array_empty(self):
        """Test grid array creation with empty DataFrame."""
        from modules.analysis_pipeline import fire_df_to_grid_array
        
        bounds = (10.0, 20.0, 30.0, 40.0)
        grid, channels = fire_df_to_grid_array(pd.DataFrame(), bounds)
        
        assert isinstance(grid, np.ndarray)
        assert grid.shape[2] == 1
        assert channels == ['fire_count']
    
    def test_fire_df_to_grid_array_with_data(self):
        """Test grid array creation with fire data."""
        from modules.analysis_pipeline import fire_df_to_grid_array
        
        df = pd.DataFrame({
            'latitude': [-35.0, -35.1, -35.2],
            'longitude': [140.0, 140.1, 140.2],
            'brightness': [320.5, 315.0, 310.0],
            'frp': [50.0, 45.0, 40.0]
        })
        
        bounds = (139.9, -35.3, 140.3, -34.9)
        grid, channels = fire_df_to_grid_array(df, bounds, grid_size=32)
        
        assert isinstance(grid, np.ndarray)
        assert grid.shape[0] == 32
        assert grid.shape[1] == 32
        assert 'fire_count' in channels
        assert 'brightness' in channels
        assert 'frp' in channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
