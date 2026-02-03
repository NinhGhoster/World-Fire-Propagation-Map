"""
Unit tests for data fetcher module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


class TestDataFetcher:
    """Tests for DataFetcher class."""
    
    def test_init_with_api_key(self):
        """Test DataFetcher initialization."""
        from modules.data_fetcher import DataFetcher, FIRMSAPIError
        
        fetcher = DataFetcher("test_api_key")
        assert fetcher.api_key == "test_api_key"
        assert fetcher.session is not None
    
    def test_get_fire_data_no_api_key(self):
        """Test that missing API key raises error."""
        from modules.data_fetcher import DataFetcher, FIRMSAPIError
        
        fetcher = DataFetcher("")
        with pytest.raises(FIRMSAPIError) as exc_info:
            fetcher.get_fire_data("10,20,30,40")
        assert "API key is required" in str(exc_info.value)
    
    @patch('modules.data_fetcher.requests.Session')
    def test_get_fire_data_success(self, mock_session_class):
        """Test successful fire data fetch."""
        from modules.data_fetcher import DataFetcher
        
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.text = """latitude,longitude,brightness
-35.0,140.0,320.5
-35.1,140.1,315.0"""
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = DataFetcher("valid_key")
        df = fetcher.get_fire_data("10,20,30,40", source="MODIS_NRT")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        mock_session.get.assert_called_once()
    
    @patch('modules.data_fetcher.requests.Session')
    def test_get_fire_data_empty_response(self, mock_session_class):
        """Test handling of empty response."""
        from modules.data_fetcher import DataFetcher
        
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.text = ""
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = DataFetcher("valid_key")
        df = fetcher.get_fire_data("10,20,30,40")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('modules.data_fetcher.requests.Session')
    def test_get_fire_data_http_error(self, mock_session_class):
        """Test HTTP error handling."""
        from modules.data_fetcher import DataFetcher, FIRMSAPIError
        import requests
        
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        error = requests.HTTPError(response=mock_response)
        mock_session.get.side_effect = error
        mock_session_class.return_value = mock_session
        
        fetcher = DataFetcher("valid_key")
        with pytest.raises(FIRMSAPIError) as exc_info:
            fetcher.get_fire_data("10,20,30,40")
        assert "Authentication failed" in str(exc_info.value)
    
    @patch('pandas.read_csv')
    def test_get_country_list_success(self, mock_read_csv):
        """Test country list fetch."""
        mock_df = pd.DataFrame({
            'abreviation': ['AU', 'US'],
            'name': ['Australia', 'United States'],
            'extent': ['BOX(110,-10,160,-40)', 'BOX(-125,25,-66,49)']
        })
        mock_read_csv.return_value = mock_df
        
        from modules.data_fetcher import DataFetcher
        
        fetcher = DataFetcher("test_key")
        df = fetcher.get_country_list()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "abreviation" in df.columns
        assert "name" in df.columns
        assert "bbox_coords" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
