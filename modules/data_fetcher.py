"""
World Fire Propagation Map - Data Fetcher Module

Fetches fire data from NASA FIRMS API with proper error handling and logging.
"""
import pandas as pd
import requests
from io import StringIO
from typing import Optional
from .logger import get_logger


logger = get_logger(__name__)


class FIRMSAPIError(Exception):
    """Custom exception for FIRMS API errors."""
    pass


class DataFetcher:
    """Handles data fetching from various APIs."""
    
    def __init__(self, api_key: str):
        """
        Initialize data fetcher.
        
        Args:
            api_key: NASA FIRMS API key
        """
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_fire_data(
        self,
        area_coords: str,
        source: str = "MODIS_NRT",
        day_range: int = 1,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetches active fire data from the FIRMS API.
        
        Args:
            area_coords: Bounding box coordinates (west,south,east,north)
            source: Data source (MODIS_NRT or VIIRS_NRT)
            day_range: Number of days to query
            start_date: Start date in YYYY-MM-DD format
        
        Returns:
            DataFrame with fire data or empty DataFrame on error
        """
        if not self.api_key:
            logger.error("FIRMS API key is not configured")
            raise FIRMSAPIError("FIRMS API key is required")
        
        url = self._build_url(area_coords, source, day_range, start_date)
        logger.info(f"Fetching fire data from FIRMS API: {source}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if not response.text or len(response.text.splitlines()) <= 1:
                logger.warning("FIRMS API returned empty or minimal data")
                return pd.DataFrame()
            
            csv_file = StringIO(response.text)
            df = pd.read_csv(csv_file)
            logger.info(f"Successfully fetched {len(df)} fire detections")
            return df
        
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
        except Exception as e:
            logger.exception(f"Error fetching FIRMS data: {e}")
            raise FIRMSAPIError(f"Failed to fetch data: {e}")
    
    def get_country_list(self) -> pd.DataFrame:
        """Fetch country list from FIRMS API."""
        url = "https://firms.modaps.eosdis.nasa.gov/api/countries/"
        try:
            df = pd.read_csv(url, sep=";")
            df['bbox_coords'] = df['extent'].str.extract(r'BOX\((.*)\)')[0].str.replace(' ', ',')
            logger.info(f"Loaded {len(df)} countries from FIRMS")
            return df
        except Exception as e:
            logger.error(f"Error fetching country list: {e}")
            raise FIRMSAPIError(f"Failed to fetch countries: {e}")
    
    def _build_url(self, area_coords: str, source: str, day_range: int, start_date: Optional[str]) -> str:
        base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.api_key}/{source}/{area_coords}/{day_range}"
        if start_date:
            base_url = f"{base_url}/{start_date}"
        return base_url
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError):
        """Handle HTTP errors with specific messages."""
        status_code = error.response.status_code if error.response else 0
        if status_code == 401:
            raise FIRMSAPIError("Authentication failed. Invalid API key.")
        elif status_code == 429:
            raise FIRMSAPIError("Rate limited. Please wait before retrying.")
        else:
            raise FIRMSAPIError(f"HTTP error {status_code}")


def get_fire_data(api_key: str, area_coords: str, source: str = "MODIS_NRT", day_range: int = 1, start_date: str = None) -> pd.DataFrame:
    """Fetch fire data from FIRMS API."""
    fetcher = DataFetcher(api_key)
    return fetcher.get_fire_data(area_coords, source, day_range, start_date)


def get_country_list() -> pd.DataFrame:
    """Fetch country list from FIRMS API."""
    fetcher = DataFetcher("")
    return fetcher.get_country_list()
