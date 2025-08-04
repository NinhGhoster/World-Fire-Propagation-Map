# modules/data_fetcher.py
import pandas as pd
import requests
from io import StringIO

def get_fire_data(api_key: str, area_coords: str, source: str = "MODIS_NRT", day_range: int = 1, start_date: str = None):
    """
    Fetches active fire data from the FIRMS API for a given area and optional date.
    """
    if not api_key:
        raise ValueError("A valid NASA FIRMS API key is required.")

    # Base URL using the /area/ endpoint
    base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{area_coords}/{day_range}"
    
    # Add the date to the URL if one is provided
    # The API expects YYYY-MM-DD format, which the date picker provides
    url = f"{base_url}/{start_date}" if start_date else base_url

    try:
        response = requests.get(url)
        response.raise_for_status()

        if not response.text or len(response.text.splitlines()) <= 1:
            return pd.DataFrame()

        csv_file = StringIO(response.text)
        df = pd.read_csv(csv_file)
        return df

    except requests.exceptions.HTTPError as e:
        # ... (error handling remains the same)
        print("--- Server Error Response ---")
        print(f"URL that caused error: {url}")
        print(e.response.text)
        print("---------------------------")
        if e.response.status_code == 401:
            raise Exception("Authentication failed. The server rejected the API key.")
        else:
            raise Exception(f"HTTP Error fetching data: {e}.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

def get_country_list():
    """
    Retrieves the list of all supported countries from the FIRMS API.
    """
    url = "https://firms.modaps.eosdis.nasa.gov/api/countries/"
    df = pd.read_csv(url, sep=";")
    df['bbox_coords'] = df['extent'].str.extract(r'BOX\((.*)\)')[0].str.replace(' ', ',')
    return df