# modules/analysis_pipeline.py
import pandas as pd
import numpy as np
import srtm
from scipy.interpolate import griddata
import requests

from .data_fetcher import get_fire_data

def point_to_boundary(lat, lon, radius_km=40):
    import math
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))
    west, south = lon - lon_offset, lat - lat_offset
    east, north = lon + lon_offset, lat + lat_offset
    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}", (west, south, east, north)

def fire_df_to_grid_array(df, bounds, grid_size=64):
    if df.empty:
        return np.zeros((grid_size, grid_size, 1)), ['fire_count']

    west, south, east, north = bounds
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    
    df_copy = df.copy()
    df_copy.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    numeric_cols = ['brightness', 'scan', 'track', 'bright_ti4', 'bright_ti5', 'frp']
    processed_columns = []
    for col in df_copy.columns:
        if col in ['latitude', 'longitude']:
            continue
        if col in numeric_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[col].fillna(0, inplace=True)
            processed_columns.append(col)
        elif pd.api.types.is_object_dtype(df_copy[col]):
            df_copy[col].fillna('unknown', inplace=True)
            df_copy[col] = df_copy[col].astype('category').cat.codes
            processed_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df_copy[col]):
             df_copy[col].fillna(0, inplace=True)
             processed_columns.append(col)

    grid_array = np.zeros((grid_size, grid_size, len(processed_columns)))
    count_grid = np.zeros((grid_size, grid_size))

    for _, row in df_copy.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if not (south <= lat <= north and west <= lon <= east):
            continue
        lat_idx = min(int((lat - south) / lat_step), grid_size - 1)
        lon_idx = min(int((lon - west) / lon_step), grid_size - 1)
        idx = (grid_size - 1 - lat_idx, lon_idx)
        count_grid[idx] += 1
        for ch_idx, col in enumerate(processed_columns):
            grid_array[idx[0], idx[1], ch_idx] = max(grid_array[idx[0], idx[1], ch_idx], row[col])

    final_grid = np.concatenate([count_grid[:, :, np.newaxis], grid_array], axis=2)
    final_channels = ['fire_count'] + processed_columns
    
    return final_grid, final_channels

def get_elevation_grid(bounds, grid_size=64):
    try:
        west, south, east, north = bounds
        elevation_data = srtm.get_data(leave_zipped=True)
        points, values = [], []
        lat_points, lon_points = np.linspace(south, north, 100), np.linspace(west, east, 100)
        for lat in lat_points:
            for lon in lon_points:
                elevation = elevation_data.get_elevation(lat, lon)
                if elevation is not None:
                    points.append([lat, lon])
                    values.append(elevation)
        if len(points) < 4:
            return np.zeros((grid_size, grid_size)), "No SRTM data found for this region."
        grid_lat, grid_lon = np.mgrid[south:north:complex(0, grid_size), west:east:complex(0, grid_size)]
        elevation_grid = griddata(points, values, (grid_lat, grid_lon), method='cubic', fill_value=0)
        return np.flipud(elevation_grid), "Success"
    except Exception as e:
        print(f"Elevation data error: {e}")
        return np.zeros((grid_size, grid_size)), f"An error occurred: {e}"

def get_weather_grids_from_openmeteo(lat, lon, selected_date, grid_size=64):
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {"latitude": lat, "longitude": lon, "start_date": selected_date, "end_date": selected_date, "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max", "timezone": "GMT"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get('daily', {})
        temp_max_grid = np.full((grid_size, grid_size), data.get('temperature_2m_max', [0])[0])
        precip_grid = np.full((grid_size, grid_size), data.get('precipitation_sum', [0])[0])
        wind_grid = np.full((grid_size, grid_size), data.get('wind_speed_10m_max', [0])[0])
        return {"Max Temperature": temp_max_grid, "Precipitation": precip_grid, "Max Wind Speed": wind_grid}, "Success"
    except Exception as e:
        return {}, f"Weather API Error: {e}"

def run_analysis_pipeline(lat, lon, selected_date, api_key, grid_size=128):
    """Main pipeline that accepts and uses the selected grid_size."""
    boundary_str, bounds = point_to_boundary(lat, lon)
    
    fire_df = get_fire_data(api_key, boundary_str, start_date=selected_date)
    fire_grid, fire_channels = fire_df_to_grid_array(fire_df, bounds, grid_size=grid_size)
    
    elevation_grid, elevation_status = get_elevation_grid(bounds, grid_size=grid_size)
    
    weather_grids, weather_status = get_weather_grids_from_openmeteo(lat, lon, selected_date, grid_size=grid_size)
    
    west, south, east, north = bounds
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    
    stats = {
        "grid_size": f"{grid_size}x{grid_size}",
        "cell_size_km": f"{abs(lat_step)*111:.1f} km x {abs(lon_step)*111*np.cos(np.radians(lat)):.1f} km",
        "total_fires": len(fire_df),
        "cells_with_fire": int(np.count_nonzero(fire_grid[:,:,0]))
    }
    
    return {
        "fire_grid": fire_grid,
        "fire_channels": fire_channels,
        "elevation_grid": elevation_grid,
        "elevation_status": elevation_status,
        "weather_grids": weather_grids,
        "weather_status": weather_status,
        "stats": stats
    }