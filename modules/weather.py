"""
Weather Integration Module

Real-time weather data for fire danger assessment.
Integrates with OpenWeatherMap API.
"""
import os
import requests
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

# Cache for weather data
WEATHER_CACHE = {}
CACHE_DURATION = 300  # 5 minutes


@dataclass
class WeatherData:
    """Current weather conditions."""
    temperature: float  # Celsius
    humidity: float  # Percentage
    wind_speed: float  # km/h
    wind_direction: str  # N, NE, E, SE, S, SW, W, NW
    wind_gust: float  # km/h
    pressure: float  # hPa
    clouds: int  # Percentage
    precipitation: float  # mm
    visibility: float  # km
    uv_index: int
    description: str
    icon: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "temperature_c": self.temperature,
            "humidity_percent": self.humidity,
            "wind_speed_kmh": self.wind_speed,
            "wind_direction": self.wind_direction,
            "wind_gust_kmh": self.wind_gust,
            "pressure_hpa": self.pressure,
            "clouds_percent": self.clouds,
            "precipitation_mm": self.precipitation,
            "visibility_km": self.visibility,
            "uv_index": self.uv_index,
            "description": self.description,
            "icon": self.icon,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class FireDangerRating:
    """Fire danger assessment based on weather."""
    rating: str  # LOW, MODERATE, HIGH, VERY_HIGH, EXTREME
    score: float  # 0-100
    fwi: float  # Fire Weather Index
    ffmc: float  # Fine Fuel Moisture Code
    dmc: float  # Duff Moisture Code
    dc: float  # Drought Code
    isi: float  # Initial Spread Index
    bui: float  # Buildup Index
    
    def to_dict(self) -> Dict:
        return {
            "rating": self.rating,
            "score": self.score,
            "fwi": self.fwi,
            "ffmc": self.ffmc,
            "dmc": self.dmc,
            "dc": self.dc,
            "isi": self.isi,
            "bui": self.bui
        }


class WeatherAPI:
    """
    Weather API wrapper with caching and fire danger calculation.
    """
    
    BASE_URL = "http://api.openweathermap.org/data/2.5"
    ONE_CALL_URL = "http://api.openweathermap.org/data/3.0/onecall"
    
    DIRECTION_MAP = {
        (0, 1): "N", (0.707, 0.707): "NE", (1, 0): "E", (0.707, -0.707): "SE",
        (0, -1): "S", (-0.707, -0.707): "SW", (-1, 0): "W", (-0.707, 0.707): "NW"
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
    
    def _get_cache_key(self, lat: float, lon: float) -> str:
        return f"{lat:.4f}_{lon:.4f}"
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to compass direction."""
        # Normalize to 0-360
        degrees = degrees % 360
        # Find closest direction
        rad = degrees * 3.14159 / 180
        x, y = -sin(rad), cos(rad)  # OpenWeatherMap uses meteorological convention
        for (dx, dy), direction in self.DIRECTION_MAP.items():
            if abs(x - dx) < 0.25 and abs(y - dy) < 0.25:
                return direction
        return "E"
    
    def get_current(self, lat: float, lon: float, use_cache: bool = True) -> Optional[WeatherData]:
        """
        Get current weather for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Use cached data if available
            
        Returns:
            WeatherData object or None if failed
        """
        cache_key = self._get_cache_key(lat, lon)
        
        if use_cache and cache_key in WEATHER_CACHE:
            cached = WEATHER_CACHE[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < CACHE_DURATION:
                return cached['data']
        
        if not self.api_key:
            # Return simulated data for development
            return self._get_simulated_weather(lat, lon)
        
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/weather",
                params=params,
                timeout=10
            )
            
            if response.status_code == 401:
                print("⚠️ Invalid OpenWeatherMap API key")
                return self._get_simulated_weather(lat, lon)
            
            response.raise_for_status()
            data = response.json()
            
            wind_deg = data.get('wind', {}).get('deg', 0)
            weather = data.get('weather', [{}])[0]
            
            weather_obj = WeatherData(
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'] * 3.6,  # m/s to km/h
                wind_direction=self._get_wind_direction(wind_deg),
                wind_gust=data.get('wind', {}).get('gust', 0) * 3.6 if 'gust' in data.get('wind', {}) else 0,
                pressure=data['main']['pressure'],
                clouds=data.get('clouds', {}).get('all', 0),
                precipitation=data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
                visibility=data.get('visibility', 10000) / 1000,
                uv_index=0,  # Not in current weather endpoint
                description=weather.get('description', ''),
                icon=weather.get('icon', ''),
                timestamp=datetime.now()
            )
            
            WEATHER_CACHE[cache_key] = {
                'data': weather_obj,
                'timestamp': datetime.now()
            }
            
            return weather_obj
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_simulated_weather(lat, lon)
    
    def get_forecast(self, lat: float, lon: float, hours: int = 24) -> List[WeatherData]:
        """
        Get weather forecast for specified hours.
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Forecast horizon (max 48 hours)
            
        Returns:
            List of WeatherData objects
        """
        if not self.api_key:
            return self._get_simulated_forecast(lat, lon, hours)
        
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "exclude": "minutely"
            }
            
            response = self.session.get(
                f"{self.ONE_CALL_URL}",
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for hourly in data.get('hourly', [])[:hours]:
                wind_deg = hourly.get('wind_deg', 0)
                weather = hourly.get('weather', [{}])[0]
                
                forecast = WeatherData(
                    temperature=hourly['temp'],
                    humidity=hourly['humidity'],
                    wind_speed=hourly['wind_speed'] * 3.6,
                    wind_direction=self._get_wind_direction(wind_deg),
                    wind_gust=hourly.get('wind_gust', 0) * 3.6 if 'wind_gust' in hourly else 0,
                    pressure=hourly['pressure'],
                    clouds=hourly.get('clouds', 0),
                    precipitation=hourly.get('rain', {}).get('1h', 0) + hourly.get('snow', {}).get('1h', 0),
                    visibility=10,  # Not in hourly
                    uv_index=hourly.get('uvi', 0),
                    description=weather.get('description', ''),
                    icon=weather.get('icon', ''),
                    timestamp=datetime.fromtimestamp(hourly['dt'])
                )
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            print(f"Weather forecast error: {e}")
            return self._get_simulated_forecast(lat, lon, hours)
    
    def calculate_fire_danger(self, weather: WeatherData) -> FireDangerRating:
        """
        Calculate fire danger rating based on weather conditions.
        Implements a simplified FWI (Fire Weather Index) system.
        """
        # Fine Fuel Moisture Code (FFMC) - daily
        ffmc = self._calculate_ffmc(weather)
        
        # Duff Moisture Code (DMC) - 10-day
        dmc = self._calculate_dmc(weather)
        
        # Drought Code (DC) - 52-day
        dc = self._calculate_dc(weather)
        
        # Initial Spread Index (ISI)
        isi = self._calculate_isi(weather, ffmc)
        
        # Buildup Index (BUI)
        bui = self._calculate_bui(dmc, dc)
        
        # Fire Weather Index (FWI) - overall
        fwi = self._calculate_fwi(isi, bui)
        
        # Rating based on FWI
        if fwi < 5:
            rating = "LOW"
            score = fwi * 2
        elif fwi < 12:
            rating = "MODERATE"
            score = 10 + (fwi - 5) * 10
        elif fwi < 24:
            rating = "HIGH"
            score = 30 + (fwi - 12) * 5
        elif fwi < 50:
            rating = "VERY_HIGH"
            score = 70 + (fwi - 24) * 1
        else:
            rating = "EXTREME"
            score = min(100, 96 + (fwi - 50) * 0.1)
        
        return FireDangerRating(
            rating=rating,
            score=score,
            fwi=fwi,
            ffmc=ffmc,
            dmc=dmc,
            dc=dc,
            isi=isi,
            bui=bui
        )
    
    def _calculate_ffmc(self, weather: WeatherData) -> float:
        """Calculate Fine Fuel Moisture Code (0-99)."""
        # Simplified calculation
        temp = weather.temperature
        humidity = weather.humidity
        wind = weather.wind_speed
        
        # Drying effect of temperature and wind
        drying = (temp / 30) + (wind / 50)
        # Moisture effect of humidity
        moisture = humidity / 100
        
        ffmc = 85 + (moisture - 0.3) * 50 - drying * 10
        return max(0, min(99, ffmc))
    
    def _calculate_dmc(self, weather: WeatherData) -> float:
        """Calculate Duff Moisture Code (0-100+)."""
        temp = weather.temperature
        humidity = weather.humidity
        precip = weather.precipitation
        
        # DMC responds slower than FFMC
        drying = (temp - 10) / 20 + humidity / 200
        dmc = 6 * drying + precip * 0.5
        return max(0, min(100, dmc))
    
    def _calculate_dc(self, weather: WeatherData) -> float:
        """Calculate Drought Code (0-500+)."""
        temp = weather.temperature
        precip = weather.precipitation
        
        # DC is very slow to change
        drying = (temp - 5) / 50
        dc = 50 * drying + precip * 0.3
        return max(0, min(500, dc))
    
    def _calculate_isi(self, weather: WeatherData, ffmc: float) -> float:
        """Calculate Initial Spread Index (0-50+)."""
        wind = weather.wind_speed
        # ISI increases with wind and lower moisture
        isi = (wind / 10) * ((100 - ffmc) / 50)
        return max(0, min(50, isi))
    
    def _calculate_bui(self, dmc: float, dc: float) -> float:
        """Calculate Buildup Index (0-100+)."""
        if dmc <= 0 and dc <= 0:
            return 0
        # BUI is weighted average of DMC and DC
        bui = 0.8 * dmc + 0.2 * dc
        return max(0, min(100, bui))
    
    def _calculate_fwi(self, isi: float, bui: float) -> float:
        """Calculate Fire Weather Index (0-100+)."""
        if isi <= 0 or bui <= 0:
            return 0
        fwi = isi * (bui / 100)
        return max(0, min(100, fwi))
    
    def _get_simulated_weather(self, lat: float, lon: float) -> WeatherData:
        """Generate simulated weather for development."""
        import random
        import math
        
        # Vary based on latitude
        base_temp = 30 - abs(lat) / 3
        temp = base_temp + random.uniform(-5, 10)
        
        return WeatherData(
            temperature=round(temp, 1),
            humidity=int(30 + random.uniform(0, 40)),
            wind_speed=round(random.uniform(5, 40), 1),
            wind_direction=["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(random.uniform(0, 8))],
            wind_gust=round(random.uniform(10, 60), 1),
            pressure=int(1000 + random.uniform(-20, 20)),
            clouds=int(random.uniform(0, 50)),
            precipitation=round(random.uniform(0, 2), 1),
            visibility=round(random.uniform(8, 15), 1),
            uv_index=int(random.uniform(3, 9)),
            description="Clear sky" if random.random() > 0.3 else "Partly cloudy",
            icon="01d" if random.random() > 0.3 else "02d",
            timestamp=datetime.now()
        )
    
    def _get_simulated_forecast(self, lat: float, lon: float, hours: int) -> List[WeatherData]:
        """Generate simulated forecast for development."""
        base_weather = self._get_simulated_weather(lat, lon)
        forecast = []
        
        for i in range(hours):
            forecast.append(WeatherData(
                temperature=round(base_weather.temperature + (i * 0.1), 1),
                humidity=int(base_weather.humidity - i * 0.5),
                wind_speed=round(base_weather.wind_speed + (i * 0.2), 1),
                wind_direction=base_weather.wind_direction,
                wind_gust=round(base_weather.wind_gust + (i * 0.3), 1),
                pressure=base_weather.pressure,
                clouds=min(100, base_weather.clouds + i * 2),
                precipitation=round(max(0, base_weather.precipitation - i * 0.05), 1),
                visibility=round(base_weather.visibility - i * 0.05, 1),
                uv_index=max(0, base_weather.uv_index - i * 0.1),
                description=base_weather.description,
                icon=base_weather.icon,
                timestamp=datetime.now() + timedelta(hours=i)
            ))
        
        return forecast


# Convenience function
def get_weather(lat: float, lon: float, api_key: str = None) -> WeatherData:
    """Get current weather for a location."""
    weather = WeatherAPI(api_key)
    return weather.get_current(lat, lon)


def get_fire_danger(lat: float, lon: float, api_key: str = None) -> FireDangerRating:
    """Get fire danger rating for a location."""
    weather = WeatherAPI(api_key)
    current = weather.get_current(lat, lon)
    if current:
        return weather.calculate_fire_danger(current)
    return None


if __name__ == "__main__":
    # Demo
    print("🌤️ Weather & Fire Danger Demo")
    print("=" * 50)
    
    weather = WeatherAPI()
    current = weather.get_current(-25.0, 133.0)  # Australia
    
    print(f"\nCurrent Weather (Australia):")
    print(f"  Temperature: {current.temperature}°C")
    print(f"  Humidity: {current.humidity}%")
    print(f"  Wind: {current.wind_speed} km/h {current.wind_direction}")
    print(f"  Pressure: {current.pressure} hPa")
    
    danger = weather.calculate_fire_danger(current)
    print(f"\n🔥 Fire Danger Rating:")
    print(f"  Rating: {danger.rating}")
    print(f"  Score: {danger.score:.1f}/100")
    print(f"  FWI: {danger.fwi:.1f}")
    print(f"  FFMC: {danger.ffmc:.1f}")
    print(f"  ISI: {danger.isi:.1f}")
    print(f"  BUI: {danger.bui:.1f}")
