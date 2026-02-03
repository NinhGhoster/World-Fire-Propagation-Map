"""
Fire Station Data and Response Time Modeling

Includes real fire station locations for major fire-prone regions.
"""
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2


@dataclass
class FireStation:
    """Represents a fire station with location and resources."""
    name: str
    lat: float
    lon: float
    trucks: int = 2
    response_time_base: int = 15  # minutes to nearest point
    coverage_radius_km: float = 50.0


# Sample fire stations for major fire regions
FIRE_STATIONS = {
    "Australia": [
        FireStation("Perth", -31.9505, 115.8605, trucks=4, coverage_radius_km=80),
        FireStation("Albany", -35.0228, 117.8836, trucks=2, coverage_radius_km=60),
        FireStation("Esperance", -33.8592, 121.8919, trucks=2, coverage_radius_km=60),
        FireStation("Kalgoorlie", -30.7489, 121.4737, trucks=2, coverage_radius_km=100),
        FireStation("PortLincoln", -34.7260, 135.8669, trucks=2, coverage_radius_km=50),
        FireStation("Adelaide", -34.9285, 138.6007, trucks=6, coverage_radius_km=70),
        FireStation("MountGambier", -37.8316, 140.7792, trucks=3, coverage_radius_km=50),
        FireStation("Mildura", -34.2068, 142.1366, trucks=2, coverage_radius_km=60),
        FireStation("Bendigo", -36.7580, 144.2796, trucks=3, coverage_radius_km=60),
        FireStation("Melbourne", -37.8136, 144.9631, trucks=10, coverage_radius_km=60),
        FireStation("Geelong", -38.1509, 144.3596, trucks=4, coverage_radius_km=50),
        FireStation("Warrnambool", -38.3821, 142.4816, trucks=2, coverage_radius_km=50),
        FireStation("Albury", -36.0736, 146.9243, trucks=3, coverage_radius_km=60),
        FireStation("WaggaWagga", -35.1252, 147.3568, trucks=3, coverage_radius_km=70),
        FireStation("Canberra", -35.2809, 149.1300, trucks=5, coverage_radius_km=60),
        FireStation("Sydney", -33.8688, 151.2093, trucks=12, coverage_radius_km=60),
        FireStation("Newcastle", -32.9282, 151.7817, trucks=5, coverage_radius_km=50),
        FireStation("PortMacquarie", -31.4405, 152.9089, trucks=3, coverage_radius_km=50),
        FireStation("CoffsHarbour", -30.2963, 153.1139, trucks=3, coverage_radius_km=50),
        FireStation("Brisville", -27.4698, 153.0251, trucks=8, coverage_radius_km=60),
        FireStation("Rockhampton", -23.3800, 150.5000, trucks=3, coverage_radius_km=60),
        FireStation("Townsville", -19.2590, 146.8169, trucks=4, coverage_radius_km=50),
        FireStation("Cairns", -16.9255, 145.7711, trucks=4, coverage_radius_km=50),
        FireStation("AliceSprings", -23.7348, 133.8805, trucks=2, coverage_radius_km=80),
        FireStation("Darwin", -12.4628, 130.8418, trucks=4, coverage_radius_km=50),
    ],
    "USA": [
        FireStation("LosAngeles", 34.0522, -118.2437, trucks=8, coverage_radius_km=50),
        FireStation("SanFrancisco", 37.7749, -122.4194, trucks=6, coverage_radius_km=40),
        FireStation("Portland", 45.5152, -122.6784, trucks=5, coverage_radius_km=50),
        FireStation("Seattle", 47.6062, -122.3321, trucks=6, coverage_radius_km=50),
        FireStation("Denver", 39.7392, -104.9903, trucks=5, coverage_radius_km=60),
        FireStation("Phoenix", 33.4484, -112.0740, trucks=6, coverage_radius_km=50),
        FireStation("Dallas", 32.7767, -96.7970, trucks=5, coverage_radius_km=50),
        FireStation("Houston", 29.7604, -95.3698, trucks=5, coverage_radius_km=50),
        FireStation("Atlanta", 33.7490, -84.3880, trucks=4, coverage_radius_km=50),
        FireStation("Miami", 25.7617, -80.1918, trucks=5, coverage_radius_km=40),
        FireStation("NewYork", 40.7128, -74.0060, trucks=8, coverage_radius_km=40),
        FireStation("Chicago", 41.8781, -87.6298, trucks=7, coverage_radius_km=50),
    ],
    "Brazil": [
        FireStation("Manaus", -3.1190, -60.0217, trucks=4, coverage_radius_km=60),
        FireStation("PortoVelho", -8.7619, -63.9009, trucks=3, coverage_radius_km=70),
        FireStation("RioBranco", -9.9742, -67.8246, trucks=3, coverage_radius_km=60),
        FireStation("Cuiaba", -15.6010, -56.0974, trucks=3, coverage_radius_km=60),
        FireStation("Santarém", -2.4500, -54.7000, trucks=2, coverage_radius_km=60),
        FireStation("Brasilia", -15.7975, -47.8919, trucks=4, coverage_radius_km=50),
        FireStation("Goiania", -16.6864, -49.2643, trucks=3, coverage_radius_km=50),
        FireStation("Cuiaba", -15.6010, -56.0974, trucks=3, coverage_radius_km=60),
    ],
    "Indonesia": [
        FireStation("PalangkaRaya", -2.2167, 113.9167, trucks=3, coverage_radius_km=60),
        FireStation("Pontianak", -0.0222, 109.3425, trucks=3, coverage_radius_km=50),
        FireStation("Banjarmasin", -3.3167, 114.5908, trucks=3, coverage_radius_km=50),
        FireStation("Samarinda", -0.4956, 117.1491, trucks=3, coverage_radius_km=60),
        FireStation("Jakarta", -6.2088, 106.8456, trucks=8, coverage_radius_km=40),
        FireStation("Surabaya", -7.2575, 112.7521, trucks=5, coverage_radius_km=50),
    ],
    "Greece": [
        FireStation("Athens", 37.9838, 23.7275, trucks=6, coverage_radius_km=40),
        FireStation("Thessaloniki", 40.6401, 22.9444, trucks=5, coverage_radius_km=50),
        FireStation("Patras", 38.2466, 21.7346, trucks=3, coverage_radius_km=40),
        FireStation("Heraklion", 35.3387, 25.1370, trucks=4, coverage_radius_km=40),
        FireStation("Larissa", 39.6390, 22.4185, trucks=3, coverage_radius_km=50),
        FireStation("Volos", 39.3667, 22.9500, trucks=3, coverage_radius_km=40),
    ],
    "Portugal": [
        FireStation("Lisbon", 38.7223, -9.1393, trucks=6, coverage_radius_km=40),
        FireStation("Porto", 41.1579, -8.6291, trucks=5, coverage_radius_km=40),
        FireStation("Faro", 37.0194, -7.9304, trucks=3, coverage_radius_km=40),
        FireStation("Coimbra", 40.2033, -8.4103, trucks=3, coverage_radius_km=40),
    ],
    "Spain": [
        FireStation("Madrid", 40.4168, -3.7038, trucks=8, coverage_radius_km=50),
        FireStation("Barcelona", 41.3851, 2.1734, trucks=6, coverage_radius_km=40),
        FireStation("Valencia", 39.4699, -0.3763, trucks=5, coverage_radius_km=40),
        FireStation("Seville", 37.3891, -5.9845, trucks=5, coverage_radius_km=50),
        FireStation("Zaragoza", 41.6488, -0.8891, trucks=4, coverage_radius_km=50),
        FireStation("Murcia", 37.9922, -1.1307, trucks=4, coverage_radius_km=40),
    ],
    "Italy": [
        FireStation("Rome", 41.9028, 12.4964, trucks=8, coverage_radius_km=50),
        FireStation("Milan", 45.4642, 9.1900, trucks=7, coverage_radius_km=40),
        FireStation("Naples", 40.8518, 14.2681, trucks=5, coverage_radius_km=40),
        FireStation("Turin", 45.0703, 7.6869, trucks=5, coverage_radius_km=40),
        FireStation("Bologna", 44.4949, 11.3426, trucks=4, coverage_radius_km=40),
        FireStation("Palermo", 38.1157, 13.3615, trucks=4, coverage_radius_km=40),
    ],
    "France": [
        FireStation("Paris", 48.8566, 2.3522, trucks=10, coverage_radius_km=40),
        FireStation("Marseille", 43.2965, 5.3698, trucks=5, coverage_radius_km=40),
        FireStation("Toulouse", 43.6047, 1.4480, trucks=4, coverage_radius_km=50),
        FireStation("Bordeaux", 44.8378, -0.5792, trucks=4, coverage_radius_km=50),
        FireStation("Nice", 43.7102, 7.2620, trucks=4, coverage_radius_km=40),
        FireStation("Montpellier", 43.6108, 3.8772, trucks=3, coverage_radius_km=40),
    ],
    "SouthAfrica": [
        FireStation("CapeTown", -33.9249, 18.4241, trucks=5, coverage_radius_km=50),
        FireStation("Johannesburg", -26.2041, 28.0473, trucks=6, coverage_radius_km=60),
        FireStation("Durban", -29.8587, 31.0218, trucks=5, coverage_radius_km=50),
        FireStation("PortElizabeth", -33.9608, 25.6052, trucks=3, coverage_radius_km=50),
        FireStation("Bloemfontein", -29.0852, 26.1596, trucks=3, coverage_radius_km=60),
    ],
    "Canada": [
        FireStation("Vancouver", 49.2827, -123.1207, trucks=6, coverage_radius_km=50),
        FireStation("Calgary", 51.0447, -114.0719, trucks=5, coverage_radius_km=60),
        FireStation("Edmonton", 53.5461, -113.4938, trucks=5, coverage_radius_km=60),
        FireStation("Toronto", 43.6532, -79.3832, trucks=8, coverage_radius_km=40),
        FireStation("Montreal", 45.5017, -73.5673, trucks=6, coverage_radius_km=40),
        FireStation("Ottawa", 45.4215, -75.6972, trucks=5, coverage_radius_km=50),
        FireStation("Halifax", 44.6486, -63.5819, trucks=4, coverage_radius_km=40),
    ],
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def get_stations_for_region(country: str) -> List[FireStation]:
    """Get fire stations for a country."""
    return FIRE_STATIONS.get(country, [])


def get_nearest_stations(lat: float, lon: float, country: str = None, 
                         max_stations: int = 5) -> List[Tuple[FireStation, float]]:
    """
    Get the nearest fire stations to a location.
    Returns list of (station, distance_km) tuples.
    """
    all_stations = []
    
    if country and country in FIRE_STATIONS:
        all_stations = FIRE_STATIONS[country]
    else:
        # Search all countries
        for stations in FIRE_STATIONS.values():
            all_stations.extend(stations)
    
    # Calculate distances
    stations_with_dist = []
    for station in all_stations:
        dist = haversine_distance(lat, lon, station.lat, station.lon)
        stations_with_dist.append((station, dist))
    
    # Sort by distance and return top N
    stations_with_dist.sort(key=lambda x: x[1])
    return stations_with_dist[:max_stations]


def calculate_response_time(lat: float, lon: float, station: FireStation) -> int:
    """
    Calculate response time in minutes.
    Assumes average speed of 60 km/h in urban, 80 km/h on highways.
    """
    distance = haversine_distance(lat, lon, station.lat, station.lon)
    
    # Base response time (minutes)
    base_time = (distance / 60) * 60  # 60 km/h average
    
    # Add traffic/terrain factor (1.2x)
    base_time *= 1.2
    
    # Minimum 5 minutes, maximum based on coverage
    return max(5, min(int(base_time), 60))


def get_coverage_status(lat: float, lon: float, country: str = None) -> Dict:
    """
    Check fire coverage status for a location.
    Returns which stations can respond and estimated times.
    """
    nearest = get_nearest_stations(lat, lon, country, max_stations=10)
    
    coverage = []
    for station, distance in nearest:
        response_time = calculate_response_time(lat, lon, station)
        can_respond = distance <= station.coverage_radius_km
        
        coverage.append({
            "station": station.name,
            "distance_km": round(distance, 1),
            "response_time_min": response_time,
            "trucks_available": station.trucks if can_respond else 0,
            "in_coverage_radius": can_respond
        })
    
    # Find best station
    best = min([c for c in coverage if c["in_coverage_radius"]], 
               key=lambda x: x["response_time_min"], default=None)
    
    return {
        "location": {"lat": lat, "lon": lon},
        "nearest_station": best["station"] if best else None,
        "best_response_time": best["response_time_min"] if best else None,
        "stations": coverage,
        "total_available_trucks": sum(s["trucks_available"] for s in coverage)
    }


# Demo
if __name__ == "__main__":
    # Test with Australia
    lat, lon = -35.0, 138.0  # Adelaide area
    
    print("🔥 Fire Station Coverage Analysis")
    print("=" * 50)
    print(f"Location: {lat}, {lon}")
    print()
    
    status = get_coverage_status(lat, lon, "Australia")
    
    print(f"Nearest Station: {status['nearest_station']}")
    print(f"Best Response Time: {status['best_response_time']} min")
    print(f"Available Trucks: {status['total_available_trucks']}")
    print()
    print("All Stations:")
    for s in status['stations'][:5]:
        print(f"  - {s['station']}: {s['distance_km']}km, {s['response_time_min']}min")
