"""
Fire Analytics Module

Advanced analytics for fire data:
- Hotspot identification
- Seasonal patterns
- Risk assessment
- Trend analysis
"""
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@dataclass
class FireHotspot:
    """Identified fire hotspot."""
    center_lat: float
    center_lon: float
    radius_km: float
    fire_count: int
    avg_brightness: float
    total_frp: float
    last_active: datetime
    severity: str  # LOW, MEDIUM, HIGH, EXTREME
    region_name: str
    
    def to_dict(self) -> Dict:
        return {
            "center": {"lat": self.center_lat, "lon": self.center_lon},
            "radius_km": self.radius_km,
            "fire_count": self.fire_count,
            "avg_brightness": round(self.avg_brightness, 1),
            "total_frp_mw": round(self.total_frp, 1),
            "last_active": self.last_active.isoformat(),
            "severity": self.severity,
            "region": self.region_name
        }


@dataclass
class SeasonalPattern:
    """Seasonal fire pattern analysis."""
    month: int
    avg_fires: float
    avg_brightness: float
    peak_region: str
    trend: str  # increasing, stable, decreasing
    
    def to_dict(self) -> Dict:
        return {
            "month": self.month,
            "avg_fires": round(self.avg_fires, 1),
            "avg_brightness": round(self.avg_brightness, 1),
            "peak_region": self.peak_region,
            "trend": self.trend
        }


@dataclass
class RiskAssessment:
    """Overall fire risk assessment for a region."""
    overall_risk: str  # LOW, MODERATE, HIGH, VERY_HIGH, EXTREME
    risk_score: float  # 0-100
    factors: Dict[str, float]
    recommendations: List[str]
    vulnerable_areas: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "overall_risk": self.overall_risk,
            "risk_score": round(self.risk_score, 1),
            "factors": {k: round(v, 2) for k, v in self.factors.items()},
            "recommendations": self.recommendations,
            "vulnerable_areas": self.vulnerable_areas
        }


class FireAnalytics:
    """
    Advanced fire data analytics engine.
    """
    
    # Region name mappings for coordinates
    REGION_NAMES = {
        (-30, 115): "Western Australia",
        (-35, 138): "South Australia",
        (-38, 145): "Victoria",
        (-33, 151): "New South Wales",
        (-27, 153): "Queensland",
        (-42, 147): "Tasmania",
        (-12, 131): "Northern Territory",
        (-25, 134): "Central Australia",
    }
    
    def __init__(self, historical_data: List[Dict] = None):
        """
        Initialize analytics engine.
        
        Args:
            historical_data: List of fire records with lat, lon, brightness, frp, date
        """
        self.historical_data = historical_data or []
        self.hotspots = []
        self.seasonal_patterns = {}
    
    def identify_hotspots(self, fires: List[Dict], min_fires: int = 3, 
                          radius_km: float = 50.0) -> List[FireHotspot]:
        """
        Identify fire hotspots using density-based clustering.
        
        Args:
            fires: List of fire records
            min_fires: Minimum fires to form a hotspot
            radius_km: Clustering radius
            
        Returns:
            List of FireHotspot objects
        """
        if not fires:
            return []
        
        # Group fires by region (simple grid-based)
        grid_size = 0.5  # degrees
        grid = defaultdict(list)
        
        for fire in fires:
            lat = fire.get('latitude', 0)
            lon = fire.get('longitude', 0)
            
            # Grid cell
            cell_lat = round(lat / grid_size) * grid_size
            cell_lon = round(lon / grid_size) * grid_size
            grid[(cell_lat, cell_lon)].append(fire)
        
        hotspots = []
        
        for (cell_lat, cell_lon), cell_fires in grid.items():
            if len(cell_fires) >= min_fires:
                # Calculate center
                avg_lat = statistics.mean(f['latitude'] for f in cell_fires)
                avg_lon = statistics.mean(f['longitude'] for f in cell_fires)
                
                # Calculate metrics
                brightnesses = [f.get('brightness', 0) for f in cell_fires]
                frps = [f.get('frp', 0) for f in cell_fires]
                dates = [datetime.fromisoformat(f.get('acq_date', '2026-01-01')) for f in cell_fires]
                
                region = self._get_region_name(avg_lat, avg_lon)
                
                # Calculate severity
                avg_brightness = statistics.mean(brightnesses)
                total_frp = sum(frps)
                fire_count = len(cell_fires)
                
                if avg_brightness > 400 or total_frp > 200 or fire_count > 20:
                    severity = "EXTREME"
                elif avg_brightness > 350 or total_frp > 100 or fire_count > 10:
                    severity = "HIGH"
                elif avg_brightness > 300 or total_frp > 50 or fire_count > 5:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                hotspot = FireHotspot(
                    center_lat=round(avg_lat, 4),
                    center_lon=round(avg_lon, 4),
                    radius_km=radius_km,
                    fire_count=fire_count,
                    avg_brightness=round(avg_brightness, 1),
                    total_frp=round(total_frp, 1),
                    last_active=max(dates),
                    severity=severity,
                    region_name=region
                )
                hotspots.append(hotspot)
        
        # Sort by severity and count
        hotspots.sort(key=lambda h: (-h.fire_count, -h.avg_brightness))
        
        self.hotspots = hotspots
        return hotspots
    
    def _get_region_name(self, lat: float, lon: float) -> str:
        """Get region name from coordinates."""
        for (ref_lat, ref_lon), name in self.REGION_NAMES.items():
            if abs(lat - ref_lat) < 10 and abs(lon - ref_lon) < 15:
                return name
        return f"Region ({lat:.1f}°, {lon:.1f}°)"
    
    def analyze_seasonal_patterns(self, fires: List[Dict]) -> List[SeasonalPattern]:
        """
        Analyze seasonal fire patterns.
        
        Args:
            fires: Historical fire data
            
        Returns:
            List of SeasonalPattern objects by month
        """
        # Group by month
        monthly_data = defaultdict(list)
        
        for fire in fires:
            try:
                date_str = fire.get('acq_date', '2026-01-01')
                if isinstance(date_str, str):
                    date = datetime.fromisoformat(date_str)
                else:
                    date = date_str
                month = date.month
                monthly_data[month].append(fire)
            except:
                continue
        
        patterns = []
        
        for month in range(1, 13):
            month_fires = monthly_data.get(month, [])
            
            if month_fires:
                avg_fires = len(month_fires)
                avg_brightness = statistics.mean(
                    f.get('brightness', 300) for f in month_fires
                )
                
                # Find peak region
                region_counts = defaultdict(int)
                for f in month_fires:
                    lat = f.get('latitude', 0)
                    lon = f.get('longitude', 0)
                    region = self._get_region_name(lat, lon)
                    region_counts[region] += 1
                
                peak_region = max(region_counts.keys(), key=lambda r: region_counts[r])
                
                # Determine trend (compare to previous 2 months average)
                prev_months = [(month - 1) % 12 or 12, (month - 2) % 12 or 12]
                prev_counts = sum(len(monthly_data.get(m, [])) for m in prev_months)
                prev_avg = prev_counts / 2 if prev_counts > 0 else avg_fires
                
                if avg_fires > prev_avg * 1.2:
                    trend = "increasing"
                elif avg_fires < prev_avg * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                patterns.append(SeasonalPattern(
                    month=month,
                    avg_fires=avg_fires,
                    avg_brightness=round(avg_brightness, 1),
                    peak_region=peak_region,
                    trend=trend
                ))
            else:
                patterns.append(SeasonalPattern(
                    month=month,
                    avg_fires=0,
                    avg_brightness=0,
                    peak_region="None",
                    trend="stable"
                ))
        
        self.seasonal_patterns = {p.month: p for p in patterns}
        return patterns
    
    def assess_risk(self, lat: float, lon: float, 
                    current_fires: List[Dict] = None,
                    weather_data: Dict = None) -> RiskAssessment:
        """
        Assess fire risk for a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
            current_fires: Recent fire data
            weather_data: Weather conditions
            
        Returns:
            RiskAssessment object
        """
        factors = {}
        score = 0
        
        # 1. Recent fire activity (weight: 30%)
        if current_fires:
            nearby_fires = [
                f for f in current_fires
                if self._haversine_distance(lat, lon, f.get('latitude', 0), f.get('longitude', 0)) < 50
            ]
            fire_factor = min(30, len(nearby_fires) * 3)
            factors['recent_fires'] = fire_factor
            score += fire_factor * 0.30
        else:
            factors['recent_fires'] = 0
        
        # 2. Weather conditions (weight: 40%)
        weather_score = 0
        if weather_data:
            temp = weather_data.get('temperature', 20)
            humidity = weather_data.get('humidity', 50)
            wind_speed = weather_data.get('wind_speed', 0)
            
            # High temp increases risk
            if temp > 30:
                weather_score += 15
            elif temp > 25:
                weather_score += 10
            elif temp > 20:
                weather_score += 5
            
            # Low humidity increases risk
            if humidity < 20:
                weather_score += 15
            elif humidity < 30:
                weather_score += 10
            elif humidity < 50:
                weather_score += 5
            
            # High wind increases risk
            if wind_speed > 50:
                weather_score += 10
            elif wind_speed > 30:
                weather_score += 7
            elif wind_speed > 20:
                weather_score += 4
        
        factors['weather'] = weather_score
        score += weather_score * 0.40
        
        # 3. Historical patterns (weight: 20%)
        history_score = 10  # Base score
        if self.seasonal_patterns:
            current_month = datetime.now().month
            pattern = self.seasonal_patterns.get(current_month)
            if pattern:
                if pattern.trend == "increasing":
                    history_score += 10
                elif pattern.trend == "stable":
                    history_score += 5
        
        factors['historical'] = history_score
        score += history_score * 0.20
        
        # 4. Geographic factors (weight: 10%)
        geo_score = 0
        # Fire-prone regions
        for (ref_lat, ref_lon), name in self.REGION_NAMES.items():
            if abs(lat - ref_lat) < 15:
                geo_score += 10
                break
        
        factors['geographic'] = geo_score
        score += geo_score * 0.10
        
        # Determine overall risk
        if score >= 80:
            overall_risk = "EXTREME"
        elif score >= 60:
            overall_risk = "VERY_HIGH"
        elif score >= 40:
            overall_risk = "HIGH"
        elif score >= 20:
            overall_risk = "MODERATE"
        else:
            overall_risk = "LOW"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_risk, factors)
        
        # Identify vulnerable areas
        vulnerable = []
        if factors.get('weather', 0) > 15:
            vulnerable.append("High wind zones - embers may spread rapidly")
        if factors.get('recent_fires', 0) > 15:
            vulnerable.append("Active fire perimeter - risk of spread")
        if weather_data and weather_data.get('humidity', 50) < 30:
            vulnerable.append("Low humidity area - rapid fuel drying")
        
        return RiskAssessment(
            overall_risk=overall_risk,
            risk_score=min(100, score),
            factors=factors,
            recommendations=recommendations,
            vulnerable_areas=vulnerable
        )
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                            lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _generate_recommendations(self, risk: str, factors: Dict) -> List[str]:
        """Generate recommendations based on risk factors."""
        recs = []
        
        if risk in ["EXTREME", "VERY_HIGH"]:
            recs.append("🚨 Immediate fire alert - prepare evacuation routes")
            recs.append("🔥 Deploy additional firefighting resources")
            recs.append("📢 Issue community warning alerts")
        elif risk == "HIGH":
            recs.append("⚠️ Elevated fire danger - increase monitoring")
            recs.append("👀 Pre-position firefighting crews")
            recs.append("📵 Restrict open fires in the area")
        elif risk == "MODERATE":
            recs.append("📝 Monitor conditions closely")
            recs.append("🚭 Exercise caution with outdoor activities")
            recs.append("✅ Ensure fire suppression equipment is ready")
        else:
            recs.append("✅ Normal fire danger conditions")
            recs.append("📋 Continue routine monitoring")
        
        if factors.get('weather', 0) > 15:
            recs.append("🌬️ High wind warning - watch for ember spread")
        
        if factors.get('recent_fires', 0) > 10:
            recs.append("🔥 Active fires nearby - assess containment lines")
        
        return recs
    
    def get_summary(self) -> Dict:
        """Get analytics summary."""
        return {
            "hotspots_identified": len(self.hotspots),
            "extreme_hotspots": len([h for h in self.hotspots if h.severity == "EXTREME"]),
            "high_hotspots": len([h for h in self.hotspots if h.severity == "HIGH"]),
            "months_analyzed": len(self.seasonal_patterns),
            "peak_fire_month": max(
                self.seasonal_patterns.values(), 
                key=lambda p: p.avg_fires, 
                default=None
            ) if self.seasonal_patterns else None
        }


# Convenience functions
def analyze_fires(fires: List[Dict]) -> Dict:
    """Quick fire analysis function."""
    analytics = FireAnalytics()
    hotspots = analytics.identify_hotspots(fires)
    patterns = analytics.analyze_seasonal_patterns(fires)
    
    return {
        "hotspots": [h.to_dict() for h in hotspots],
        "patterns": [p.to_dict() for p in patterns],
        "summary": analytics.get_summary()
    }


if __name__ == "__main__":
    # Demo
    print("🔥 Fire Analytics Demo")
    print("=" * 50)
    
    # Sample fire data
    sample_fires = [
        {"latitude": -22.8, "longitude": 131.9, "brightness": 350, "frp": 100, "acq_date": "2026-02-01"},
        {"latitude": -22.9, "longitude": 131.95, "brightness": 380, "frp": 120, "acq_date": "2026-02-02"},
        {"latitude": -22.85, "longitude": 131.92, "brightness": 320, "frp": 80, "acq_date": "2026-02-02"},
        {"latitude": -35.5, "longitude": 138.5, "brightness": 300, "frp": 50, "acq_date": "2026-02-01"},
        {"latitude": -35.6, "longitude": 138.6, "brightness": 310, "frp": 60, "acq_date": "2026-02-02"},
    ]
    
    analytics = FireAnalytics()
    
    # Identify hotspots
    hotspots = analytics.identify_hotspots(sample_fires)
    print(f"\n📍 Hotspots Identified: {len(hotspots)}")
    for h in hotspots:
        print(f"  {h.region_name}: {h.fire_count} fires, {h.severity}")
    
    # Seasonal patterns
    patterns = analytics.analyze_seasonal_patterns(sample_fires)
    print(f"\n📊 Seasonal Patterns: {len(patterns)} months")
    
    # Risk assessment
    risk = analytics.assess_risk(-22.8, 131.9, sample_fires, {
        "temperature": 35,
        "humidity": 20,
        "wind_speed": 40
    })
    print(f"\n⚠️ Risk Assessment:")
    print(f"  Overall: {risk.overall_risk} ({risk.risk_score:.1f}/100)")
    print(f"  Factors: {risk.factors}")
    print(f"  Recs: {risk.recommendations[0]}")
    
    print(f"\n📈 Summary: {analytics.get_summary()}")
