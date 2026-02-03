"""
World Fire Propagation Map - Data Export Module

Export simulation results and fire data to various formats.
"""
import json
import csv
import zipfile
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64


class DataExporter:
    """Export data to various formats."""
    
    @staticmethod
    def to_json(data: Any, filename: str = None) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: Data to export
            filename: Optional filename
        
        Returns:
            JSON string
        """
        result = json.dumps(data, indent=2, default=str)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(result)
        
        return result
    
    @staticmethod
    def to_csv(data: List[Dict], filename: str = None, fieldnames: List[str] = None) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: List of dictionaries
            filename: Optional filename
            fieldnames: Explicit fieldnames, or auto-detect
        
        Returns:
            CSV string
        """
        if not data:
            return ""
        
        fieldnames = fieldnames or list(data[0].keys()) if data else []
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
        result = output.getvalue()
        
        if filename:
            with open(filename, 'w') as f:
                f.write(result)
        
        return result
    
    @staticmethod
    def to_geojson(fires: List[Dict], filename: str = None) -> str:
        """
        Export fire data to GeoJSON format.
        
        Args:
            fires: List of fire records with lat/lon
            filename: Optional filename
        
        Returns:
            GeoJSON string
        """
        features = []
        
        for fire in fires:
            lat = fire.get("latitude")
            lon = fire.get("longitude")
            
            if lat is None or lon is None:
                continue
            
            properties = {k: v for k, v in fire.items() 
                         if k not in ["latitude", "longitude"]}
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": properties
            }
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated": datetime.now().isoformat(),
                "count": len(features)
            }
        }
        
        result = json.dumps(geojson, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(result)
        
        return result
    
    @staticmethod
    def to_kml(fires: List[Dict], filename: str = None, name: str = "Fires") -> str:
        """
        Export fire data to KML format for Google Earth.
        
        Args:
            fires: List of fire records with lat/lon
            filename: Optional filename
            name: Name for the KML document
        
        Returns:
            KML string
        """
        placemarks = []
        
        for fire in fires:
            lat = fire.get("latitude")
            lon = fire.get("longitude")
            
            if lat is None or lon is None:
                continue
            
            desc = "<br/>".join([f"{k}: {v}" for k, v in fire.items() 
                               if k not in ["latitude", "longitude"]])
            
            placemark = f"""
        <Placemark>
            <name>Fire at {lat:.4f}, {lon:.4f}</name>
            <description><![CDATA[{desc}]]></description>
            <Point>
                <coordinates>{lon},{lat}</coordinates>
            </Point>
        </Placemark>"""
            
            placemarks.append(placemark)
        
        kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{name}</name>
        {''.join(placemarks)}
    </Document>
</kml>"""
        
        if filename:
            with open(filename, 'w') as f:
                f.write(kml)
        
        return kml
    
    @staticmethod
    def to_image_grid(grid_data: List[int], grid_size: int, filename: str = None) -> str:
        """
        Export grid data as ASCII art.
        
        Args:
            grid_data: Flat list of node values
            grid_size: Size of the grid (n x n)
            filename: Optional filename
        
        Returns:
            ASCII art string
        """
        # Create 2D grid
        grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                value = grid_data[idx] if idx < len(grid_data) else 0
                
                if value == 0:
                    row.append("░")  # Unburned
                elif value == 1:
                    row.append("🔥")  # Burning
                elif value == 2:
                    row.append("🛡️")  # Protected
                elif value == 3:
                    row.append("⬛")  # Burned
                else:
                    row.append("?")
            
            grid.append("".join(row))
        
        result = "\n".join(grid)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(result)
        
        return result
    
    @staticmethod
    def create_zip(files: Dict[str, str], filename: str = None) -> bytes:
        """
        Create a ZIP file with multiple exports.
        
        Args:
            files: Dictionary of {filename: content}
            filename: Optional filename to save
        
        Returns:
            ZIP file bytes
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for name, content in files.items():
                zip_file.writestr(name, content)
        
        result = zip_buffer.getvalue()
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(result)
        
        return result
    
    @staticmethod
    def create_simulation_report(simulation_result: Dict, config: Dict) -> str:
        """
        Create a formatted simulation report.
        
        Args:
            simulation_result: Results from simulation
            config: Simulation configuration
        
        Returns:
            Markdown report string
        """
        report = f"""# Fire Spread Simulation Report

Generated: {datetime.now().isoformat()}

## Configuration

| Parameter | Value |
|-----------|-------|
| Grid Size | {config.get('grid_size', 'N/A')} |
| Lambda (Spread Rate) | {config.get('lambda_spread', 'N/A')} |
| Firefighters | {config.get('num_firefighters', 'N/A')} |
| Strategy | {config.get('strategy', 'N/A')} |

## Results

| Metric | Value |
|--------|-------|
| Total Burned | {simulation_result.get('total_burned', 'N/A')} |
| Total Protected | {simulation_result.get('total_protected', 'N/A')} |
| Time Steps | {simulation_result.get('time_steps', 'N/A')} |
| Firefighter Placements | {len(simulation_result.get('firefighter_placements', {}))} |

## Firefighter Placements

"""
        placements = simulation_result.get('firefighter_placements', {})
        for timestep, node in placements.items():
            report += f"- Time step {timestep}: Node {node}\n"
        
        return report


def export_simulation_result(
    result: Dict,
    config: Dict,
    format: str = "json",
    filename: str = None
) -> str:
    """
    Export simulation result in the specified format.
    
    Args:
        result: Simulation result dictionary
        config: Simulation configuration
        format: Export format (json, csv, markdown, zip)
        filename: Optional filename
    
    Returns:
        Exported data as string or bytes
    """
    exporter = DataExporter()
    
    if format == "json":
        data = {
            "config": config,
            "result": result,
            "exported_at": datetime.now().isoformat()
        }
        return exporter.to_json(data, filename)
    
    elif format == "csv":
        return exporter.to_csv([result], filename)
    
    elif format == "markdown":
        return exporter.create_simulation_report(result, config)
    
    elif format == "zip":
        files = {
            "simulation_result.json": exporter.to_json({"config": config, "result": result}),
            "report.md": exporter.create_simulation_report(result, config)
        }
        return exporter.create_zip(files, filename)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Example export
    result = {
        "total_burned": 15,
        "total_protected": 34,
        "time_steps": 5,
        "burned_nodes": list(range(15)),
        "protected_nodes": list(range(15, 49)),
        "firefighter_placements": {0: 24, 1: 18, 2: 12}
    }
    
    config = {
        "grid_size": 7,
        "lambda_spread": 0.005,
        "num_firefighters": 1,
        "strategy": "greedy"
    }
    
    # Export to different formats
    print("JSON:")
    print(export_simulation_result(result, config, "json")[:500])
    
    print("\n\nMarkdown Report:")
    print(export_simulation_result(result, config, "markdown"))
