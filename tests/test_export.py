"""
Unit tests for export module.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock


class TestDataExporter:
    """Tests for DataExporter class."""
    
    def test_to_json(self):
        """Test JSON export."""
        from modules.export import DataExporter
        
        data = {"key": "value", "number": 42}
        result = DataExporter.to_json(data)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42
    
    def test_to_json_file(self, tmp_path):
        """Test JSON export to file."""
        from modules.export import DataExporter
        
        data = {"test": "data"}
        filepath = tmp_path / "test.json"
        
        result = DataExporter.to_json(data, str(filepath))
        
        assert filepath.exists()
        
        with open(filepath) as f:
            assert json.load(f) == data
    
    def test_to_csv(self):
        """Test CSV export."""
        from modules.export import DataExporter
        
        data = [
            {"name": "Fire 1", "lat": -35.0, "lon": 140.0},
            {"name": "Fire 2", "lat": -35.1, "lon": 140.1}
        ]
        result = DataExporter.to_csv(data)
        
        assert "name,lat,lon" in result
        assert "Fire 1" in result
        assert "Fire 2" in result
    
    def test_to_csv_empty(self):
        """Test CSV export with empty data."""
        from modules.export import DataExporter
        
        result = DataExporter.to_csv([])
        
        assert result == ""
    
    def test_to_geojson(self):
        """Test GeoJSON export."""
        from modules.export import DataExporter
        
        fires = [
            {"latitude": -35.0, "longitude": 140.0, "brightness": 320.5},
            {"latitude": -35.1, "longitude": 140.1, "brightness": 315.0}
        ]
        result = DataExporter.to_geojson(fires)
        
        parsed = json.loads(result)
        
        assert parsed["type"] == "FeatureCollection"
        assert len(parsed["features"]) == 2
        assert parsed["features"][0]["geometry"]["type"] == "Point"
        assert parsed["features"][0]["geometry"]["coordinates"] == [140.0, -35.0]
    
    def test_to_geojson_empty(self):
        """Test GeoJSON export with empty data."""
        from modules.export import DataExporter
        
        result = DataExporter.to_geojson([])
        
        parsed = json.loads(result)
        
        assert parsed["type"] == "FeatureCollection"
        assert len(parsed["features"]) == 0
    
    def test_to_kml(self):
        """Test KML export."""
        from modules.export import DataExporter
        
        fires = [
            {"latitude": -35.0, "longitude": 140.0, "brightness": 320.5}
        ]
        result = DataExporter.to_kml(fires, name="Test Fires")
        
        assert "<?xml version" in result
        assert "<kml xmlns" in result
        assert "Test Fires" in result
        assert "Fire at -35.0000, 140.0000" in result
    
    def test_to_image_grid(self):
        """Test ASCII grid export."""
        from modules.export import DataExporter
        
        grid_data = list(range(49))  # 7x7 grid
        result = DataExporter.to_image_grid(grid_data, 7)
        
        lines = result.split("\n")
        assert len(lines) == 7
        
        # Check for fire emoji (burning nodes)
        assert "🔥" in result
    
    def test_create_zip(self):
        """Test ZIP file creation."""
        from modules.export import DataExporter
        
        files = {
            "data.json": '{"test": "data"}',
            "report.md": "# Report"
        }
        result = DataExporter.create_zip(files)
        
        assert isinstance(result, bytes)
        
        # Verify ZIP contents
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            assert "data.json" in zf.namelist()
            assert "report.md" in zf.namelist()
    
    def test_create_simulation_report(self):
        """Test simulation report creation."""
        from modules.export import DataExporter
        
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
        
        report = DataExporter.create_simulation_report(result, config)
        
        assert "# Fire Spread Simulation Report" in report
        assert "Grid Size" in report
        assert "Lambda" in report
        assert "Total Burned" in report
        assert "15" in report


class TestExportSimulationResult:
    """Tests for export_simulation_result function."""
    
    def test_export_json(self):
        """Test JSON export of simulation result."""
        from modules.export import export_simulation_result
        
        result = {"total_burned": 15}
        config = {"grid_size": 7}
        
        output = export_simulation_result(result, config, "json")
        
        parsed = json.loads(output)
        assert parsed["result"]["total_burned"] == 15
        assert parsed["config"]["grid_size"] == 7
    
    def test_export_markdown(self):
        """Test Markdown export of simulation result."""
        from modules.export import export_simulation_result
        
        result = {"total_burned": 15}
        config = {"grid_size": 7}
        
        output = export_simulation_result(result, config, "markdown")
        
        assert "# Fire Spread Simulation Report" in output
        assert "15" in output
    
    def test_export_zip(self):
        """Test ZIP export of simulation result."""
        from modules.export import export_simulation_result
        
        result = {"total_burned": 15}
        config = {"grid_size": 7}
        
        output = export_simulation_result(result, config, "zip")
        
        assert isinstance(output, bytes)
        
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(output)) as zf:
            assert "simulation_result.json" in zf.namelist()
            assert "report.md" in zf.namelist()
    
    def test_invalid_format(self):
        """Test invalid format raises error."""
        from modules.export import export_simulation_result
        
        with pytest.raises(ValueError) as exc_info:
            export_simulation_result({}, {}, "invalid")
        
        assert "Unsupported format" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
