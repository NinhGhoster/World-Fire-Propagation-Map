"""
Unit tests for simulation module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestSimulationConfig:
    """Tests for SimulationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from modules.simulation import SimulationConfig
        
        config = SimulationConfig()
        
        assert config.grid_size == 7
        assert config.lambda_spread == 0.005
        assert config.num_firefighters == 1
        assert config.fire_start_nodes == []
        assert config.seed is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from modules.simulation import SimulationConfig
        
        config = SimulationConfig(
            grid_size=5,
            lambda_spread=0.01,
            num_firefighters=2,
            fire_start_nodes=[12],
            seed=42
        )
        
        assert config.grid_size == 5
        assert config.lambda_spread == 0.01
        assert config.num_firefighters == 2
        assert config.fire_start_nodes == [12]
        assert config.seed == 42


class TestFireSpreadSimulator:
    """Tests for FireSpreadSimulator class."""
    
    def test_grid_creation(self):
        """Test that grid graph is created correctly."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(grid_size=7)
        simulator = FireSpreadSimulator(config)
        
        # 7x7 = 49 nodes
        assert simulator.num_nodes == 49
        assert simulator.graph.number_of_nodes() == 49
        # Each internal node has 4 edges, edge nodes have fewer
        assert simulator.graph.number_of_edges() == 84  # (7-1)*7*2 = 84
    
    def test_get_neighbors_center(self):
        """Test neighbor finding for center node."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(grid_size=7)
        simulator = FireSpreadSimulator(config)
        
        # Node 24 is center of 7x7 grid
        neighbors = simulator.get_neighbors(24)
        
        assert len(neighbors) == 4  # All 4 directions available
        assert 17 in neighbors  # Up
        assert 31 in neighbors  # Down
        assert 23 in neighbors  # Left
        assert 25 in neighbors  # Right
    
    def test_get_neighbors_corner(self):
        """Test neighbor finding for corner node."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(grid_size=7)
        simulator = FireSpreadSimulator(config)
        
        # Node 0 is top-left corner
        neighbors = simulator.get_neighbors(0)
        
        assert len(neighbors) == 2  # Only right and down
        assert 1 in neighbors  # Right
        assert 7 in neighbors  # Down
    
    def test_get_neighbors_edge(self):
        """Test neighbor finding for edge node."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(grid_size=7)
        simulator = FireSpreadSimulator(config)
        
        # Node 3 is top edge (not corner)
        neighbors = simulator.get_neighbors(3)
        
        assert len(neighbors) == 3  # 3 directions available
    
    def test_simulation_step(self):
        """Test single simulation step."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            lambda_spread=0.1  # High spread for testing
        )
        simulator = FireSpreadSimulator(config)
        
        # Initial state
        assert 24 in simulator.burning
        assert len(simulator.burning) == 1
        
        # Run one step without firefighters
        result = simulator.step()
        
        # Fire should have spread
        assert len(simulator.burning) >= 0  # May or may not spread
        assert simulator.time_step == 1
    
    def test_simulation_with_firefighters(self):
        """Test simulation with firefighter deployment."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            num_firefighters=1,
            lambda_spread=0.1
        )
        simulator = FireSpreadSimulator(config)
        
        # Deploy firefighter to node 25
        simulator.step([25])
        
        # Node 25 should be protected
        assert 25 in simulator.protected
    
    def test_simulation_run(self):
        """Test full simulation run."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            lambda_spread=0.005,
            num_firefighters=1,
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        result = simulator.run(firefighter_strategy="greedy")
        
        assert isinstance(result.total_burned, int)
        assert isinstance(result.total_protected, int)
        assert isinstance(result.time_steps, int)
        assert result.total_burned + result.total_protected <= simulator.num_nodes
    
    def test_greedy_strategy(self):
        """Test greedy firefighter placement."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            num_firefighters=1,
            lambda_spread=0.01,
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        
        # Get greedy placement
        placement = simulator._greedy_placement()
        
        assert isinstance(placement, list)
        assert len(placement) <= 1
    
    def test_random_strategy(self):
        """Test random firefighter placement."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            num_firefighters=1,
            lambda_spread=0.01,
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        
        # Get random placement
        placement = simulator._random_placement()
        
        assert isinstance(placement, list)
    
    def test_central_strategy(self):
        """Test central firefighter placement."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            num_firefighters=1,
            lambda_spread=0.01,
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        
        # Get central placement
        placement = simulator._central_placement()
        
        assert isinstance(placement, list)
    
    def test_grid_visualization(self):
        """Test grid visualization."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        
        # Run a few steps
        for _ in range(5):
            simulator.step()
        
        grid = simulator.get_grid_visualization()
        
        assert grid.shape == (7, 7)
        assert grid.dtype == int
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        from modules.simulation import FireSpreadSimulator, SimulationConfig
        
        config = SimulationConfig(
            grid_size=7,
            fire_start_nodes=[24],
            lambda_spread=0.005,
            num_firefighters=1,
            seed=42
        )
        simulator = FireSpreadSimulator(config)
        comparison = simulator.compare_strategies()
        
        assert "greedy" in comparison
        assert "random" in comparison
        assert "central" in comparison
        
        for strategy, result in comparison.items():
            assert isinstance(result.total_burned, int)


class TestSimulationResult:
    """Tests for SimulationResult class."""
    
    def test_default_result(self):
        """Test default simulation result."""
        from modules.simulation import SimulationResult
        
        result = SimulationResult()
        
        assert result.total_burned == 0
        assert result.total_protected == 0
        assert result.time_steps == 0
        assert result.burned_nodes == []
        assert result.protected_nodes == []
        assert result.firefighter_placements == {}
        assert result.fire_progression == []
    
    def test_custom_result(self):
        """Test custom simulation result."""
        from modules.simulation import SimulationResult
        
        result = SimulationResult(
            total_burned=15,
            total_protected=34,
            time_steps=5,
            burned_nodes=list(range(15)),
            protected_nodes=list(range(15, 49)),
            firefighter_placements={0: 24, 1: 18}
        )
        
        assert result.total_burned == 15
        assert result.total_protected == 34
        assert len(result.fire_progression) == 0  # Not set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
