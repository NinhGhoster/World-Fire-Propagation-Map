"""
Fire Spread Simulation with Wind Integration

Moving Firefighter Problem (MFF) with realistic fire spread modeling.
"""
import random
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import math


@dataclass
class SimulationConfig:
    """Configuration for fire spread simulation."""
    grid_size: int = 7
    lambda_spread: float = 0.1
    num_firefighters: int = 2
    fire_start_nodes: List[int] = field(default_factory=lambda: [24])
    seed: int = 42
    
    # Wind parameters
    wind_speed: float = 0.0  # km/h (0-100)
    wind_direction: str = "NE"  # N, NE, E, SE, S, SW, W, NW
    
    # Response time (minutes from dispatch to arrival)
    response_time: int = 15
    
    # Weather factors
    temperature: float = 25.0  # Celsius
    humidity: float = 30.0  # Percent
    vegetation_dryness: float = 0.8  # 0-1, fuel moisture inverse


class FireSpreadSimulator:
    """
    Simulates fire spread across a grid with wind and weather effects.
    
    Wind affects spread probability:
    - Downwind: higher probability (wind pushes fire)
    - Upwind: lower probability (fire fights wind)
    - Perpendicular: baseline probability
    
    Spread model:
    P(spread) = base_lambda * wind_factor * weather_factor
    """
    
    # Direction vectors (row, col) - increases row south
    DIRECTION_VECTORS = {
        "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
        "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)
    }
    
    # Wind multipliers (downwind gets boost, upwind gets penalty)
    WIND_MULTIPLIERS = {
        "N": {"N": 3.0, "NE": 2.0, "E": 1.0, "SE": 0.5, "S": 0.3, "SW": 0.5, "W": 1.0, "NW": 2.0},
        "NE": {"N": 2.0, "NE": 3.0, "E": 2.0, "SE": 1.0, "S": 0.5, "SW": 0.3, "W": 0.5, "NW": 1.0},
        "E": {"N": 1.0, "NE": 2.0, "E": 3.0, "SE": 2.0, "S": 1.0, "SW": 0.5, "W": 0.3, "NW": 0.5},
        "SE": {"N": 0.5, "NE": 1.0, "E": 2.0, "SE": 3.0, "S": 2.0, "SW": 1.0, "W": 0.5, "NW": 0.3},
        "S": {"N": 0.3, "NE": 0.5, "E": 1.0, "SE": 2.0, "S": 3.0, "SW": 2.0, "W": 1.0, "NW": 0.5},
        "SW": {"N": 0.5, "NE": 0.3, "E": 0.5, "SE": 1.0, "S": 2.0, "SW": 3.0, "W": 2.0, "NW": 1.0},
        "W": {"N": 1.0, "NE": 0.5, "E": 0.3, "SE": 0.5, "S": 1.0, "SW": 2.0, "W": 3.0, "NW": 2.0},
        "NW": {"N": 2.0, "NE": 1.0, "E": 0.5, "SE": 0.3, "S": 0.5, "SW": 1.0, "W": 2.0, "NW": 3.0}
    }
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.total_nodes = config.grid_size ** 2
        self.burning: Set[int] = set()
        self.burned: Set[int] = set()
        self.firefighters: Dict[int, int] = {}  # node -> time_step_placed
        self.protected: Set[int] = set()
        self.time_step = 0
        self.history: List[Dict] = []
        self.grid = [[0 for _ in range(config.grid_size)] for _ in range(config.grid_size)]
        
        random.seed(config.seed)
        
        # Initialize burning nodes
        for node in config.fire_start_nodes:
            self.burning.add(node)
            row, col = self._node_to_coords(node)
            self.grid[row][col] = 2  # 2 = burning
    
    def _node_to_coords(self, node: int) -> Tuple[int, int]:
        """Convert node index to (row, col) coordinates."""
        return node // self.grid_size, node % self.grid_size
    
    def _coords_to_node(self, row: int, col: int) -> Optional[int]:
        """Convert (row, col) to node index, or None if out of bounds."""
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return row * self.grid_size + col
        return None
    
    def _get_neighbors(self, node: int) -> List[int]:
        """Get all valid neighbor nodes (8-directional)."""
        row, col = self._node_to_coords(node)
        neighbors = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = self._coords_to_node(row + dr, col + dc)
                if neighbor is not None:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _get_spread_direction(self, from_node: int, to_node: int) -> str:
        """Get relative direction from source to target."""
        from_row, from_col = self._node_to_coords(from_node)
        to_row, to_col = self._node_to_coords(to_node)
        
        dr = to_row - from_row
        dc = to_col - from_col
        
        if dr < 0 and dc == 0: return "N"
        if dr < 0 and dc > 0: return "NE"
        if dr == 0 and dc > 0: return "E"
        if dr > 0 and dc > 0: return "SE"
        if dr > 0 and dc == 0: return "S"
        if dr > 0 and dc < 0: return "SW"
        if dr == 0 and dc < 0: return "W"
        if dr < 0 and dc < 0: return "NW"
        
        return "E"  # Default
    
    def _calculate_spread_prob(self, from_node: int, to_node: int) -> float:
        """
        Calculate spread probability considering wind and weather.
        
        Formula: P = λ × wind_factor × weather_factor
        """
        base_prob = self.config.lambda_spread
        
        # Wind effect
        spread_dir = self._get_spread_direction(from_node, to_node)
        wind_mult = self.WIND_MULTIPLIERS.get(self.config.wind_direction, {}).get(spread_dir, 1.0)
        
        # Wind speed factor (0-100 km/h maps to 1.0-5.0 multiplier)
        wind_speed_factor = 1.0 + (self.config.wind_speed / 25.0)
        
        # Weather factors
        temp_factor = 1.0 + (self.config.temperature - 25.0) / 50.0  # +6% per 5°C above 25
        humidity_factor = 1.0 - (self.config.humidity - 30.0) / 200.0  # -5% per 10% above 30
        dryness_factor = self.config.vegetation_dryness
        
        # Final probability
        prob = base_prob * wind_mult * wind_speed_factor * temp_factor * humidity_factor * dryness_factor
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, prob))
    
    def _get_greedy_node(self) -> Optional[int]:
        """
        Greedy strategy: protect the node that would be burned next by most fires.
        """
        neighbor_counts = defaultdict(int)
        
        for fire_node in self.burning:
            for neighbor in self._get_neighbors(fire_node):
                if neighbor not in self.burning and neighbor not in self.burned:
                    neighbor_counts[neighbor] += 1
        
        if not neighbor_counts:
            return None
        
        return max(neighbor_counts.keys(), key=lambda n: neighbor_counts[n])
    
    def _get_central_node(self) -> Optional[int]:
        """Central strategy: protect node closest to grid center."""
        center = self.grid_size / 2 - 0.5
        
        candidates = []
        for fire_node in self.burning:
            for neighbor in self._get_neighbors(fire_node):
                if neighbor not in self.burning and neighbor not in self.burned:
                    row, col = self._node_to_coords(neighbor)
                    dist = ((row - center) ** 2 + (col - center) ** 2) ** 0.5
                    candidates.append((dist, neighbor))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def place_firefighter(self, strategy: str = "greedy") -> Optional[int]:
        """
        Place a firefighter using the specified strategy.
        Returns the node where firefighter was placed, or None.
        """
        if len(self.firefighters) >= self.config.num_firefighters:
            return None
        
        if strategy == "greedy":
            node = self._get_greedy_node()
        elif strategy == "central":
            node = self._get_central_node()
        else:  # random
            candidates = []
            for fire_node in self.burning:
                for neighbor in self._get_neighbors(fire_node):
                    if neighbor not in self.burning and neighbor not in self.burned:
                        candidates.append(neighbor)
            node = random.choice(candidates) if candidates else None
        
        if node is not None and node not in self.firefighters:
            self.firefighters[node] = self.time_step
            self.protected.add(node)
            row, col = self._node_to_coords(node)
            self.grid[row][col] = 1  # 1 = protected
        
        return node
    
    def step(self) -> bool:
        """
        Advance simulation by one time step.
        Returns True if fire is still spreading, False if contained.
        """
        self.time_step += 1
        
        # Record state
        self.history.append({
            "time_step": self.time_step,
            "burning": len(self.burning),
            "burned": len(self.burned),
            "protected": len(self.protected)
        })
        
        # Place firefighters at start of each step
        for _ in range(self.config.num_firefighters):
            self.place_firefighter()
        
        # Fire spreads
        new_burning = set()
        
        for fire_node in self.burning:
            for neighbor in self._get_neighbors(fire_node):
                if neighbor in self.burning or neighbor in self.burned:
                    continue
                if neighbor in self.protected:
                    continue
                
                # Calculate wind-adjusted spread probability
                spread_prob = self._calculate_spread_prob(fire_node, neighbor)
                
                if random.random() < spread_prob:
                    new_burning.add(neighbor)
        
        # Update state
        self.burned.update(self.burning)
        self.burning = new_burning
        
        # Update grid
        for node in self.burned:
            row, col = self._node_to_coords(node)
            self.grid[row][col] = 3  # 3 = burned
        
        for node in self.burning:
            row, col = self._node_to_coords(node)
            self.grid[row][col] = 2  # 2 = burning
        
        return len(self.burning) > 0
    
    def run(self, firefighter_strategy: str = "greedy", max_steps: int = 50) -> 'SimulationResult':
        """Run the full simulation."""
        # Reset
        self.burning = set(self.config.fire_start_nodes)
        self.burned = set()
        self.firefighters = {}
        self.protected = set()
        self.time_step = 0
        self.history = []
        
        # Initialize grid
        for node in self.config.fire_start_nodes:
            row, col = self._node_to_coords(node)
            self.grid[row][col] = 2
        
        # Run until fire contained or max steps
        while self.step() and self.time_step < max_steps:
            pass
        
        return SimulationResult(
            total_burned=len(self.burned),
            total_protected=len(self.protected),
            time_steps=self.time_step,
            burned_nodes=list(self.burned),
            protected_nodes=list(self.protected),
            firefighter_placements=self.firefighters,
            history=self.history,
            wind_speed=self.config.wind_speed,
            wind_direction=self.config.wind_direction
        )
    
    def compare_strategies(self) -> Dict[str, 'SimulationResult']:
        """Compare all firefighter placement strategies."""
        strategies = ["greedy", "random", "central"]
        results = {}
        
        for strategy in strategies:
            result = self.run(firefighter_strategy=strategy)
            results[strategy] = result
        
        return results
    
    def get_grid_visualization(self) -> List[List[int]]:
        """Return current grid state for visualization."""
        return self.grid


@dataclass
class SimulationResult:
    """Results from a fire spread simulation."""
    total_burned: int
    total_protected: int
    time_steps: int
    burned_nodes: List[int]
    protected_nodes: List[int]
    firefighter_placements: Dict[int, int]
    history: List[Dict]
    
    # Wind info
    wind_speed: float = 0.0
    wind_direction: str = "NE"
    
    def to_dict(self) -> Dict:
        return {
            "total_burned": self.total_burned,
            "total_protected": self.total_protected,
            "time_steps": self.time_steps,
            "burned_nodes": self.burned_nodes,
            "protected_nodes": self.protected_nodes,
            "firefighter_placements": self.firefighter_placements,
            "history": self.history,
            "wind": {"speed": self.wind_speed, "direction": self.wind_direction}
        }


# Convenience function
def run_simulation(
    grid_size: int = 7,
    lambda_spread: float = 0.1,
    firefighters: int = 2,
    wind_speed: float = 30.0,
    wind_direction: str = "NE",
    strategy: str = "greedy"
) -> SimulationResult:
    """Quick simulation function."""
    config = SimulationConfig(
        grid_size=grid_size,
        lambda_spread=lambda_spread,
        num_firefighters=firefighters,
        wind_speed=wind_speed,
        wind_direction=wind_direction
    )
    simulator = FireSpreadSimulator(config)
    return simulator.run(firefighter_strategy=strategy)


if __name__ == "__main__":
    # Demo with wind
    print("🔥 Fire Spread Simulation with Wind")
    print("=" * 50)
    
    result = run_simulation(
        grid_size=7,
        lambda_spread=0.1,
        firefighters=2,
        wind_speed=50,  # 50 km/h wind
        wind_direction="NE"
    )
    
    print(f"Wind: {result.wind_speed} km/h {result.wind_direction}")
    print(f"Burned: {result.total_burned} nodes")
    print(f"Protected: {result.total_protected} nodes")
    print(f"Time steps: {result.time_steps}")
