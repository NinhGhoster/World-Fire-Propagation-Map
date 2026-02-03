"""
World Fire Propagation Map - Fire Spread Simulation Module

Simulates fire spread on grid graphs with configurable parameters.
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random


@dataclass
class SimulationConfig:
    """Configuration for fire spread simulation."""
    grid_size: int = 7
    lambda_spread: float = 0.005  # Fire spread probability
    num_firefighters: int = 1  # B value
    fire_start_nodes: List[int] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Results from a fire spread simulation."""
    total_burned: int = 0
    total_protected: int = 0
    firefighters_used: int = 0
    time_steps: int = 0
    burned_nodes: List[int] = field(default_factory=list)
    protected_nodes: List[int] = field(default_factory=list)
    firefighter_placements: Dict[int, int] = field(default_factory=dict)
    fire_progression: List[set] = field(default_factory=list)


class FireSpreadSimulator:
    """Simulates fire spread on a grid graph."""
    
    # Cardinal directions for grid neighbors
    DIRECTIONS = [-1, 1, -7, 7]  # Up, Down, Left, Right for 7x7 grid
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.grid_size = config.grid_size
        self.num_nodes = self.grid_size ** 2
        
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        self.graph = self._create_grid_graph()
        self.reset()
    
    def _create_grid_graph(self) -> nx.Graph:
        """Create a grid graph for the simulation."""
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.num_nodes):
            G.add_node(i)
        
        # Add edges (cardinal directions only)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                node = row * self.grid_size + col
                
                # Up
                if row > 0:
                    G.add_edge(node, node - self.grid_size)
                # Down
                if row < self.grid_size - 1:
                    G.add_edge(node, node + self.grid_size)
                # Left
                if col > 0:
                    G.add_edge(node, node - 1)
                # Right
                if col < self.grid_size - 1:
                    G.add_edge(node, node + 1)
        
        return G
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.burning = set(self.config.fire_start_nodes)
        self.protected = set()
        self.burned = set()
        self.firefighter_placements = {}
        self.time_step = 0
        self.fire_progression = [set(self.burning)]
    
    def get_neighbors(self, node: int) -> List[int]:
        """Get neighboring nodes of a burning node."""
        neighbors = []
        row, col = node // self.grid_size, node % self.grid_size
        
        # Up
        if row > 0:
            neighbors.append(node - self.grid_size)
        # Down
        if row < self.grid_size - 1:
            neighbors.append(node + self.grid_size)
        # Left
        if col > 0:
            neighbors.append(node - 1)
        # Right
        if col < self.grid_size - 1:
            neighbors.append(node + 1)
        
        return neighbors
    
    def step(self, firefighter_nodes: List[int] = None) -> bool:
        """
        Run one time step of the simulation.
        
        Args:
            firefighter_nodes: Nodes where firefighters are deployed this turn
        
        Returns:
            True if simulation is still ongoing, False if complete
        """
        if firefighter_nodes is None:
            firefighter_nodes = []
        
        # Deploy firefighters
        for i, node in enumerate(firefighter_nodes[:self.config.num_firefighters]):
            if node not in self.burning and node not in self.protected:
                self.protected.add(node)
                self.firefighter_placements[self.time_step] = node
        
        # Fire spreads from burning nodes
        new_burning = set()
        
        for node in self.burning:
            if node in self.protected:
                continue  # Firefighter contained this node
            
            neighbors = self.get_neighbors(node)
            
            for neighbor in neighbors:
                if (neighbor not in self.burning and 
                    neighbor not in self.burned and 
                    neighbor not in self.protected):
                    
                    # Fire spreads with probability lambda
                    if random.random() < self.config.lambda_spread:
                        new_burning.add(neighbor)
        
        # Update state
        self.burned.update(self.burning)
        self.burning = new_burning
        self.time_step += 1
        self.fire_progression.append(set(self.burning))
        
        # Check if simulation is complete
        return len(self.burning) > 0 and len(self.burned) < self.num_nodes
    
    def run(self, firefighter_strategy: str = "greedy") -> SimulationResult:
        """
        Run the full simulation with a firefighter strategy.
        
        Args:
            firefighter_strategy: Strategy for firefighter placement
                - "greedy": Protect highest degree nodes
                - "random": Random placement
                - "central": Protect center of fire
                - "manual": Use predefined placements
        
        Returns:
            SimulationResult with final state
        """
        self.reset()
        
        while self.step():
            if firefighter_strategy == "greedy":
                # Protect highest degree node adjacent to fire
                nodes_to_protect = self._greedy_placement()
            elif firefighter_strategy == "random":
                nodes_to_protect = self._random_placement()
            elif firefighter_strategy == "central":
                nodes_to_protect = self._central_placement()
            else:
                nodes_to_protect = []
            
            self.step(nodes_to_protect)
        
        return SimulationResult(
            total_burned=len(self.burned),
            total_protected=len(self.protected),
            firefighters_used=len(self.firefighter_placements),
            time_steps=self.time_step,
            burned_nodes=list(self.burned),
            protected_nodes=list(self.protected),
            firefighter_placements=self.firefighter_placements,
            fire_progression=self.fire_progression
        )
    
    def _greedy_placement(self) -> List[int]:
        """Protect highest degree nodes adjacent to fire."""
        candidates = set()
        
        for node in self.burning:
            neighbors = self.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in self.burning and neighbor not in self.burned:
                    candidates.add(neighbor)
        
        if not candidates:
            return []
        
        # Sort by degree (highest first)
        sorted_candidates = sorted(candidates, 
                                   key=lambda x: self.graph.degree(x), 
                                   reverse=True)
        
        return [sorted_candidates[0]] if sorted_candidates else []
    
    def _random_placement(self) -> List[int]:
        """Random firefighter placement."""
        candidates = set()
        
        for node in self.burning:
            neighbors = self.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in self.burning and neighbor not in self.burned:
                    candidates.add(neighbor)
        
        if not candidates:
            return []
        
        return [random.choice(list(candidates))]
    
    def _central_placement(self) -> List[int]:
        """Protect center of fire spread."""
        if not self.burning:
            return []
        
        # Find centroid of burning nodes
        centroid = sum(self.burning) // len(self.burning)
        
        # Find closest unprotected node to centroid
        candidates = set()
        for node in self.burning:
            neighbors = self.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in self.burning and neighbor not in self.burned:
                    candidates.add(neighbor)
        
        if not candidates:
            return []
        
        closest = min(candidates, key=lambda x: abs(x - centroid))
        return [closest]
    
    def get_grid_visualization(self) -> np.ndarray:
        """Get a grid visualization of the current state.
        
        Returns:
            Grid where 0=unburned, 1=burning, 2=protected, 3=burned
        """
        grid = np.zeros(self.grid_size ** 2, dtype=int)
        
        for i in range(self.num_nodes):
            if i in self.burning:
                grid[i] = 1
            elif i in self.protected:
                grid[i] = 2
            elif i in self.burned:
                grid[i] = 3
        
        return grid.reshape(self.grid_size, self.grid_size)
    
    def compare_strategies(self) -> Dict[str, SimulationResult]:
        """Compare all firefighter placement strategies."""
        strategies = ["greedy", "random", "central"]
        results = {}
        
        for strategy in strategies:
            result = self.run(firefighter_strategy=strategy)
            results[strategy] = result
            self.reset()  # Reset for next strategy
        
        return results


def run_parameter_sweep(
    grid_size: int = 7,
    lambda_values: List[float] = None,
    firefighter_values: List[int] = None,
    num_trials: int = 10
) -> Dict:
    """
    Run a parameter sweep for sensitivity analysis.
    
    Args:
        grid_size: Size of the grid
        lambda_values: Fire spread probabilities to test
        firefighter_values: Number of firefighters to test
        num_trials: Number of trials per configuration
    
    Returns:
        Dictionary with results
    """
    if lambda_values is None:
        lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    if firefighter_values is None:
        firefighter_values = [1, 2, 3]
    
    results = {
        "configurations": [],
        "summary": {}
    }
    
    for lambda_val in lambda_values:
        for num_ff in firefighter_values:
            config = SimulationConfig(
                grid_size=grid_size,
                lambda_spread=lambda_val,
                num_firefighters=num_ff,
                fire_start_nodes=[grid_size ** 2 // 2]  # Start in center
            )
            
            simulator = FireSpreadSimulator(config)
            strategy_results = simulator.compare_strategies()
            
            config_result = {
                "lambda": lambda_val,
                "firefighters": num_ff,
                "results": {}
            }
            
            for strategy, result in strategy_results.items():
                config_result["results"][strategy] = {
                    "burned": result.total_burned,
                    "protected": result.total_protected,
                    "time_steps": result.time_steps
                }
            
            results["configurations"].append(config_result)
    
    return results


if __name__ == "__main__":
    # Example usage
    config = SimulationConfig(
        grid_size=7,
        lambda_spread=0.005,
        num_firefighters=1,
        fire_start_nodes=[24],  # Center of 7x7 grid
        seed=42
    )
    
    simulator = FireSpreadSimulator(config)
    result = simulator.run(firefighter_strategy="greedy")
    
    print(f"Simulation Results:")
    print(f"  Burned nodes: {result.total_burned}")
    print(f"  Protected nodes: {result.total_protected}")
    print(f"  Time steps: {result.time_steps}")
    print(f"  Firefighter placements: {result.firefighter_placements}")
    
    # Compare strategies
    print("\nStrategy Comparison:")
    simulator.reset()
    comparison = simulator.compare_strategies()
    
    for strategy, res in comparison.items():
        print(f"  {strategy}: {res.total_burned} burned, {res.total_protected} protected")
