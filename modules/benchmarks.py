"""
World Fire Propagation Map - Benchmark Module

Performance benchmarks for the MFF solver and visualization.
"""
import time
import json
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from statistics import mean, stdev
from functools import wraps
import sys


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float = 0.0
    metadata: Dict = field(default_factory=dict)


def benchmark(name: str = None, iterations: int = 100, warmup: int = 10):
    """
    Decorator to benchmark a function.
    
    Args:
        name: Name for the benchmark (defaults to function name)
        iterations: Number of iterations
        warmup: Number of warmup runs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> BenchmarkResult:
            func_name = name or func.__name__
            
            # Warmup runs
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Benchmark runs
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            
            # Calculate statistics
            avg_time = mean(times)
            min_time = min(times)
            max_time = max(times)
            
            std = stdev(times) if len(times) > 1 else 0.0
            
            return BenchmarkResult(
                name=func_name,
                iterations=iterations,
                total_time=sum(times),
                avg_time=avg_time,
                min_time=min_time,
                max_time=max_time,
                std_dev=std
            )
        
        return wrapper
    return decorator


class BenchmarkSuite:
    """Collection of benchmarks for the fire propagation system."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []
        
        print("Running benchmarks...\n")
        
        # Run each benchmark method
        for attr_name in dir(self):
            if attr_name.startswith("benchmark_"):
                method = getattr(self, attr_name)
                if callable(method):
                    try:
                        result = method()
                        if isinstance(result, BenchmarkResult):
                            self.results.append(result)
                            self._print_result(result)
                    except Exception as e:
                        print(f"Error in {attr_name}: {e}\n")
        
        return self.results
    
    def _print_result(self, result: BenchmarkResult):
        """Print a benchmark result."""
        print(f"  {result.name}:")
        print(f"    Avg: {result.avg_time*1000:.3f} ms")
        print(f"    Min: {result.min_time*1000:.3f} ms")
        print(f"    Max: {result.max_time*1000:.3f} ms")
        print(f"    Std: {result.std_dev*1000:.3f} ms")
        print()
    
    def to_json(self) -> str:
        """Export results to JSON."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "total_time": r.total_time,
                    "avg_time": r.avg_time,
                    "min_time": r.min_time,
                    "max_time": r.max_time,
                    "std_dev": r.std_dev,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }
        return json.dumps(data, indent=2)
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60 + "\n")
        
        for result in self.results:
            print(f"{result.name}: {result.avg_time*1000:.2f} ms/iter")
        
        print("\n" + "="*60)
        
        # Fastest and slowest
        if self.results:
            fastest = min(self.results, key=lambda r: r.avg_time)
            slowest = max(self.results, key=lambda r: r.avg_time)
            
            print(f"Fastest: {fastest.name} ({fastest.avg_time*1000:.3f} ms)")
            print(f"Slowest: {slowest.name} ({slowest.avg_time*1000:.3f} ms)")
            print("="*60)


# Specific benchmarks for fire propagation
class FirePropagationBenchmarks(BenchmarkSuite):
    """Benchmarks for fire propagation calculations."""
    
    def __init__(self):
        super().__init__()
        self.grid_size = 7
        self.num_nodes = self.grid_size ** 2
    
    @benchmark(name="grid_creation", iterations=1000, warmup=50)
    def benchmark_grid_creation(self):
        """Benchmark grid graph creation."""
        import networkx as nx
        
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                node = row * self.grid_size + col
                if row > 0:
                    G.add_edge(node, node - self.grid_size)
                if col > 0:
                    G.add_edge(node, node - 1)
    
    @benchmark(name="degree_calculation", iterations=1000, warmup=50)
    def benchmark_degree_calculation(self):
        """Benchmark degree calculation."""
        import networkx as nx
        
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                node = row * self.grid_size + col
                if row > 0:
                    G.add_edge(node, node - self.grid_size)
                if col > 0:
                    G.add_edge(node, node - 1)
        
        degrees = dict(G.degree())
    
    @benchmark(name="neighbor_finding", iterations=1000, warmup=50)
    def benchmark_neighbor_finding(self):
        """Benchmark neighbor finding for all nodes."""
        for node in range(self.num_nodes):
            row, col = node // self.grid_size, node % self.grid_size
            neighbors = []
            if row > 0:
                neighbors.append(node - self.grid_size)
            if row < self.grid_size - 1:
                neighbors.append(node + self.grid_size)
            if col > 0:
                neighbors.append(node - 1)
            if col < self.grid_size - 1:
                neighbors.append(node + 1)
    
    @benchmark(name="simulation_step", iterations=500, warmup=25)
    def benchmark_simulation_step(self):
        """Benchmark a single simulation step."""
        import random
        random.seed(42)
        
        burning = {24}
        protected = set()
        burned = set()
        
        for _ in range(10):  # Multiple steps
            new_burning = set()
            
            for node in burning:
                row, col = node // self.grid_size, node % self.grid_size
                
                neighbors = []
                if row > 0:
                    neighbors.append(node - self.grid_size)
                if row < self.grid_size - 1:
                    neighbors.append(node + self.grid_size)
                if col > 0:
                    neighbors.append(node - 1)
                if col < self.grid_size - 1:
                    neighbors.append(node + 1)
                
                for neighbor in neighbors:
                    if (neighbor not in burning and 
                        neighbor not in burned and 
                        neighbor not in protected and
                        random.random() < 0.005):
                        new_burning.add(neighbor)
            
            burned.update(burning)
            burning = new_burning
    
    @benchmark(name="visualization_grid", iterations=500, warmup=25)
    def benchmark_visualization_grid(self):
        """Benchmark grid visualization creation."""
        import numpy as np
        
        grid = np.zeros(self.num_nodes, dtype=int)
        
        for i in range(self.num_nodes):
            if i % 3 == 0:
                grid[i] = 1  # Burning
            elif i % 5 == 0:
                grid[i] = 2  # Protected
            elif i % 7 == 0:
                grid[i] = 3  # Burned
        
        # Reshape to grid
        grid = grid.reshape(self.grid_size, self.grid_size)
    
    @benchmark(name="strategy_comparison", iterations=100, warmup=10)
    def benchmark_strategy_comparison(self):
        """Benchmark strategy comparison."""
        import random
        random.seed(42)
        
        grid_size = 7
        num_nodes = grid_size ** 2
        burning = {24}
        protected = set()
        
        strategies = ["greedy", "random", "central"]
        
        for _ in range(10):  # Multiple steps
            for strategy in strategies:
                if strategy == "greedy":
                    # Simple greedy: protect first available neighbor
                    candidates = set()
                    for node in burning:
                        row, col = node // grid_size, node % grid_size
                        if row > 0:
                            candidates.add(node - grid_size)
                        if row < grid_size - 1:
                            candidates.add(node + grid_size)
                        if col > 0:
                            candidates.add(node - 1)
                        if col < grid_size - 1:
                            candidates.add(node + 1)
                    candidates = candidates - burning - protected
                    
                    if candidates:
                        protected.add(next(iter(candidates)))
                
                # Spread fire
                new_burning = set()
                for node in burning:
                    row, col = node // grid_size, node % grid_size
                    neighbors = []
                    if row > 0:
                        neighbors.append(node - grid_size)
                    if row < grid_size - 1:
                        neighbors.append(node + grid_size)
                    if col > 0:
                        neighbors.append(node - 1)
                    if col < grid_size - 1:
                        neighbors.append(node + 1)
                    
                    for neighbor in neighbors:
                        if (neighbor not in burning and 
                            neighbor not in protected and
                            random.random() < 0.005):
                            new_burning.add(neighbor)
                
                burning = new_burning


def run_all_benchmarks() -> BenchmarkSuite:
    """Run all benchmarks and return results."""
    suite = FirePropagationBenchmarks()
    suite.run_all()
    return suite


if __name__ == "__main__":
    print("="*60)
    print("FIRE PROPAGATION BENCHMARKS")
    print("="*60 + "\n")
    
    suite = run_all_benchmarks()
    suite.print_summary()
    
    # Export results
    print("\nExporting results...")
    json_output = suite.to_json()
    
    with open("benchmark_results.json", "w") as f:
        f.write(json_output)
    
    print("Results saved to benchmark_results.json")
