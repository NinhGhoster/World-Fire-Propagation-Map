#!/usr/bin/env python3
"""
World Fire Propagation Map - CLI Interface

Simple command-line interface for quick operations.
"""
import argparse
import sys
import json
from pathlib import Path


def cmd_simulate(args):
    """Run fire spread simulation."""
    from modules.simulation import FireSpreadSimulator, SimulationConfig
    
    config = SimulationConfig(
        grid_size=args.grid_size,
        lambda_spread=args.lambda_spread,
        num_firefighters=args.firefighters,
        fire_start_nodes=[args.start_node or (args.grid_size ** 2) // 2],
        seed=args.seed
    )
    
    simulator = FireSpreadSimulator(config)
    result = simulator.run(firefighter_strategy=args.strategy)
    
    print("\n🔥 SIMULATION RESULTS")
    print("=" * 40)
    print(f"Grid Size:      {config.grid_size}x{config.grid_size}")
    print(f"Lambda (λ):     {config.lambda_spread}")
    print(f"Firefighters:   {config.num_firefighters}")
    print(f"Strategy:       {args.strategy}")
    print()
    print(f"Burned:         {result.total_burned}")
    print(f"Protected:      {result.total_protected}")
    print(f"Time Steps:     {result.time_steps}")
    print()
    print("Firefighter Placements:")
    for timestep, node in sorted(result.firefighter_placements.items()):
        print(f"  Time {timestep}: Node {node}")
    print()
    
    if args.json:
        output = {
            "config": {
                "grid_size": config.grid_size,
                "lambda": config.lambda_spread,
                "firefighters": config.num_firefighters,
                "strategy": args.strategy
            },
            "results": {
                "burned": result.total_burned,
                "protected": result.total_protected,
                "time_steps": result.time_steps
            }
        }
        print(json.dumps(output, indent=2))


def cmd_compare(args):
    """Compare strategies."""
    from modules.simulation import FireSpreadSimulator, SimulationConfig
    
    config = SimulationConfig(
        grid_size=args.grid_size,
        lambda_spread=args.lambda_spread,
        num_firefighters=args.firefighters,
        fire_start_nodes=[args.start_node or (args.grid_size ** 2) // 2],
        seed=args.seed
    )
    
    simulator = FireSpreadSimulator(config)
    comparison = simulator.compare_strategies()
    
    print("\n📊 STRATEGY COMPARISON")
    print("=" * 50)
    print(f"{'Strategy':<12} {'Burned':<10} {'Protected':<12} {'Time Steps':<12}")
    print("-" * 50)
    
    for strategy, result in comparison.items():
        print(f"{strategy.capitalize():<12} {result.total_burned:<10} {result.total_protected:<12} {result.time_steps:<12}")
    
    print("-" * 50)
    best = min(comparison.keys(), key=lambda s: comparison[s].total_burned)
    print(f"Best strategy: {best.capitalize()} (fewest burned nodes)")


def cmd_visualize(args):
    """Visualize simulation."""
    from modules.simulation import FireSpreadSimulator, SimulationConfig
    
    config = SimulationConfig(
        grid_size=args.grid_size,
        lambda_spread=args.lambda_spread,
        num_firefighters=args.firefighters,
        fire_start_nodes=[args.start_node or (args.grid_size ** 2) // 2],
        seed=args.seed
    )
    
    simulator = FireSpreadSimulator(config)
    result = simulator.run(firefighter_strategy=args.strategy)
    
    grid = simulator.get_grid_visualization()
    
    print("\n🔥 FIRE SPREAD MAP")
    print("=" * 40)
    print("Legend: 🔥=Burning, 🛡️=Protected, ⬛=Burned, ░=Unburned")
    print()
    
    for row in range(args.grid_size):
        line = ""
        for col in range(args.grid_size):
            idx = row * args.grid_size + col
            if idx in result.firefighter_placements.values():
                line += "🛡️ "
            elif idx in result.burned_nodes:
                line += "⬛ "
            elif idx in simulator.burning:
                line += "🔥 "
            else:
                line += "░ "
        print(line)
    
    print()
    print(f"Final: {result.total_burned} burned, {result.total_protected} protected")


def cmd_api(args):
    """Test API endpoints."""
    import requests
    
    base_url = f"http://localhost:{args.port}/api/v1"
    
    print(f"\n🌐 API BASE: {base_url}")
    print("=" * 40)
    
    # Test parameters endpoint
    print("\n1. GET /api/v1/parameters")
    try:
        r = requests.get(f"{base_url}/parameters")
        print(f"   Status: {r.status_code}")
        data = r.json()
        print(f"   Grid Sizes: {data.get('grid_sizes', [])}")
        print(f"   Lambda Values: {data.get('lambda_values', [])}")
        print(f"   Strategies: {data.get('strategies', [])}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test simulation endpoint
    print("\n2. POST /api/v1/simulate")
    try:
        r = requests.post(f"{base_url}/simulate", json={
            "grid_size": 7,
            "lambda_spread": 0.005,
            "firefighters": 1,
            "strategy": "greedy",
            "seed": 42
        })
        print(f"   Status: {r.status_code}")
        data = r.json()
        results = data.get("results", {})
        print(f"   Burned: {results.get('total_burned')}")
        print(f"   Protected: {results.get('total_protected')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test compare endpoint
    print("\n3. POST /api/v1/compare")
    try:
        r = requests.post(f"{base_url}/compare", json={
            "grid_size": 7,
            "lambda_spread": 0.005,
            "firefighters": 1
        })
        print(f"   Status: {r.status_code}")
        data = r.json()
        for strategy, res in data.get("results", {}).items():
            print(f"   {strategy}: {res.get('burned')} burned")
    except Exception as e:
        print(f"   Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="World Fire Propagation Map CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py simulate --grid 7 --lambda 0.005 --firefighters 1
  python cli.py compare --grid 7
  python cli.py visualize --grid 7 --lambda 0.01
  python cli.py api --port 8051
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run fire spread simulation")
    sim_parser.add_argument("--grid", dest="grid_size", type=int, default=7, help="Grid size (default: 7)")
    sim_parser.add_argument("--lambda", dest="lambda_spread", type=float, default=0.005, help="Fire spread rate (default: 0.005)")
    sim_parser.add_argument("--firefighters", type=int, default=1, help="Number of firefighters (default: 1)")
    sim_parser.add_argument("--strategy", default="greedy", choices=["greedy", "random", "central"], help="Strategy (default: greedy)")
    sim_parser.add_argument("--start-node", type=int, default=None, help="Starting node (default: center)")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    sim_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare firefighter strategies")
    cmp_parser.add_argument("--grid", dest="grid_size", type=int, default=7, help="Grid size (default: 7)")
    cmp_parser.add_argument("--lambda", dest="lambda_spread", type=float, default=0.005, help="Fire spread rate (default: 0.005)")
    cmp_parser.add_argument("--firefighters", type=int, default=1, help="Number of firefighters (default: 1)")
    cmp_parser.add_argument("--start-node", type=int, default=None, help="Starting node (default: center)")
    cmp_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize fire spread")
    viz_parser.add_argument("--grid", dest="grid_size", type=int, default=7, help="Grid size (default: 7)")
    viz_parser.add_argument("--lambda", dest="lambda_spread", type=float, default=0.005, help="Fire spread rate (default: 0.005)")
    viz_parser.add_argument("--firefighters", type=int, default=1, help="Number of firefighters (default: 1)")
    viz_parser.add_argument("--strategy", default="greedy", choices=["greedy", "random", "central"], help="Strategy (default: greedy)")
    viz_parser.add_argument("--start-node", type=int, default=None, help="Starting node (default: center)")
    viz_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Test API endpoints")
    api_parser.add_argument("--port", type=int, default=8051, help="API port (default: 8051)")
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "api":
        cmd_api(args)
    elif args.command == "version":
        from config import Config
        print(f"World Fire Propagation Map v{Config.VERSION}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
