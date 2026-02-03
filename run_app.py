#!/usr/bin/env python3
"""
World Fire Propagation Map - Application Launcher

Simple launcher with browser auto-open and status display.
"""
import subprocess
import sys
import webbrowser
import time
import threading
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required = ["dash", "pandas", "numpy", "networkx", "plotly"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    
    return missing


def install_dependencies(missing):
    """Install missing dependencies."""
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)


def run_demo_notebook():
    """Run a Jupyter notebook demo."""
    notebook_path = Path(__file__).parent / "demo.ipynb"
    if notebook_path.exists():
        subprocess.run(["jupyter", "notebook", str(notebook_path)], check=True)
    else:
        print("Demo notebook not found. Run the dashboard instead.")


def main():
    print("=" * 60)
    print("🔥 WORLD FIRE PROPAGATION MAP")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        install_dependencies(missing)
        print()
    
    # Configuration check
    print("Configuration:")
    from config import Config, get_config
    config = get_config()
    is_valid, errors = config.validate()
    
    if is_valid:
        print(f"  ✅ API Key: Configured")
        print(f"  ✅ Version: {Config.VERSION}")
        print(f"  ✅ Debug: {Config.DEBUG}")
    else:
        print("  ⚠️  Configuration issues:")
        for error in errors:
            print(f"     - {error}")
        print()
        print("  Create .env file with FIRMS_API_KEY")
    
    print()
    print("Options:")
    print("  1. Run Dashboard (web interface)")
    print("  2. Run API Server only")
    print("  3. Run Simulation Demo")
    print("  4. Run Benchmarks")
    print("  5. Run Tests")
    print("  6. Exit")
    print()
    
    choice = input("Enter choice [1-6]: ").strip()
    print()
    
    if choice == "1":
        print("Starting Dashboard...")
        print("=" * 60)
        
        # Open browser after delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8050")
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the app
        subprocess.run([sys.executable, "app.py"])
    
    elif choice == "2":
        print("Starting API Server on port 8051...")
        print("=" * 60)
        subprocess.run([sys.executable, "-c", 
            "from modules.api import api_app; api_app.run(host='0.0.0.0', port=8051)"])
    
    elif choice == "3":
        print("Running Simulation Demo...")
        print("=" * 60)
        subprocess.run([sys.executable, "-c", """
from modules.simulation import FireSpreadSimulator, SimulationConfig

print("🔥 Fire Spread Simulation Demo")
print("=" * 40)

# Create simulation
config = SimulationConfig(
    grid_size=7,
    lambda_spread=0.005,
    num_firefighters=1,
    fire_start_nodes=[24],  # Center of 7x7 grid
    seed=42
)

simulator = FireSpreadSimulator(config)

# Run with greedy strategy
print("\\nGreedy Strategy:")
result = simulator.run(firefighter_strategy="greedy")
print(f"  Burned: {result.total_burned}")
print(f"  Protected: {result.total_protected}")
print(f"  Time Steps: {result.time_steps}")

# Compare strategies
print("\\nStrategy Comparison:")
simulator.reset()
comparison = simulator.compare_strategies()

for strategy, res in comparison.items():
    print(f"  {strategy.capitalize():8s}: {res.total_burned:2d} burned, {res.total_protected:2d} protected")

# Show visualization
print("\\nFinal State (ASCII):")
print(simulator.get_grid_visualization().__str__().replace('0', '░').replace('1', '🔥').replace('2', '🛡').replace('3', '⬛'))
"""])
    
    elif choice == "4":
        print("Running Benchmarks...")
        print("=" * 60)
        subprocess.run([sys.executable, "-m", "modules.benchmarks"])
    
    elif choice == "5":
        print("Running Tests...")
        print("=" * 60)
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
    
    elif choice == "6":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
