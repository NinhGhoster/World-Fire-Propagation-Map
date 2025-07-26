# modules/simulation.py
import numpy as np

# This is the corrected function name.
# It is now 'run_simulation' to match what callbacks.py expects.
def run_simulation(grid_size=100, burn_probability=0.4):
    """
    A simplified placeholder for a fire spread simulation.
    In a real model, this would take inputs like wind, vegetation, and topography.
    """
    # Create a grid: 0=unburned, 1=burning, 2=burned
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Start a fire in the center
    center = grid_size // 2
    grid[center, center] = 1

    # This is a placeholder and does not currently animate the spread.
    # It just demonstrates that the simulation function can be called.
    print(f"Placeholder simulation run with grid size {grid_size}x{grid_size}.")

    # In a real app, this would return the simulation results (e.g., a series of grids).
    return None