# Moving Firefighter Problem Data Loader

A comprehensive Python module for loading, analyzing, and visualizing Moving Firefighter Problem instances and solutions saved as JSON files.

## 📁 Files Overview

- `mfp_data_loader.py` - Main data loader class and utilities
- `demo_data_loader.py` - Demonstration script showing all functionality
- `README_DataLoader.md` - This documentation file

## 🚀 Quick Start

```python
from mfp_data_loader import MovingFirefighterDataLoader

# Load a single instance
loader = MovingFirefighterDataLoader()
data = loader.load_data('problem.json', 'solution.json')

# Get summary
summary = loader.get_data_summary()
print(summary)

# Visualize
loader.visualize_problem(save_path='analysis.png')
```

## 📊 Data Structure Explained

### Problem File (`*_problem.json`)

```json
{
  "metadata": {
    "created_at": "2024-01-01T12:00:00",
    "description": "Moving Firefighter Problem Instance",
    "version": "1.0"
  },
  "parameters": {
    "n": 15,                    // Number of vertices
    "lambda_d": 2.0,           // Distance multiplier
    "burnt_nodes": 1,          // Initial fires count
    "D": 3,                    // Defense rounds limit
    "B": 5,                    // Burning rounds limit
    "seed": 42                 // Random seed
  },
  "graph": {
    "adjacency_matrix": [...], // n×n connectivity matrix
    "distance_matrix": [...],  // (n+1)×(n+1) travel times
    "burnt_nodes": [8],        // Initially burning vertices
    "coordinates": [...]       // 3D positions (optional)
  }
}
```

### Solution File (`*_solution.json`)

```json
{
  "metadata": {
    "solver": "SCIP",          // Solver used
    "problem_file": "..."      // Linked problem file
  },
  "solution": {
    "feasible": true,          // Solution found?
    "objective": 8.0,          // Vertices burned
    "runtime": 7.89,           // Solving time
    "defense_sequence": [      // Firefighter actions
      [15, 0, 0],             // [vertex, burn_round, def_round]
      [2, 1, 0],              // Move to vertex 2
      [2, 1, 1]               // Stay at vertex 2
    ]
  },
  "analysis": {
    "vertices_saved": 7,       // Performance metrics
    "defended_vertices": 4
  }
}
```

## 🔧 Key Components

### MovingFirefighterDataLoader Class

The main class providing all loading and analysis functionality:

#### Methods:
- `load_data(problem_file, solution_file)` - Load JSON files
- `get_data_summary()` - Extract key statistics
- `visualize_problem()` - Create comprehensive plots
- `export_to_dataframe()` - Convert to pandas DataFrame

#### Data Access:
- `loader.problem_data` - Raw problem JSON
- `loader.solution_data` - Raw solution JSON
- `loader.graph` - Reconstructed graph object

### Utility Functions

- `load_multiple_instances(pattern)` - Batch load matching files
- `create_comparative_analysis(loaders)` - Compare multiple instances

## 📈 Analysis Capabilities

### Summary Statistics
```python
summary = loader.get_data_summary()
# Returns:
{
  "problem_info": {
    "vertices": 15,
    "lambda": 2.0,
    "graph_density": 0.267
  },
  "solution_info": {
    "success_rate": 46.7,
    "runtime": 7.89
  },
  "strategy_info": {
    "movements": 3,
    "reinforcements": 7
  }
}
```

### Visualization
- **Problem State**: Initial fires and graph structure
- **Distance Matrix**: Travel time heatmap
- **Solution Strategy**: Firefighter path and defended vertices
- **Performance Stats**: Summary metrics and success rates

### DataFrame Export
Convert to pandas DataFrame for further analysis:
```python
df = loader.export_to_dataframe()
# Columns include: problem_vertices, solution_runtime, strategy_movements, etc.
```

## 🎯 Data Interpretation Guide

### Adjacency Matrix
- `adjacency_matrix[i][j] = 1` → vertices i and j are connected
- Symmetric (undirected graph)
- Size: n×n

### Distance Matrix
- `distance_matrix[i][j]` → travel time from vertex i to j
- Includes anchor vertex (row/column n)
- Size: (n+1)×(n+1)
- Values scaled by λ parameter

### Defense Sequence
Each action: `[vertex_id, burning_round, defense_round]`
- **vertex_id**: Firefighter position
- **burning_round**: Fire spread cycle
- **defense_round**: Action within that cycle
- **Consecutive duplicates**: Reinforcement (staying put)

### Objective Value
- Number of vertices that ultimately burn
- **Lower is better** (minimize damage)
- Range: [initial_fires, total_vertices]

## 📋 Usage Examples

### Single Instance Analysis
```python
# Load and analyze one problem
loader = MovingFirefighterDataLoader()
loader.load_data('mfp_n15_lambda2_b1_problem.json', 
                'mfp_n15_lambda2_b1_solution.json')

# Get firefighter strategy
if loader.solution_data:
    defense_seq = loader.solution_data['solution']['defense_sequence']
    path = [vertex for vertex, _, _ in defense_seq]
    print(f"Firefighter path: {' → '.join(map(str, path))}")

# Analyze performance
summary = loader.get_data_summary()
success_rate = summary['solution_info']['success_rate']
print(f"Success rate: {success_rate:.1f}%")
```

### Batch Analysis
```python
# Load all instances in directory
loaders = load_multiple_instances('mfp_*_problem.json')

# Create comparative analysis
df = create_comparative_analysis(loaders)

# Find best performing instances
best = df.loc[df['solution_success_rate'].idxmax()]
print(f"Best instance: {best['problem_file']}")
print(f"Success rate: {best['solution_success_rate']:.1f}%")
```

### Visualization
```python
# Create comprehensive visualization
loader.visualize_problem(figsize=(15, 10), save_path='analysis.png')

# Custom visualization using raw data
import matplotlib.pyplot as plt
import networkx as nx

adj_matrix = loader.graph.A
G = nx.from_numpy_array(adj_matrix)
nx.draw(G, with_labels=True)
plt.show()
```

## 🔍 Troubleshooting

### Common Issues:

1. **File Not Found**
   ```
   FileNotFoundError: Problem file not found
   ```
   - Ensure JSON files exist in the specified path
   - Check file naming convention

2. **Invalid JSON**
   ```
   json.JSONDecodeError: Expecting value
   ```
   - File may be corrupted or incomplete
   - Try re-saving from the notebook

3. **Missing Solution Data**
   ```
   No solution data available
   ```
   - Solution file not found or not provided
   - Analysis will work with problem data only

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'networkx'
   ```
   - Install required dependencies:
   ```bash
   pip install numpy matplotlib networkx pandas
   ```

## 📦 Dependencies

Required packages:
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `networkx` - Graph operations
- `pandas` - DataFrame operations
- `json` - JSON file handling (built-in)
- `os` - File operations (built-in)

## 🎯 Best Practices

1. **File Organization**
   - Keep problem and solution files together
   - Use descriptive naming conventions
   - Organize by experiment or parameter sets

2. **Batch Analysis**
   - Load related instances together
   - Use comparative analysis for insights
   - Export to CSV for external analysis

3. **Visualization**
   - Save plots for documentation
   - Use appropriate figure sizes
   - Include metadata in plot titles

4. **Data Validation**
   - Check data integrity after loading
   - Verify problem-solution correspondence
   - Handle missing or incomplete data gracefully

## 📝 Example Workflow

```python
# 1. Load your data
loader = MovingFirefighterDataLoader()
loader.load_data('my_problem.json', 'my_solution.json')

# 2. Understand the data structure
summary = loader.get_data_summary()
print("Problem summary:", summary['problem_info'])
print("Solution summary:", summary['solution_info'])

# 3. Visualize results
loader.visualize_problem(save_path='my_analysis.png')

# 4. Export for further analysis
df = loader.export_to_dataframe()
df.to_csv('my_results.csv', index=False)

# 5. Compare with other instances
all_loaders = load_multiple_instances('mfp_*_problem.json')
comparison = create_comparative_analysis(all_loaders)
print(comparison.describe())
```

This comprehensive data loader enables deep analysis of Moving Firefighter Problem instances, supporting both individual case studies and large-scale comparative research. 