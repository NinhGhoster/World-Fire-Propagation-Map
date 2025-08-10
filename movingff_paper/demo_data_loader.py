"""
Demonstration Script: Moving Firefighter Problem Data Loader

This script demonstrates how to use the MovingFirefighterDataLoader class
to load, analyze, and visualize Moving Firefighter Problem data.

Run this script to see examples of all the loader functionality.
"""

from mfp_data_loader import MovingFirefighterDataLoader, load_multiple_instances, create_comparative_analysis
import os
import matplotlib.pyplot as plt


def demonstrate_single_instance_loading():
    """Demonstrate loading and analyzing a single problem instance."""
    print("🔥 DEMONSTRATION: Single Instance Loading")
    print("=" * 60)
    
    # Create a data loader
    loader = MovingFirefighterDataLoader()
    
    # Example file names (you would replace these with your actual files)
    problem_file = "mfp_n15_lambda2_b1_20240101_120000_problem.json"
    solution_file = "mfp_n15_lambda2_b1_20240101_120000_solution.json"
    
    # Check if example files exist
    if not os.path.exists(problem_file):
        print(f"⚠️  Example files not found. Please save some data first using the notebook.")
        print(f"   Looking for: {problem_file}")
        return None
    
    try:
        # Load the data
        data = loader.load_data(problem_file, solution_file)
        
        print("\n" + "=" * 60)
        print("✅ LOADING COMPLETE - Here's what each part contains:")
        print("=" * 60)
        
        # Show data structure breakdown
        print("\n📊 DATA STRUCTURE BREAKDOWN:")
        print("-" * 40)
        
        print("🏷️  data['problem']['metadata']:")
        print("   Contains creation timestamp, description, version info")
        print(f"   Example: {data['problem']['metadata']}")
        
        print("\n⚙️  data['problem']['parameters']:")
        print("   Contains all problem setup parameters:")
        print("   • n: Number of vertices")
        print("   • lambda_d: Distance multiplier") 
        print("   • burnt_nodes: Number of initial fires")
        print("   • D, B: Defense and burning round limits")
        print("   • seed: Random seed for reproducibility")
        print(f"   Example: {data['problem']['parameters']}")
        
        print("\n🔗 data['problem']['graph']:")
        print("   Contains the complete graph structure:")
        print("   • adjacency_matrix: Which vertices are connected")
        print("   • distance_matrix: Travel times between all vertex pairs")
        print("   • burnt_nodes: List of initially burning vertices")
        print("   • coordinates: 3D spatial positions (if available)")
        
        adj_shape = len(data['problem']['graph']['adjacency_matrix'])
        dist_shape = len(data['problem']['graph']['distance_matrix'])
        print(f"   Adjacency matrix: {adj_shape}×{adj_shape}")
        print(f"   Distance matrix: {dist_shape}×{dist_shape}")
        print(f"   Initial fires: {data['problem']['graph']['burnt_nodes']}")
        
        if data['solution']:
            print("\n🎯 data['solution']['solution']:")
            print("   Contains the optimization results:")
            print("   • feasible: Whether a solution was found")
            print("   • objective: Final number of burned vertices")
            print("   • runtime: Time taken to solve")
            print("   • defense_sequence: Complete firefighter action sequence")
            print("   • distances: Travel distances for each move")
            
            sol = data['solution']['solution']
            print(f"   Feasible: {sol['feasible']}")
            print(f"   Objective: {sol['objective']} vertices burned")
            print(f"   Runtime: {sol['runtime']:.2f} seconds")
            
            defense_seq = sol['defense_sequence']
            if defense_seq:
                print(f"   Defense sequence length: {len(defense_seq)} actions")
                print(f"   First 3 actions: {defense_seq[:3]}")
                print("   Each action format: (vertex, burning_round, defense_round)")
            
            print("\n📈 data['solution']['analysis']:")
            print("   Contains performance metrics:")
            print("   • total_vertices: Graph size")
            print("   • vertices_saved: How many were protected")
            print("   • defended_vertices: How many were actively defended")
            
            analysis = data['solution']['analysis']
            print(f"   Total vertices: {analysis['total_vertices']}")
            print(f"   Vertices saved: {analysis['vertices_saved']}")
            print(f"   Success rate: {(analysis['vertices_saved']/analysis['total_vertices']*100):.1f}%")
        
        # Get and display summary
        print("\n📋 SUMMARY ANALYSIS:")
        print("-" * 30)
        summary = loader.get_data_summary()
        
        for section, data_dict in summary.items():
            print(f"\n{section.upper()}:")
            for key, value in data_dict.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.3f}")
                else:
                    print(f"   • {key}: {value}")
        
        # Create visualization
        print("\n🎨 CREATING VISUALIZATION...")
        loader.visualize_problem(figsize=(15, 10), save_path="mfp_analysis_demo.png")
        
        # Export to DataFrame
        print("\n📊 EXPORTING TO DATAFRAME...")
        df = loader.export_to_dataframe()
        print("DataFrame columns:")
        for col in df.columns:
            print(f"   • {col}: {df[col].iloc[0]}")
        
        return loader
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def demonstrate_batch_loading():
    """Demonstrate loading multiple instances for comparative analysis."""
    print("\n\n🔥 DEMONSTRATION: Batch Loading and Analysis")
    print("=" * 60)
    
    # Look for multiple problem files
    loaders = load_multiple_instances("mfp_*_problem.json", ".")
    
    if not loaders:
        print("⚠️  No problem files found for batch analysis.")
        print("   Save multiple instances using the notebook first.")
        return
    
    print(f"\n📊 LOADED {len(loaders)} INSTANCES")
    print("-" * 40)
    
    # Show what each loader contains
    for i, loader in enumerate(loaders[:3]):  # Show first 3 for brevity
        print(f"\nINSTANCE {i+1}:")
        summary = loader.get_data_summary()
        prob_info = summary.get('problem_info', {})
        sol_info = summary.get('solution_info', {})
        
        print(f"   • Vertices: {prob_info.get('vertices', 'N/A')}")
        print(f"   • Lambda: {prob_info.get('lambda', 'N/A')}")
        print(f"   • Initial fires: {prob_info.get('initial_fires', 'N/A')}")
        
        if sol_info:
            print(f"   • Success rate: {sol_info.get('success_rate', 0):.1f}%")
            print(f"   • Runtime: {sol_info.get('runtime', 'N/A'):.2f}s" if isinstance(sol_info.get('runtime'), (int, float)) else f"   • Runtime: N/A")
    
    if len(loaders) > 3:
        print(f"   ... and {len(loaders) - 3} more instances")
    
    # Create comparative analysis
    print(f"\n📈 CREATING COMPARATIVE ANALYSIS...")
    analysis_df = create_comparative_analysis(loaders, save_path="comparative_analysis.csv")
    
    if not analysis_df.empty:
        print("\n🔍 COMPARATIVE ANALYSIS RESULTS:")
        print("   This DataFrame contains all instances with columns for:")
        print("   • Problem parameters (vertices, lambda, etc.)")
        print("   • Solution quality (success rate, objective)")
        print("   • Performance metrics (runtime, strategy info)")
        print("   • Metadata (creation time, file names)")
        
        print(f"\n   Total instances analyzed: {len(analysis_df)}")
        print(f"   DataFrame shape: {analysis_df.shape}")
        print(f"   Columns: {list(analysis_df.columns)}")


def demonstrate_data_structure_details():
    """Provide detailed explanation of the JSON data structure."""
    print("\n\n🔥 DETAILED DATA STRUCTURE EXPLANATION")
    print("=" * 60)
    
    print("""
📁 PROBLEM FILE STRUCTURE (*.problem.json):
{
  "metadata": {
    "created_at": "2024-01-01T12:00:00",     // Timestamp when saved
    "description": "Moving Firefighter Problem Instance",
    "version": "1.0"                          // Data format version
  },
  
  "parameters": {
    "n": 15,                                  // Number of main vertices
    "lambda_d": 2.0,                         // Distance multiplier (affects travel times)
    "burnt_nodes": 1,                        // Number of initially burning vertices
    "instance": 0,                           // Random instance ID
    "dimension": 3,                          // Spatial dimension for coordinates
    "edge_probability": 0.167,               // Edge probability used in graph generation
    "D": 3,                                  // Defense rounds upper bound
    "B": 5,                                  // Burning rounds upper bound
    "seed": 42                               // Random seed for reproducibility
  },
  
  "graph": {
    "adjacency_matrix": [[0,1,0,...], ...], // n×n matrix: 1 if vertices connected
    "distance_matrix": [[0,1.5,2.1,...], ...], // (n+1)×(n+1) matrix: travel times
    "burnt_nodes": [8],                      // List of initially burning vertex IDs
    "num_vertices": 15,                      // Number of vertices (same as n)
    "num_edges": 12,                         // Total number of edges in graph
    "coordinates": [[x1,y1,z1], [x2,y2,z2], ...] // 3D coordinates (optional)
  }
}

📁 SOLUTION FILE STRUCTURE (*.solution.json):
{
  "metadata": {
    "created_at": "2024-01-01T12:05:00",     // When solution was saved
    "problem_file": "problem.json",           // Corresponding problem file
    "solver": "SCIP",                         // Solver used (SCIP/Gurobi)
    "version": "1.0"
  },
  
  "solution": {
    "feasible": true,                         // Whether solution was found
    "objective": 8.0,                         // Number of vertices that burned
    "runtime": 7.89,                          // Solving time in seconds
    "not_interrupted": true,                  // Whether solver completed normally
    "defense_sequence": [                     // Complete firefighter action sequence
      [15, 0, 0],                            // [vertex, burn_round, defense_round]
      [2, 1, 0],                             // Move to vertex 2 in round 1
      [2, 1, 1],                             // Stay at vertex 2 (reinforcement)
      ...
    ],
    "distances": [1.5, 0.0, 2.1, ...]       // Travel distances for each move
  },
  
  "analysis": {
    "total_vertices": 15,                     // Graph size
    "initially_burning": 1,                   // Number of initial fires
    "final_burned": 8,                        // Final burned count (objective)
    "vertices_saved": 7,                      // How many were saved
    "defended_vertices": 4                    // How many actively defended
  }
}

🔍 KEY DATA INTERPRETATIONS:
""")
    
    print("ADJACENCY MATRIX:")
    print("   • adjacency_matrix[i][j] = 1 means vertices i and j are connected")
    print("   • Symmetric matrix (undirected graph)")
    print("   • Diagonal is always 0 (no self-loops)")
    print("   • Size: n×n where n is number of main vertices")
    
    print("\nDISTANCE MATRIX:")
    print("   • distance_matrix[i][j] = travel time from vertex i to vertex j")
    print("   • Includes anchor vertex (firefighter start) as last row/column")
    print("   • Size: (n+1)×(n+1) including anchor")
    print("   • Multiplied by lambda_d parameter")
    
    print("\nDEFENSE SEQUENCE:")
    print("   • Each entry: [vertex_id, burning_round, defense_round]")
    print("   • vertex_id: Where firefighter is positioned")
    print("   • burning_round: Which fire spread round this occurs in")
    print("   • defense_round: Which defense action within that burning round")
    print("   • Consecutive duplicates = reinforcement (staying at same vertex)")
    
    print("\nOBJECTIVE VALUE:")
    print("   • Number of vertices that ultimately burn")
    print("   • Lower is better (minimize burned vertices)")
    print("   • Best possible: only initially burning vertices")
    print("   • Worst possible: all vertices burn")


def main():
    """Run all demonstrations."""
    print("🎯 Moving Firefighter Problem Data Loader Demonstration")
    print("=" * 70)
    print("This script shows how to load and analyze saved MFP data.")
    print("Make sure you have saved some problem instances first!")
    print("=" * 70)
    
    # Demonstrate single instance loading
    loader = demonstrate_single_instance_loading()
    
    # Demonstrate batch loading
    demonstrate_batch_loading()
    
    # Show data structure details
    demonstrate_data_structure_details()
    
    print("\n\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("You now know how to:")
    print("✅ Load single problem instances")
    print("✅ Understand the data structure")
    print("✅ Extract summary information")
    print("✅ Create visualizations")
    print("✅ Export to pandas DataFrames")
    print("✅ Perform batch analysis")
    print("✅ Compare multiple instances")
    print("\nStart by saving some data using the notebook, then run this script!")


if __name__ == "__main__":
    main() 