#!/usr/bin/env python3
"""
Plot Moving Firefighter Problem from JSON files
Reads problem and solution JSON files and creates interactive time step visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import os
import types
from pathlib import Path

# Import the MFP modules
try:
    import moving_firefighter_problem_generator.movingfp.gen as mfp
    from movingff_paper.get_D_value import start_recursion
    from movingff_paper.gens_for_paper import get_seed
    from movingff_paper.miqcp_scip_alternative import mfp_constraints_scip
    SCIP_AVAILABLE = True
    print("✅ SCIP alternative available")
except ImportError as e:
    print(f"⚠️  Some MFP modules not available: {e}")
    SCIP_AVAILABLE = False

# For interactive visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

def get_graph_positioning_cache(graph, n):
    """
    Get or create consistent graph positioning for interactive visualizations.
    
    Parameters:
    - graph: Graph object with A and D attributes
    - n: Number of vertices
    
    Returns:
    - interactive_pos: Dictionary of vertex positions
    - use_cached: Boolean indicating if cache was used
    """
    # Create unique hash for this graph
    graph_hash = hash(str(graph.A.tobytes()) + str(graph.D.tobytes()) + str(n))
    positioning_cache_file = f"graph_positioning_{graph_hash}.json"
    
    # Try to load cached positioning
    interactive_pos = {}
    use_cached = False
    
    try:
        if os.path.exists(positioning_cache_file):
            with open(positioning_cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data['graph_hash'] == graph_hash and cached_data['n'] == n:
                    interactive_pos = {int(k): tuple(v) for k, v in cached_data['positions'].items()}
                    use_cached = True
                    print(f"   ✅ Loaded cached positioning from {positioning_cache_file}")
    except Exception as e:
        print(f"   ⚠️  Could not load cached positioning: {e}")
    
    if not use_cached:
        print("   🔄 Computing new positioning...")
        
        # Check if this is a grid graph
        is_grid = is_grid_graph(graph.A, n)
        
        if is_grid:
            # Use grid layout for grid graphs
            interactive_pos = create_grid_layout(graph.A, n)
            method = 'grid'
            print(f"   ✅ Using grid layout for grid graph")
        else:
            # Create consistent positioning for non-grid graphs
            distance_matrix = graph.D[:n, :n]
            
            # Try MDS positioning first with fixed random state based on graph hash
            try:
                from sklearn.manifold import MDS
                import warnings
                warnings.filterwarnings('ignore')
                
                # Use graph hash as random state for consistent positioning
                consistent_seed = abs(graph_hash) % 2**32  # Ensure positive seed
                
                mds = MDS(n_components=2, dissimilarity='precomputed', 
                         random_state=consistent_seed, max_iter=1000, eps=1e-6, normalized_stress='auto')
                coords_2d = mds.fit_transform(distance_matrix)
                
                for i in range(n):
                    interactive_pos[i] = (coords_2d[i][0], coords_2d[i][1])
                
                method = 'MDS'
                print(f"   ✅ Using MDS embedding for interactive plot (seed: {consistent_seed})")
                
            except ImportError:
                # Fallback to spring layout with distance weights
                G_weighted = nx.Graph()
                for i in range(n):
                    G_weighted.add_node(i)
                
                for i in range(n):
                    for j in range(i+1, n):
                        if distance_matrix[i,j] > 0:
                            weight = 1.0 / distance_matrix[i,j]
                            G_weighted.add_edge(i, j, weight=weight)
                
                # Use graph hash as seed for consistent positioning
                consistent_seed = abs(graph_hash) % 2**32
                interactive_pos = nx.spring_layout(G_weighted, weight='weight', seed=consistent_seed, 
                                                 k=2.0, iterations=500, threshold=1e-6)
                method = 'spring'
                print(f"   Using weighted spring layout for interactive plot (seed: {consistent_seed})")
        
        # Save positioning cache for future use
        try:
            # Convert positions to JSON-serializable format
            positions_serializable = {str(k): list(v) for k, v in interactive_pos.items()}
            cache_data = {
                'positions': positions_serializable,
                'graph_hash': graph_hash,
                'n': n,
                'method': method,
                'created_at': datetime.datetime.now().isoformat()
            }
            
            with open(positioning_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"   💾 Saved positioning cache to {positioning_cache_file}")
        except Exception as e:
            print(f"   ⚠️  Could not save positioning cache: {e}")
    
    return interactive_pos, use_cached

def clear_positioning_cache():
    """Clear all positioning cache files."""
    import glob
    cache_files = glob.glob("graph_positioning_*.json")
    for file in cache_files:
        try:
            os.remove(file)
            print(f"🗑️  Removed cache file: {file}")
        except Exception as e:
            print(f"⚠️  Could not remove {file}: {e}")
    print(f"🧹 Cleared {len(cache_files)} positioning cache files")

def is_grid_graph(adjacency_matrix, n):
    """
    Check if the graph is a grid graph based on adjacency patterns.
    
    Parameters:
    - adjacency_matrix: numpy array of adjacency matrix
    - n: number of vertices
    
    Returns:
    - bool: True if the graph appears to be a grid
    """
    # Check if each node has at most 4 neighbors (grid property)
    for i in range(n):
        degree = np.sum(adjacency_matrix[i])
        if degree > 4:
            return False
    
    # Check if we can find a reasonable grid dimension
    # Try to find square root or close factors
    import math
    sqrt_n = int(math.sqrt(n))
    
    # Check if it's a perfect square grid
    if sqrt_n * sqrt_n == n:
        return check_grid_pattern(adjacency_matrix, sqrt_n, sqrt_n)
    
    # Check other rectangular dimensions
    for rows in range(2, int(math.sqrt(n)) + 2):
        if n % rows == 0:
            cols = n // rows
            if check_grid_pattern(adjacency_matrix, rows, cols):
                return True
    
    return False

def check_grid_pattern(adjacency_matrix, rows, cols):
    """
    Check if the adjacency matrix matches a grid pattern.
    
    Parameters:
    - adjacency_matrix: numpy array of adjacency matrix
    - rows: number of rows in the grid
    - cols: number of columns in the grid
    
    Returns:
    - bool: True if the pattern matches a grid
    """
    n = rows * cols
    if n != len(adjacency_matrix):
        return False
    
    # Check grid connections
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            
            # Expected neighbors in a grid
            expected_neighbors = []
            if i > 0:  # up
                expected_neighbors.append((i-1) * cols + j)
            if i < rows - 1:  # down
                expected_neighbors.append((i+1) * cols + j)
            if j > 0:  # left
                expected_neighbors.append(i * cols + (j-1))
            if j < cols - 1:  # right
                expected_neighbors.append(i * cols + (j+1))
            
            # Check if actual neighbors match expected
            actual_neighbors = [k for k in range(n) if adjacency_matrix[node][k] == 1]
            
            if set(actual_neighbors) != set(expected_neighbors):
                return False
    
    return True

def create_grid_layout(adjacency_matrix, n):
    """
    Create grid layout positions for a grid graph.
    
    Parameters:
    - adjacency_matrix: numpy array of adjacency matrix
    - n: number of vertices
    
    Returns:
    - dict: Dictionary mapping node indices to (x, y) positions
    """
    import math
    
    # Find grid dimensions
    sqrt_n = int(math.sqrt(n))
    rows, cols = sqrt_n, sqrt_n
    
    # If not a perfect square, find the best rectangular fit
    if sqrt_n * sqrt_n != n:
        for r in range(2, int(math.sqrt(n)) + 2):
            if n % r == 0:
                c = n // r
                if check_grid_pattern(adjacency_matrix, r, c):
                    rows, cols = r, c
                    break
    
    positions = {}
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if node < n:
                # Position nodes in a grid pattern
                positions[node] = (j * 1.0, (rows - 1 - i) * 1.0)  # Flip y-axis for standard orientation
    
    print(f"   📐 Grid dimensions: {rows} x {cols}")
    return positions

def load_problem_and_solution(problem_file, solution_file=None):
    """
    Load problem and solution from JSON files.
    
    Parameters:
    - problem_file: Path to the problem JSON file
    - solution_file: Path to the solution JSON file (optional)
    
    Returns:
    - problem_data: Dictionary containing the problem instance
    - solution_data: Dictionary containing the solution (or None)
    - graph: Recreated graph object
    """
    print(f"📂 Loading problem from: {problem_file}")
    
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)
    
    # Recreate graph object
    graph = types.SimpleNamespace()
    graph.A = np.array(problem_data['graph']['adjacency_matrix'])
    graph.D = np.array(problem_data['graph']['distance_matrix'])
    graph.burnt_nodes = set(problem_data['graph']['burnt_nodes'])
    
    if problem_data['graph'].get('coordinates'):
        graph.xyz = np.array(problem_data['graph']['coordinates'])
    
    # Extract parameters
    params = problem_data['parameters']
    n = params['n']
    lambda_d = params.get('lambda_d', 2)
    burnt_nodes = params['burnt_nodes']
    instance = params.get('instance', 0)
    dim = params.get('dimension', 3)
    p = params.get('edge_probability', 0.25)
    D = params.get('D', 3)
    B = params.get('B', 3)
    firefighter_start = params.get('firefighter_start', None)
    if firefighter_start is None:
        # Check for firefighter_stations in graph section
        firefighter_stations = problem_data['graph'].get('firefighter_stations', None)
        if firefighter_stations is not None:
            firefighter_start = firefighter_stations[0] if firefighter_stations else 0
        else:
            firefighter_start = 0
    
    print(f"✅ Problem loaded successfully!")
    print(f"   • Vertices: {n}")
    print(f"   • Initial fires: {list(graph.burnt_nodes)}")
    print(f"   • Lambda: {lambda_d}")
    print(f"   • Defense rounds (D): {D}")
    print(f"   • Burning rounds (B): {B}")
    print(f"   • Edges: {int(np.sum(graph.A)/2)}")
    print(f"   • Firefighter start: {firefighter_start}")
    
    # Load solution if provided
    solution_data = None
    if solution_file and os.path.exists(solution_file):
        print(f"📂 Loading solution from: {solution_file}")
        with open(solution_file, 'r') as f:
            solution_data = json.load(f)
        print(f"✅ Solution loaded successfully!")
        print(f"   • Feasible: {solution_data['solution']['feasible']}")
        print(f"   • Objective: {solution_data['solution']['objective']}")
        print(f"   • Runtime: {solution_data['solution']['runtime']:.2f}s")
        if solution_data['solution']['defense_sequence']:
            print(f"   • Defense actions: {len(solution_data['solution']['defense_sequence'])}")
    
    return problem_data, solution_data, graph, params

def create_timeline_visualization(graph, params, solution_data, interactive_pos, problem_data):
    """
    Create interactive timeline visualization from solution data.
    
    Parameters:
    - graph: Graph object with A, D, and burnt_nodes attributes
    - params: Dictionary of problem parameters
    - solution_data: Dictionary containing solution results
    - interactive_pos: Dictionary of vertex positions for visualization
    - problem_data: Full problem data for accessing graph section
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available - install with: pip install plotly")
        return
    
    if not solution_data or not solution_data['solution']['defense_sequence']:
        print("❌ No defense sequence available for visualization")
        return
    
    n = params['n']
    defense_sequence = solution_data['solution']['defense_sequence']
    firefighter_start = params.get('firefighter_start', None)
    if firefighter_start is None:
        # Check for firefighter_stations in graph section
        firefighter_stations = problem_data['graph'].get('firefighter_stations', None)
        if firefighter_stations is not None:
            firefighter_start = firefighter_stations[0] if firefighter_stations else 0
        else:
            firefighter_start = 0
    
    print(f"\n🎬 Creating interactive timeline visualization...")
    print(f"   📊 Defense sequence: {len(defense_sequence)} actions")
    print(f"   🚒 Firefighter starts at: {firefighter_start}")
    print(f"   🎮 Turn-based simulation (MFP rules)")
    
    # Parse defense sequence into round-based format
    defense_by_round = {}
    for vertex, round_num, action_type in defense_sequence:
        if vertex < n:  # Valid vertex
            defense_by_round[round_num] = vertex
    
    print(f"   🔍 Defense by round: {defense_by_round}")
    
    # Initialize game state
    current_burning = set(graph.burnt_nodes)
    current_defended = set()
    firefighter_pos = firefighter_start
    
    # Firefighter's starting position is automatically defended
    if firefighter_pos < n:
        current_defended.add(firefighter_pos)
    
    B = params.get('B', 3)
    D = params.get('D', 5)
    max_rounds = max(B, max(defense_by_round.keys()) if defense_by_round else 0)
    
    print(f"   🔥 Fire spreads for {B} rounds")
    print(f"   🚒 Defense actions for {D} rounds")
    print(f"   🎯 Total simulation: {max_rounds} rounds")
    
    # Create timeline states for turn-based simulation
    timeline_states = []
    
    # Round 0: Initial state
    timeline_states.append({
        'step': 0,
        'round': 0,
        'phase': 'initial',
        'burning': current_burning.copy(),
        'defended': current_defended.copy(),
        'newly_burned': current_burning.copy(),  # Initially burning nodes
        'newly_defended': current_defended.copy(),  # Starting defended position
        'firefighter_pos': firefighter_pos,
        'action': f'🎮 Round 0: Game starts! Fire at {sorted(list(current_burning))}, firefighter at V{firefighter_pos}'
    })
    
    # Simulate each round with proper alternating sequence: Defense → Fire Spread → Defense → Fire Spread...
    for round_num in range(1, max_rounds + 1):
        
        # Phase 1: Firefighter Defense Action (first in each round)
        if round_num in defense_by_round:
            target_vertex = defense_by_round[round_num]
            prev_pos = firefighter_pos
            firefighter_pos = target_vertex
            
            # Determine newly defended vertices
            newly_defended = set()
            if target_vertex not in current_defended:
                newly_defended.add(target_vertex)
                current_defended.add(target_vertex)
            
            if target_vertex != prev_pos:
                action_text = f'🚒 Round {round_num} - Defense: Firefighter moves {prev_pos}→{target_vertex}, defends V{target_vertex}'
            else:
                action_text = f'🚒 Round {round_num} - Defense: Firefighter reinforces defense at V{target_vertex}'
            
            timeline_states.append({
                'step': len(timeline_states),
                'round': round_num,
                'phase': 'defense',
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': set(),
                'newly_defended': newly_defended,
                'firefighter_pos': firefighter_pos,
                'action': action_text
            })
        
        # Phase 2: Fire Spread (second in each round, if within burning rounds)
        if round_num <= B:
            old_burning = set(current_burning)
            new_burning = set(current_burning)
            
            # Fire spreads to all undefended adjacent vertices
            for burning_node in list(old_burning):  # Use list to avoid modification during iteration
                for neighbor in range(n):
                    if (graph.A[burning_node][neighbor] == 1 and 
                        neighbor not in current_defended and 
                        neighbor not in old_burning):
                        new_burning.add(neighbor)
            
            newly_burned = new_burning - current_burning
            current_burning = new_burning
            
            timeline_states.append({
                'step': len(timeline_states),
                'round': round_num,
                'phase': 'fire_spread',
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': newly_burned,
                'newly_defended': set(),
                'firefighter_pos': firefighter_pos,
                'action': f'🔥 Round {round_num} - Fire Spread: {len(newly_burned)} new vertices catch fire {sorted(list(newly_burned)) if newly_burned else "(none)"}'
            })
        
        elif round_num > B and round_num in defense_by_round:
            # Fire spreading is complete, but defense may continue
            timeline_states.append({
                'step': len(timeline_states),
                'round': round_num,
                'phase': 'post_fire',
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': set(),
                'newly_defended': set(),
                'firefighter_pos': firefighter_pos,
                'action': f'🏁 Round {round_num}: Fire spreading complete, defense continues'
            })
    
    print(f"   ✅ Created {len(timeline_states)} turn-based steps")
    print(f"   📊 Structure: {max_rounds} rounds of MFP simulation")
    print(f"   🎯 Final Result: {len(current_burning)} burned, {len(current_defended)} defended, {n - len(current_burning)} saved")
    
    # Debug: Print final state
    print(f"   🔥 Final burning vertices: {sorted(list(current_burning))}")
    print(f"   🛡️ Final defended vertices: {sorted(list(current_defended))}")
    saved_vertices = [i for i in range(n) if i not in current_burning]
    print(f"   📍 Safe vertices: {sorted(saved_vertices)}")
    
    # Create animation frames
    frames = []
    
    # Prepare edge data
    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i+1, n):
            if graph.A[i][j] == 1:
                edge_x.extend([interactive_pos[i][0], interactive_pos[j][0], None])
                edge_y.extend([interactive_pos[i][1], interactive_pos[j][1], None])
    
    # Prepare firefighter path
    path_edge_x, path_edge_y = [], []
    for i in range(len(defense_sequence) - 1):
        current_pos = defense_sequence[i][0]
        next_pos = defense_sequence[i+1][0]
        if current_pos < n and next_pos < n:
            path_edge_x.extend([interactive_pos[current_pos][0], interactive_pos[next_pos][0], None])
            path_edge_y.extend([interactive_pos[current_pos][1], interactive_pos[next_pos][1], None])
    
    for state in timeline_states:
        node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
        
        for vertex in range(n):
            if vertex in interactive_pos:
                node_x.append(interactive_pos[vertex][0])
                node_y.append(interactive_pos[vertex][1])
                
                # Determine node state and coloring
                if state['firefighter_pos'] is not None and vertex == state['firefighter_pos']:
                    node_colors.append('blue')
                    node_sizes.append(30)
                    if vertex in state['defended']:
                        node_text.append(f'🚒🛡️ Firefighter V{vertex} (Defended)')
                    else:
                        node_text.append(f'🚒 Firefighter V{vertex}')
                
                elif vertex in state.get('newly_burned', set()):
                    node_colors.append('orangered')
                    node_sizes.append(25)
                    node_text.append(f'💥 NEW FIRE V{vertex}')
                
                elif vertex in state['burning']:
                    if vertex in graph.burnt_nodes:
                        node_colors.append('darkred')
                        node_text.append(f'🔥 Initial Fire V{vertex}')
                    else:
                        node_colors.append('red')
                        node_text.append(f'🔥 Burned V{vertex}')
                    node_sizes.append(22)
                
                elif vertex in state.get('newly_defended', set()):
                    node_colors.append('limegreen')
                    node_sizes.append(24)
                    node_text.append(f'🛡️ NEW DEFENSE V{vertex}')
                
                elif vertex in state['defended']:
                    node_colors.append('green')
                    node_sizes.append(22)
                    node_text.append(f'🛡️ Defended V{vertex}')
                
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(18)
                    node_text.append(f'📍 Safe V{vertex}')
        
        # Create frame data
        frame_data = []
        
        # Add graph edges
        frame_data.append(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='gray'), 
            hoverinfo='none', showlegend=False, name='edges'
        ))
        
        # Add firefighter path
        frame_data.append(go.Scatter(
            x=path_edge_x, y=path_edge_y, mode='lines',
            line=dict(width=4, color='blue', dash='dash'), 
            hoverinfo='none', showlegend=False, name='path'
        ))
        
        # Add nodes
        frame_data.append(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, 
                       line=dict(width=2, color='black'),
                       symbol='circle'),
            text=[str(i) for i in range(len(node_x))], 
            textposition='middle center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hovertext=node_text, hoverinfo='text', 
            showlegend=False, name='nodes'
        ))
        
        # Add round step annotation
        round_info = f"Round {state.get('round', 0)}" if 'round' in state else f"Step {state['step']}"
        phase_info = f" - {state['phase'].title()}" if state.get('phase') else ""
        frame_data.append(go.Scatter(
            x=[0.05], y=[0.95], mode='text',
            text=[f"🎮 {round_info}{phase_info} | Step: {state['step']+1}/{len(timeline_states)}"],
            textposition='top left',
            textfont=dict(size=14, color='black', family='Arial Black'),
            showlegend=False
        ))
        
        frame = go.Frame(data=frame_data, name=str(state['step']))
        frames.append(frame)
    
    # Create figure
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="🎮 Interactive Turn-Based Firefighter Simulation",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x"),
        width=1000, height=800,
        plot_bgcolor='white',
        sliders=[{
                        "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}}],
                      "label": f"R{timeline_states[i].get('round', i)}: {timeline_states[i]['action'][:50]}...", "method": "animate"}
                     for i in range(len(timeline_states))],
            "active": 0, 
            "currentvalue": {"prefix": "🎮 Current Round: "},
            "pad": {"t": 50}
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}]
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    print("🎯 Use the slider to navigate through TIME-BASED gameplay")
    print("🖱️ Hover over nodes for detailed information about their state")
    print("📏 Node positions reflect actual travel times from distance matrix")
    print("🔗 Gray lines show graph connectivity, blue dashed line shows firefighter path")
    print("")
    print("🎮 TURN-BASED SIMULATION:")
    print("   📋 Sequence: Defense → Fire Spread → Defense → Fire Spread → ...")
    print("   🚒 Defense phase: Firefighter moves and defends (happens FIRST each round)")
    print("   🔥 Fire spread phase: Fire spreads to adjacent undefended vertices (happens SECOND)")
    print("   📊 Timeline shows discrete rounds, following classic MFP rules")
    print(f"   🎯 Total rounds simulated: {max_rounds}")
    print("   📈 Total steps: {}".format(len(timeline_states)))
    print("")
    print("🎮 CONTROLS:")
    print("   ▶️  Play button: Auto-play through all time steps")
    print("   ⏸️  Pause button: Stop auto-play")
    print("   📊 Slider: Manual navigation through time steps")
    print("")
    print("🎨 COLOR LEGEND:")
    print("   🚒 BLUE: Firefighter current position")
    print("   🚒🛡️ BLUE (with shield): Firefighter position + defended")
    print("   💥 ORANGE-RED: Newly burning vertices (this time step)")
    print("   🔥 DARK RED: Initially burning vertices")
    print("   🔥 RED: Previously burned vertices")
    print("   🛡️ LIME GREEN: Newly defended vertices (this time step)")
    print("   🛡️ GREEN: Previously defended vertices")
    print("   📍 GRAY: Safe vertices")
    
    fig.show()

def main():
    """Main function to load JSON files and create visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot Moving Firefighter Problem from JSON files')
    parser.add_argument('problem_file', help='Path to the problem JSON file')
    parser.add_argument('--solution-file', help='Path to the solution JSON file (optional)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear positioning cache')
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_positioning_cache()
    
    # Load problem and solution
    problem_data, solution_data, graph, params = load_problem_and_solution(args.problem_file, args.solution_file)
    
    # Get graph positioning
    interactive_pos, use_cached = get_graph_positioning_cache(graph, params['n'])
    
    # Create visualization
    create_timeline_visualization(graph, params, solution_data, interactive_pos, problem_data)
    
    print("\n🎯 JSON Plotting Complete!")
    print("=" * 50)
    print("✅ Problem and solution loaded from JSON")
    print("✅ Interactive timeline visualization created")
    print("=" * 50)

if __name__ == "__main__":
    main() 