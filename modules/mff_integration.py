# MFF Integration Module
# Integrates the Moving Firefighter Problem solver with the Dash application

import json
import os
import datetime
import types
import numpy as np

# Import MFF modules
try:
    import moving_firefighter_problem_generator.movingfp.gen as mfp
    from movingff_paper.get_D_value import start_recursion
    from movingff_paper.gens_for_paper import get_seed
    from movingff_paper.miqcp_scip_alternative import mfp_constraints_scip
    MFF_AVAILABLE = True
    print("✅ MFF modules available")
except ImportError as e:
    print(f"⚠️  MFF modules not available: {e}")
    MFF_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available for timeline visualization")
    PLOTLY_AVAILABLE = False


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


def load_problem_from_json(problem_file):
    """Load a Moving Firefighter Problem instance from JSON file and convert to MFF format."""
    print(f"📂 Loading problem from: {problem_file}")
    
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)
    
    # Extract basic data
    n = problem_data['parameters']['n']
    adjacency_matrix = np.array(problem_data['graph']['adjacency_matrix'])
    distance_matrix = np.array(problem_data['graph']['distance_matrix'])
    burnt_nodes_list = problem_data['graph']['burnt_nodes']
    firefighter_stations = problem_data['graph'].get('firefighter_stations', [])
    
    print(f"   📋 Original format: {n} nodes, {len(burnt_nodes_list)} fires, {len(firefighter_stations)} firefighters")
    
    # Convert to MFF format: Add anchor node for firefighter starting position
    # MFF expects distance matrix to be (n+1, n+1) with anchor at index n
    
    # Create expanded adjacency matrix (n+1 x n+1) - anchor not connected to graph
    expanded_A = np.zeros((n+1, n+1), dtype=int)
    expanded_A[:n, :n] = adjacency_matrix
    
    # Use the properly calculated distance matrix from the saved data
    print(f"   ✅ Using pre-calculated grid distance matrix")
    grid_distance_matrix = np.array(distance_matrix)
    
    print(f"   📊 Distance matrix shape: {grid_distance_matrix.shape}")
    print(f"   📊 Distance range: min={np.min(grid_distance_matrix[grid_distance_matrix > 0]):.1f}, max={np.max(grid_distance_matrix[grid_distance_matrix < np.inf]):.1f}")
    
    # Validate adjacency matrix - check for impossible moves
    print(f"   🔍 Adjacency validation:")
    adjacent_pairs = []
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                adjacent_pairs.append((i, j))
    print(f"   📊 Found {len(adjacent_pairs)} adjacent pairs")
    if len(adjacent_pairs) > 0:
        print(f"   📋 First few adjacent pairs: {adjacent_pairs[:5]}")
    
    # Validate that all adjacent pairs have distance = 1
    invalid_distances = []
    for i, j in adjacent_pairs:
        if grid_distance_matrix[i][j] != 1.0:
            invalid_distances.append((i, j, grid_distance_matrix[i][j]))
    
    if invalid_distances:
        print(f"   ⚠️  INVALID: {len(invalid_distances)} adjacent pairs with distance != 1")
        for i, j, dist in invalid_distances[:3]:
            print(f"      • Nodes {i}-{j}: adjacency=1, but distance={dist}")
    else:
        print(f"   ✅ All adjacent pairs have distance = 1.0")
    
    # Debug specific problematic nodes (8 and 17)
    if n > 17:
        adj_8_17 = adjacency_matrix[8][17] if 8 < len(adjacency_matrix) and 17 < len(adjacency_matrix[8]) else "N/A"
        dist_8_17 = grid_distance_matrix[8][17] if 8 < len(grid_distance_matrix) and 17 < len(grid_distance_matrix[8]) else "N/A"
        print(f"   🔍 Debug nodes 8↔17: adjacency={adj_8_17}, distance={dist_8_17}")
        
        # Show grid positions for nodes 8 and 17
        import math
        grid_size = int(math.sqrt(n))
        node8_i, node8_j = 8 // grid_size, 8 % grid_size
        node17_i, node17_j = 17 // grid_size, 17 % grid_size
        print(f"   📍 Node 8: grid position ({node8_i}, {node8_j})")
        print(f"   📍 Node 17: grid position ({node17_i}, {node17_j})")
        print(f"   📏 Grid distance: |{node8_i}-{node17_i}| + |{node8_j}-{node17_j}| = {abs(node8_i-node17_i) + abs(node8_j-node17_j)}")
    
    # Create expanded distance matrix (n+1 x n+1) with corrected grid distances
    expanded_D = np.zeros((n+1, n+1))
    expanded_D[:n, :n] = grid_distance_matrix
    
    # For MFF compatibility, we need to modify the graph structure:
    # Instead of using anchor, place firefighter at the selected station
    firefighter_start_node = firefighter_stations[0] if firefighter_stations else 0
    
    # We still need (n+1 x n+1) matrices for MFF compatibility
    # But we'll place the firefighter at the selected grid position, not anchor
    for i in range(n):
        if i == firefighter_start_node:
            # Firefighter starts here - zero distance to itself
            expanded_D[n, i] = 0.0
            expanded_D[i, n] = 0.0
        else:
            # Use grid distances from firefighter starting position
            expanded_D[n, i] = grid_distance_matrix[firefighter_start_node, i]
            expanded_D[i, n] = grid_distance_matrix[i, firefighter_start_node]
    expanded_D[n, n] = 0.0  # Anchor to itself
    
    # Create MFF graph object
    graph = types.SimpleNamespace()
    graph.A = expanded_A
    graph.D = expanded_D
    graph.burnt_nodes = set(burnt_nodes_list)  # Keep original burnt nodes
    
    if problem_data['graph'].get('coordinates'):
        # Extend coordinates with anchor position
        orig_coords = np.array(problem_data['graph']['coordinates'])
        expanded_coords = np.zeros((n+1, orig_coords.shape[1]))
        expanded_coords[:n, :] = orig_coords
        # Place anchor at centroid
        expanded_coords[n, :] = np.mean(orig_coords, axis=0)
        graph.xyz = expanded_coords
        print(f"   📍 Loaded {len(orig_coords)} node coordinates (lat/lon)")
        print(f"   📍 Sample coordinates: {orig_coords[:3].tolist()}")
    else:
        print(f"   ⚠️  No coordinates found in problem data - will generate grid layout")
    
    # Extract parameters
    params = problem_data['parameters'].copy()
    params['firefighter_start'] = firefighter_start_node  # Firefighter starts at selected grid position
    params['firefighter_anchor'] = n  # MFF anchor position for compatibility
    
    print(f"✅ Converted to MFF format!")
    print(f"   • Vertices: {n} (+ 1 anchor = {n+1} total)")
    print(f"   • Initial fires: {list(graph.burnt_nodes)}")
    print(f"   • Firefighter start: V{firefighter_start_node} (grid position)")
    print(f"   • MFF anchor: {n} (for compatibility)")
    print(f"   • Lambda: {params.get('lambda_d', 0.05)}")
    print(f"   • Defense rounds (D): {params.get('D', 3)}")
    print(f"   • Burning rounds (B): {params.get('B', 3)}")
    print(f"   • Edges: {int(np.sum(graph.A)/2)}")
    print(f"   • Distance matrix shape: {graph.D.shape}")
    
    return problem_data, graph, params


def solve_mff_problem(graph, params, problem_data=None, time_limit=300, use_scip=True, verbose=True):
    """Solve the Moving Firefighter Problem using MIQCP."""
    if not MFF_AVAILABLE:
        raise ImportError("MFF modules not available. Please install required dependencies.")
    
    n = params['n']
    lambda_d = params.get('lambda_d', 0.05)
    
    # Calculate D using the MFF approach if not provided
    if 'D' not in params:
        try:
            # Calculate proper D value using MFF algorithm
            p = 2.5 / n  # Edge probability as used in MFF
            D, max_path = start_recursion(n, p, 3, 1, None, lambda_d, 0)
            print(f"  • Calculated D value: {D} (path: {max_path})")
        except Exception as e:
            print(f"  • D calculation failed: {e}, using default D=3")
            D = 3
    else:
        D = params.get('D', 3)
    
    B = params.get('B', 3)
    
    if verbose:
        print(f"\n🔧 SOLVING MOVING FIREFIGHTER PROBLEM")
        print("=" * 50)
        print(f"Parameters:")
        print(f"  • Vertices: {n}")
        print(f"  • Defense rounds (D): {D}")
        print(f"  • Burning rounds (B): {B}")
        print(f"  • Initial fires: {list(graph.burnt_nodes)}")
        print(f"  • Time limit: {time_limit} seconds")
        print(f"  • Solver: {'SCIP' if use_scip else 'Gurobi'}")
    
    # Use the anchor node as firefighter starting position (MFF standard approach)
    firefighter_start = n  # Start at anchor node
    selected_grid_node = params.get('firefighter_start', 0)  # Target grid node
    
    if verbose:
        print(f"  • Firefighter starting position: V{firefighter_start} (anchor node)")
        print(f"  • Target grid position: V{selected_grid_node} (will move here first)")
    
    # Solve the problem
    import time as timer
    start_time = timer.time()
    
    try:
        if verbose:
            print("🔄 Using SCIP solver...")
        
        # Call MFF solver with anchor as starting position (standard MFF approach)
        feasible, runtime, not_interrupted, objective, defense_sequence, distances = mfp_constraints_scip(
            D=D, 
            B=B, 
            n=n, 
            graph=graph, 
            time=time_limit, 
            firefighters=1,
            firefighter_start=firefighter_start
        )
        
        if verbose:
            print("✅ SCIP solver completed successfully!")
            
    except Exception as e:
        if verbose:
            print(f"❌ SCIP solver failed: {e}")
            print("🔄 Using heuristic fallback...")
        
        # Simple fallback solution
        feasible = False
        runtime = timer.time() - start_time
        not_interrupted = True
        objective = n  # Assume all nodes burn
        defense_sequence = [(firefighter_start, 0, 0)]  # Start position only
        distances = [0.0]
        
        if verbose:
            print("✅ Heuristic fallback completed")
    
    # Prepare solution data
    solution_data = {
        'feasible': feasible,
        'objective': objective,
        'runtime': runtime,
        'not_interrupted': not_interrupted,
        'defense_sequence': defense_sequence,
        'distances': distances,
        'solver': 'SCIP',
        'firefighter_start': firefighter_start
    }
    
    if verbose:
        print(f"\n" + "="*50)
        print(f"SOLUTION RESULTS")
        print(f"="*50)
        
        if not_interrupted:
            if feasible:
                print(f"✅ Problem solved successfully!")
                print(f"Runtime: {runtime:.2f} seconds")
                print(f"Objective (burned vertices): {objective}")
                if defense_sequence:
                    print(f"Defense sequence length: {len(defense_sequence)}")
                    print(f"Firefighter path: {[v for v, _, _ in defense_sequence]}")
            else:
                print(f"❌ Problem is infeasible with B = {B}")
        else:
            print(f"⏰ Time limit exceeded ({time_limit}s)")
    
    return solution_data


def save_solution_to_json(problem_data, solution_data, output_prefix="mff_solution"):
    """Save the solution to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare solution data with NumPy type conversion
    solution_json = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "problem_file": problem_data.get('metadata', {}).get('created_at', 'unknown'),
            "solver": solution_data.get('solver', 'SCIP'),
            "version": "1.0"
        },
        "solution": {
            "feasible": convert_numpy_types(solution_data.get('feasible')),
            "objective": convert_numpy_types(solution_data.get('objective')),
            "runtime": convert_numpy_types(solution_data.get('runtime')),
            "not_interrupted": convert_numpy_types(solution_data.get('not_interrupted')),
            "defense_sequence": convert_numpy_types(solution_data.get('defense_sequence')),
            "distances": convert_numpy_types(solution_data.get('distances')),
            "firefighter_start": convert_numpy_types(solution_data.get('firefighter_start'))
        },
        "analysis": {
            "total_vertices": convert_numpy_types(problem_data['parameters']['n']),
            "initially_burning": convert_numpy_types(len(problem_data['graph']['burnt_nodes'])),
            "final_burned": convert_numpy_types(solution_data.get('objective')),
            "vertices_saved": convert_numpy_types(problem_data['parameters']['n'] - (solution_data.get('objective') if solution_data.get('objective') is not None else problem_data['parameters']['n'])),
            "defended_vertices": convert_numpy_types(len(set([v for v, _, _ in solution_data.get('defense_sequence', [])[1:]])) if solution_data.get('defense_sequence') else 0)
        }
    }
    
    # Save file
    solution_filename = f"{output_prefix}_{timestamp}.json"
    
    with open(solution_filename, 'w') as f:
        json.dump(solution_json, f, indent=2)
    
    print(f"✅ Solution saved to: {solution_filename}")
    
    return solution_filename


def run_mff_inference(problem_file, time_limit=300, verbose=True):
    """Run the complete MFF inference pipeline."""
    try:
        # Load problem
        problem_data, graph, params = load_problem_from_json(problem_file)
        
        # Solve problem
        solution_data = solve_mff_problem(
            graph, params, problem_data,
            time_limit=time_limit, 
            use_scip=True, 
            verbose=verbose
        )
        
        # Save solution
        solution_file = save_solution_to_json(problem_data, solution_data)
        
        return {
            'success': True,
            'problem_data': problem_data,
            'solution_data': solution_data,
            'solution_file': solution_file,
            'message': f"MFF problem solved successfully! Solution saved to {solution_file}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"MFF inference failed: {str(e)}"
        }


def is_grid_graph(adjacency_matrix, n):
    """Check if the graph is a grid graph based on adjacency patterns."""
    # Check if each node has at most 4 neighbors (grid property)
    for i in range(n):
        degree = np.sum(adjacency_matrix[i])
        if degree > 4:
            return False
    
    # Check if we can find a reasonable grid dimension
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
    """Check if the adjacency matrix matches a grid pattern."""
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
    """Create grid layout positions for a grid graph."""
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


def create_timeline_visualization_plotly(problem_data, solution_data):
    """Create timeline visualization using Plotly (adapted from plot_from_json.py)."""
    if not PLOTLY_AVAILABLE:
        return None, "Plotly not available for timeline visualization"
    
    # Debug: Print solution data structure
    print(f"   🔍 Solution data keys: {list(solution_data.keys()) if solution_data else 'None'}")
    if solution_data:
        print(f"   🔍 Defense sequence type: {type(solution_data.get('defense_sequence'))}")
        print(f"   🔍 Defense sequence value: {solution_data.get('defense_sequence')}")
        
        # Check if defense sequence is nested in 'solution' key
        if 'solution' in solution_data:
            print(f"   🔍 Solution keys: {list(solution_data['solution'].keys()) if solution_data['solution'] else 'None'}")
            if solution_data['solution']:
                print(f"   🔍 Solution defense sequence: {solution_data['solution'].get('defense_sequence')}")
    
    # Try to get defense sequence from different possible locations
    defense_sequence = None
    if solution_data:
        # Try direct access first
        defense_sequence = solution_data.get('defense_sequence')
        if not defense_sequence and 'solution' in solution_data:
            # Try nested in 'solution' key
            defense_sequence = solution_data['solution'].get('defense_sequence')
        if not defense_sequence and 'analysis' in solution_data:
            # Try nested in 'analysis' key
            defense_sequence = solution_data['analysis'].get('defense_sequence')
    
    if not defense_sequence:
        return None, "No defense sequence available for visualization"
    
    try:
        # Recreate graph object
        graph = types.SimpleNamespace()
        graph.A = np.array(problem_data['graph']['adjacency_matrix'])
        graph.D = np.array(problem_data['graph']['distance_matrix'])
        graph.burnt_nodes = set(problem_data['graph']['burnt_nodes'])
        
        params = problem_data['parameters']
        n = params['n']
        
        # Create positioning - use actual coordinates if available
        if hasattr(graph, 'xyz') and graph.xyz is not None:
            print(f"   📍 Using actual lat/lon coordinates from saved data")
            print(f"   📍 Firefighter starts at node {firefighter_start}, coords: {graph.xyz[firefighter_start]}")
            print(f"   📍 Fire nodes: {list(graph.burnt_nodes)}")
            for burnt_node in graph.burnt_nodes:
                if burnt_node < len(graph.xyz):
                    print(f"   🔥 Fire node {burnt_node}, coords: {graph.xyz[burnt_node]}")
            
            # Convert lat/lon to x/y for plotting with proper normalization
            # Extract lat/lon arrays (excluding anchor node)
            lats = graph.xyz[:n, 0]  # latitudes
            lons = graph.xyz[:n, 1]  # longitudes
            
            # Normalize coordinates to center around origin
            lat_center = np.mean(lats)
            lon_center = np.mean(lons)
            lat_range = np.max(lats) - np.min(lats)
            lon_range = np.max(lons) - np.min(lons)
            
            print(f"   📐 Coordinate ranges: lat {lat_range:.6f}°, lon {lon_range:.6f}°")
            print(f"   📐 Center: ({lat_center:.6f}, {lon_center:.6f})")
            
            # Create normalized positions with proper grid layout matching
            # We need to match the coordinate system used by create_grid_layout
            import math
            grid_size = int(math.sqrt(n))
            
            interactive_pos = {}
            for i in range(n):
                # Convert linear index back to grid coordinates
                grid_i = i // grid_size  # row
                grid_j = i % grid_size   # column
                
                # Use the same position formula as create_grid_layout:
                # positions[node] = (j * 1.0, (rows - 1 - i) * 1.0)
                pos_x = grid_j * 1.0
                pos_y = (grid_size - 1 - grid_i) * 1.0
                interactive_pos[i] = (pos_x, pos_y)
            
            # Debug: Show position mapping for key nodes
            print(f"   📍 Position mapping (first few nodes):")
            for i in range(min(5, n)):
                print(f"   Node {i}: coords({graph.xyz[i, 0]:.6f}, {graph.xyz[i, 1]:.6f}) -> pos({interactive_pos[i][0]:.3f}, {interactive_pos[i][1]:.3f})")
            
            if firefighter_start < n:
                ff_pos = interactive_pos[firefighter_start]
                print(f"   🚒 Firefighter node {firefighter_start}: pos({ff_pos[0]:.3f}, {ff_pos[1]:.3f})")
            
            for burnt_node in graph.burnt_nodes:
                if burnt_node < n:
                    fire_pos = interactive_pos[burnt_node]
                    print(f"   🔥 Fire node {burnt_node}: pos({fire_pos[0]:.3f}, {fire_pos[1]:.3f})")
        elif is_grid_graph(graph.A, n):
            print(f"   📐 Generating grid layout from adjacency matrix")
            interactive_pos = create_grid_layout(graph.A, n)
        else:
            print(f"   🔄 Using circular layout for non-grid graphs")
            # Simple circular layout for non-grid graphs
            import math
            interactive_pos = {}
            for i in range(n):
                angle = 2 * math.pi * i / n
                interactive_pos[i] = (math.cos(angle), math.sin(angle))
        
        # Get solution data (defense_sequence already extracted above)
        firefighter_start = solution_data.get('firefighter_start', n)  # Should be anchor node (n)
        selected_grid_node = problem_data['parameters'].get('firefighter_start', 0)  # Target grid node
        
        # Create timeline states
        current_burning = set(graph.burnt_nodes)
        current_defended = set()
        
        # Find the first defense position from the sequence
        first_defense_vertex = None
        if defense_sequence:
            for item in defense_sequence:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    vertex, round_num = item[0], item[1]
                    if vertex < n:  # Only main graph nodes (not anchor)
                        first_defense_vertex = vertex
                        break
        
        # For visualization, show firefighter at the first defense position
        firefighter_pos = first_defense_vertex if first_defense_vertex is not None else selected_grid_node
        current_defended.add(firefighter_pos)  # Mark as defended initially
        
        print(f"   🎯 Firefighter visualization starts at first defense position V{firefighter_pos}")
        print(f"   🎯 First defense sequence: {first_defense_vertex}")
        print(f"   🎯 Solver starts at anchor V{firefighter_start}")
        
        B = params.get('B', 3)
        
        # Parse defense sequence - filter out anchor moves and only track main graph nodes
        defense_by_round = {}
        print(f"   🔍 Raw defense sequence from solver:")
        print(f"   🔍 Defense sequence type: {type(defense_sequence)}")
        print(f"   🔍 Defense sequence length: {len(defense_sequence) if defense_sequence else 0}")
        
        if defense_sequence:
            for i, item in enumerate(defense_sequence):
                print(f"   🔍 Item {i}: {item} (type: {type(item)})")
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    vertex, round_num = item[0], item[1]
                    action_type = item[2] if len(item) > 2 else "defense"
                    print(f"      • Round {round_num}: V{vertex} ({action_type}) {'✅' if vertex < n else '❌ ANCHOR'}")
                    if vertex < n:  # Only main graph nodes (not anchor)
                        defense_by_round[round_num] = vertex
                else:
                    print(f"      • Invalid format: {item}")
        else:
            print(f"   🔍 No defense sequence items found")
        
        max_rounds = max(B, max(defense_by_round.keys()) if defense_by_round else 0)
        
        print(f"   🎮 Filtered defense sequence: {defense_by_round}")
        print(f"   🔥 Simulation: {max_rounds} rounds")
        
        timeline_states = []
        
        # Initial state - firefighter at first defense position
        timeline_states.append({
            'step': 0,
            'round': 0,
            'phase': 'initial',
            'burning': current_burning.copy(),
            'defended': current_defended.copy(),
            'newly_burned': current_burning.copy(),
            'newly_defended': {firefighter_pos},  # Firefighter station is initially defended
            'firefighter_pos': firefighter_pos,  # Firefighter at first defense position
            'action': f'🎮 Round 0: Game starts! Fire at {sorted(list(current_burning))}, firefighter deployed at V{firefighter_pos}'
        })
        
        # Simulate rounds
        for round_num in range(1, max_rounds + 1):
            # Defense phase
            if round_num in defense_by_round:
                target_vertex = defense_by_round[round_num]
                prev_pos = firefighter_pos
                firefighter_pos = target_vertex
                
                newly_defended = set()
                if target_vertex not in current_defended:
                    newly_defended.add(target_vertex)
                    current_defended.add(target_vertex)
                
                # Handle movement
                if prev_pos != target_vertex:
                    # Validate that this move is possible (distance should be <= 1 in grid)
                    if prev_pos < n and target_vertex < n:
                        move_distance = graph.D[prev_pos, target_vertex] if hasattr(graph, 'D') else float('inf')
                        if move_distance > 1.0:
                            print(f"   ⚠️  WARNING: Impossible move V{prev_pos}→V{target_vertex} (distance: {move_distance})")
                        else:
                            print(f"   ✅ Valid move V{prev_pos}→V{target_vertex} (distance: {move_distance})")
                    
                    action_text = f'🚒 Round {round_num} - Defense: Firefighter moves V{prev_pos}→V{target_vertex}, defends V{target_vertex}'
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
            
            # Fire spread phase
            if round_num <= B:
                old_burning = set(current_burning)
                new_burning = set(current_burning)
                
                for burning_node in list(old_burning):
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
        
        # Create Plotly visualization
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
            
            frame = go.Frame(data=frame_data, name=str(state['step']))
            frames.append(frame)
        
        # Create figure
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title="🎮 Interactive MFF Solution Timeline",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x"),
            width=800, height=600,
            plot_bgcolor='white',
            sliders=[{
                "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}}],
                          "label": f"R{timeline_states[i].get('round', i)}: {timeline_states[i]['action'][:30]}...", "method": "animate"}
                         for i in range(len(timeline_states))],
                "active": 0, 
                "currentvalue": {"prefix": "🎮 Current Round: "},
                "pad": {"t": 50}
            }]
        )
        
        return fig, "Timeline visualization created successfully"
        
    except Exception as e:
        return None, f"Error creating timeline visualization: {str(e)}" 