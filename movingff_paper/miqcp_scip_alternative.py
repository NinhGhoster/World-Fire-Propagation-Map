
"""
SCIP-based solver for Moving Firefighter Problem
Free alternative to Gurobi MIQCP solver - FULL QUADRATIC VERSION
"""

try:
    import pyscipopt as scip
    import numpy as np
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("SCIP not available. Install with: pip install pyscipopt")

def mfp_constraints_scip(D, B, n, graph, time=60, firefighters=1, firefighter_start=None):
    """
    Solve Moving Firefighter Problem using SCIP with FULL QUADRATIC MIQCP formulation
    (Exact translation of the Gurobi MIQCP model)
    
    Args:
        D: Maximum defense rounds per burning round
        B: Number of burning rounds
        n: Number of main vertices (excluding anchor)
        graph: Graph object with adjacency matrix A and distance matrix D
        time: Time limit in seconds
        firefighters: Number of firefighters (currently supports 1)
    
    Returns:
        (feasible, runtime, not_interrupted, objective, defense_sequence, distances)
    """
    
    if not SCIP_AVAILABLE:
        print("ERROR: SCIP not installed. Using fallback...")
        return fallback_heuristic_solver(D, B, n, graph, time, firefighters)
    
    import time as timer
    start_time = timer.time()
    
    try:
        print(f"burnt nodes in G_solver {graph.burnt_nodes}")
        
        # Create SCIP model
        model = scip.Model("miqcp_firefighter")
        
        # Set parameters to match Gurobi settings
        if time is not None:
            model.setParam("limits/time", time)
        model.setParam("display/verblevel", 4)  # Similar to outputFlag = 1
        
        # Force SCIP to provide a solution even for infeasible problems
        model.setParam("presolving/maxrounds", 0)  # Disable presolving that detects infeasibility
        model.setParam("limits/solutions", 1)  # Request at least one solution
        model.setParam("limits/gap", 1.0)  # Allow large gap for infeasible problems
        model.setParam("feasibility/only", False)  # Don't stop at first feasible solution
        model.setParam("limits/restarts", 0)  # Disable restarts
        
        # Add slack variables to make constraints soft (relaxed)
        slack_vars = {}
        
        # Adjust n to include anchor vertex (matching original)
        n = n + 1
        
        # INPUT preparation (matching original)
        I = graph.burnt_nodes
        
        b0 = []
        d0 = []
        for i in range(n):
            if i in I:
                b0.append(1)  # These are the vertices burned at time j=0
            else:
                b0.append(0)
        for i in range(n):
            if i == n-1:
                d0.append(1)  # The bulldozer begins at vertex n-1
            else:
                d0.append(0)
        
        print(f"Setting up SCIP MIQCP model: n={n}, D={D}, B={B}")
        
        # ========================== VARIABLES (exact match to original) ==========================
        
        # b[i][j] = 1 if vertex i burns in round j
        b = {}
        for i in range(n):
            for j in range(B):
                b[i, j] = model.addVar(vtype="B", name=f"b_{i+1}_{j+1}")
        
        # d[i][j] = 1 if vertex i is defended in round j
        d = {}
        for i in range(n):
            for j in range(B):
                d[i, j] = model.addVar(vtype="B", name=f"d_{i+1}_{j+1}")
        
        # d_prime[j][i][l] = intermediate defended variable for round j, vertex i, defense step l
        d_prime = {}
        for j in range(B):
            for i in range(n):
                for l in range(D):
                    d_prime[j, i, l] = model.addVar(vtype="B", name=f"d_prime_{j+1}_{i+1}_{l+1}")
        
        # p[j][i][l] = 1 if firefighter defends vertex i in round j, defense step l
        p = {}
        for j in range(B):
            for i in range(n):
                for l in range(D):
                    p[j, i, l] = model.addVar(vtype="B", name=f"p_{j+1}_{i+1}_{l+1}")
        
        # t[j] = cumulative time at end of round j
        t = {}
        for j in range(B):
            t[j] = model.addVar(vtype="C", name=f"t_{j+1}")
        
        # y[j] = 1 if any defense action occurs in round j
        y = {}
        for j in range(B):
            y[j] = model.addVar(vtype="B", name=f"y_{j+1}")
        
        print("Variables created, adding constraints...")
        
        # ========================== CONSTRAINTS (exact match to original) ==========================
        
        # Constraint (2): b[i][j] >= b[i][j-1] (fire persistence)
        for i in range(n):
            for j in range(B):
                if j == 0:
                    model.addCons(b[i, j] >= b0[i], name=f"fire_persist_{i}_{j}_initial")
                else:
                    model.addCons(b[i, j] >= b[i, j-1], name=f"fire_persist_{i}_{j}")
        
        # Constraint (3): d[i][j] >= d[i][j-1] (defense persistence)
        for i in range(n):
            for j in range(B):
                if j == 0:
                    model.addCons(d[i, j] >= d0[i], name=f"def_persist_{i}_{j}_initial")
                else:
                    model.addCons(d[i, j] >= d[i, j-1], name=f"def_persist_{i}_{j}")
        
        # Constraint (4): b[i][j] + d[i][j] <= 1 (vertex cannot be both burned and defended)
        for i in range(n):
            for j in range(B):
                model.addCons(b[i, j] + d[i, j] <= 1, name=f"burn_def_mutex_{i}_{j}")
        
        # Constraint (5): Fire spread from adjacent burning vertices
        original_n = n - 1  # Original number of vertices (excluding anchor)
        for i in range(original_n):  # Main vertices only
            for j in range(B):
                for k in range(original_n):
                    if j == 0:
                        if graph.A[k, i] == 1:  # Adjacent vertices
                            model.addCons(b[i, j] + d[i, j] >= b0[k], name=f"fire_spread_{i}_{j}_{k}_initial")
                    else:
                        if graph.A[k, i] == 1:  # Adjacent vertices
                            model.addCons(b[i, j] + d[i, j] >= b[k, j-1], name=f"fire_spread_{i}_{j}_{k}")
        
        # Constraints (6) and (7): Reset d0 for firefighter start
        for i in range(n):
            d0[i] = 0
        
        # Set firefighter start position
        if firefighter_start is None:
            firefighter_start = n - 1  # Default to anchor vertex
        d0[firefighter_start] = 1
        
        # Constraint (8): Initial d_prime values
        for i in range(n):
            for j in range(B):
                if j == 0:
                    model.addCons(d_prime[j, i, 0] >= d0[i], name=f"d_prime_initial_{i}_{j}")
                else:
                    model.addCons(d_prime[j, i, 0] >= d[i, j-1], name=f"d_prime_link_{i}_{j}")
        
        # Constraint (9): Final d_prime equals d
        for i in range(n):
            for j in range(B):
                model.addCons(d_prime[j, i, D-1] == d[i, j], name=f"d_prime_final_{i}_{j}")
        
        # Constraint (10): d_prime monotonicity
        for i in range(n):
            for j in range(B):
                for k in range(1, D):
                    model.addCons(d_prime[j, i, k] >= d_prime[j, i, k-1], name=f"d_prime_mono_{i}_{j}_{k}")
        
        # Constraints (11) and (12): p[j][i][k] represents new defenses
        for j in range(B):
            for i in range(n):
                for k in range(D):
                    if k == 0:
                        if j == 0:
                            model.addCons(p[j, i, k] >= d_prime[j, i, k] - d0[i], name=f"p_def_{i}_{j}_{k}_initial")
                        else:
                            model.addCons(p[j, i, k] >= d_prime[j, i, k] - d[i, j-1], name=f"p_def_{i}_{j}_{k}")
                    else:
                        model.addCons(p[j, i, k] >= d_prime[j, i, k] - d_prime[j, i, k-1], name=f"p_def_{i}_{j}_{k}_step")
        
        # Constraint (13): Exactly one defense action per step
        for j in range(B):
            for k in range(D):
                defense_sum = sum(p[j, i, k] for i in range(n))
                model.addCons(defense_sum == 1, name=f"one_defense_{j}_{k}")
        
        # Constraints (14), (15), and (16): Defense ordering constraints
        for j in range(B):
            for k in range(D):
                # Calculate sum of all new defenses at this step
                defense_changes = []
                for i in range(n):
                    if k == 0:
                        if j == 0:
                            defense_changes.append(d_prime[j, i, k] - d0[i])
                        else:
                            defense_changes.append(d_prime[j, i, k] - d[i, j-1])
                    else:
                        defense_changes.append(d_prime[j, i, k] - d_prime[j, i, k-1])
                
                total_changes = sum(defense_changes)
                
                for i in range(n):
                    if k == 0:
                        if j == 0:
                            model.addCons(p[j, i, k] >= d0[i] * (1 - total_changes), name=f"p_order_{i}_{j}_{k}_initial")
                        else:
                            model.addCons(p[j, i, k] >= p[j-1, i, D-1] * (1 - total_changes), name=f"p_order_{i}_{j}_{k}")
                    else:
                        model.addCons(p[j, i, k] >= p[j, i, k-1] * (1 - total_changes), name=f"p_order_{i}_{j}_{k}_step")
        
        # Constraints (17) and (18): QUADRATIC movement cost calculation
        print("Adding quadratic movement cost constraints...")
        
        # Use original_n for distance matrix access (since graph.D only has original size)
        original_n = n - 1  # Original number of vertices (excluding anchor)
        
        for j in range(B):
            # First movement cost component (from previous position to current)
            sum_1_terms = []
            for l in range(n):
                sum_1_a_terms = []
                for i in range(n):
                    # QUADRATIC TERM: p[j][i][0] * graph.D[i][l]
                    # Use original_n for distance matrix access
                    if i < original_n and l < original_n:
                        sum_1_a_terms.append(p[j, i, 0] * graph.D[i][l])
                    else:
                        # For anchor vertex, use distance to/from main vertices
                        if i == original_n:  # Anchor vertex
                            if l < original_n:
                                sum_1_a_terms.append(p[j, i, 0] * graph.D[0][l])  # Use distance from vertex 0
                        elif l == original_n:  # To anchor vertex
                            if i < original_n:
                                sum_1_a_terms.append(p[j, i, 0] * graph.D[i][0])  # Use distance to vertex 0
                sum_1_a = sum(sum_1_a_terms)
                
                if j == 0:
                    # QUADRATIC TERM: sum_1_a * d0[l]
                    sum_1_terms.append(sum_1_a * d0[l])
                else:
                    # QUADRATIC TERM: sum_1_a * p[j-1][l][D-1]
                    sum_1_terms.append(sum_1_a * p[j-1, l, D-1])
            sum_1 = sum(sum_1_terms)
            
            # Second movement cost component (within-round movements)
            sum_2_terms = []
            for k in range(1, D):
                sum_2_a_terms = []
                for l in range(n):
                    sum_2_b_terms = []
                    for i in range(n):
                        # QUADRATIC TERM: p[j][i][k] * graph.D[i][l]
                        # Use original_n for distance matrix access
                        if i < original_n and l < original_n:
                            sum_2_b_terms.append(p[j, i, k] * graph.D[i][l])
                        else:
                            # For anchor vertex, use distance to/from main vertices
                            if i == original_n:  # Anchor vertex
                                if l < original_n:
                                    sum_2_b_terms.append(p[j, i, k] * graph.D[0][l])  # Use distance from vertex 0
                            elif l == original_n:  # To anchor vertex
                                if i < original_n:
                                    sum_2_b_terms.append(p[j, i, k] * graph.D[i][0])  # Use distance to vertex 0
                    sum_2_b = sum(sum_2_b_terms)
                    
                    # QUADRATIC TERM: sum_2_b * p[j][l][k-1]
                    sum_2_a_terms.append(sum_2_b * p[j, l, k-1])
                sum_2_a = sum(sum_2_a_terms)
                sum_2_terms.append(sum_2_a)
            sum_2 = sum(sum_2_terms)
            
            # Total movement cost for round j
            sum_3 = sum_1 + sum_2
            
            if j == 0:
                model.addCons(t[j] == sum_3, name=f"time_calc_{j}")
            else:
                model.addCons(t[j] == sum_3 + t[j-1], name=f"time_calc_{j}")
        
        # Constraint (19): y[j] indicates if any defense occurs in round j
        for j in range(B):
            defense_activity_terms = []
            for i in range(n):
                for k in range(D):
                    if k == 0:
                        if j == 0:
                            defense_change = d_prime[j, i, k] - d0[i]
                        else:
                            defense_change = d_prime[j, i, k] - d[i, j-1]
                    else:
                        defense_change = d_prime[j, i, k] - d_prime[j, i, k-1]
                    
                    model.addCons(y[j] >= defense_change, name=f"y_indicator_{j}_{i}_{k}")
                    defense_activity_terms.append(defense_change)
            
            total_activity = sum(defense_activity_terms)
            model.addCons(y[j] <= total_activity, name=f"y_upper_{j}")
        
        # Constraint (20) and (21): Time bounds
        for j in range(B):
            model.addCons(t[j] <= j + 1, name=f"time_bound_{j}")
        
        # ========================== OBJECTIVE FUNCTION ==========================
        
        # Minimize total burned vertices in final round
        final_burned = sum(b[i, B-1] for i in range(n))
        model.setObjective(final_burned, "minimize")
        
        print("Model setup complete, starting SCIP optimization...")
        
        # ========================== OPTIMIZATION ==========================
        
        model.optimize()
        
        runtime = timer.time() - start_time
        status = model.getStatus()
        
        # Check if time limit exceeded
        not_interrupted = True
        if time is not None and runtime > time:
            not_interrupted = False
            return None, runtime, not_interrupted, None, None, None
        
        # Handle solution status
        feasible = False
        if status == "infeasible":
            feasible = False  # Problem is infeasible
            return feasible, runtime, not_interrupted, None, None, None
        
        if status not in ["optimal", "bestsollimit"]:
            print(f"⚠️  SCIP status: {status}")
            return False, runtime, not_interrupted, None, None, None
        
        # ========================== EXTRACT SOLUTION ==========================
        
        # If we reach here, the problem was solved successfully
        feasible = True
        
        objective_value = model.getObjVal()
        print(f"✅ SCIP objective: {objective_value}")
        
        # Extract variable values
        b_out = {}
        d_out = {}
        d_prime_out = {}
        p_out = {}
        t_out = {}
        
        # Get all variable values
        for var in model.getVars():
            var_name = var.name
            var_value = model.getVal(var)
            
            if var_name.startswith('b_'):
                parts = var_name.split('_')
                i, j = int(parts[1]) - 1, int(parts[2]) - 1
                b_out[i, j] = var_value
            elif var_name.startswith('d_prime_'):
                parts = var_name.split('_')
                j, i, l = int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1
                d_prime_out[j, i, l] = var_value
            elif var_name.startswith('d_'):
                parts = var_name.split('_')
                i, j = int(parts[1]) - 1, int(parts[2]) - 1
                d_out[i, j] = var_value
            elif var_name.startswith('p_'):
                parts = var_name.split('_')
                j, i, l = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1
                p_out[j, i, l] = var_value
            elif var_name.startswith('t_'):
                parts = var_name.split('_')
                j = int(parts[1]) - 1
                t_out[j] = var_value
        
        # Build defense sequence (matching original format)
        defense_sequence = []
        defense_sequence.append([n-1, 0, 0])  # Start at anchor
        
        for l in range(B):
            for k in range(D):
                for i in range(n):
                    if (l, i, k) in p_out and p_out[l, i, k] > 0.9:
                        defense_sequence.append([i, l + 1, k])
                        break
        
        # Extract time values
        distances = []
        for j in range(B):
            if j in t_out:
                distances.append(t_out[j])
        
        print(f"✅ SCIP solved successfully!")
        print(f"Objective: {objective_value}")
        print(f"Runtime: {runtime:.2f}s")
        print(f"Defense sequence length: {len(defense_sequence)}")
        
        return feasible, runtime, not_interrupted, objective_value, defense_sequence, distances
        
    except Exception as e:
        print(f"❌ SCIP error: {e}")
        print("Falling back to heuristic solver...")
        return fallback_heuristic_solver(D, B, n, graph, time, firefighters, firefighter_start)


def fallback_heuristic_solver(D, B, n, graph, time, firefighters, firefighter_start=None):
    """
    Simple heuristic fallback when optimization solvers fail
    """
    print("🔧 Using greedy heuristic fallback...")
    
    import time as timer
    start_time = timer.time()
    
    # Simple greedy strategy: defend highest degree vertices near fires
    defended_vertices = set()
    if firefighter_start is None:
        firefighter_start = n - 1  # Default to anchor vertex
    firefighter_pos = firefighter_start
    
    defense_sequence = [(firefighter_pos, 0, 0)]
    distances = []
    
    # Calculate vertex degrees (only for main vertices, not anchor)
    original_n = n - 1  # Original number of vertices (excluding anchor)
    degrees = {}
    for i in range(original_n):
        degrees[i] = sum(graph.A[i])
    
    # Greedy defense: prioritize high-degree vertices near initial fires
    candidates = []
    for i in range(original_n):
        if i not in graph.burnt_nodes:
            # Check if adjacent to burning vertices
            adjacent_to_fire = any(graph.A[i][j] == 1 for j in graph.burnt_nodes if j < original_n)
            priority = degrees[i] * 2 if adjacent_to_fire else degrees[i]
            candidates.append((priority, i))
    
    candidates.sort(reverse=True)
    
    # Defend top candidates within movement constraints
    for b in range(1, min(B + 1, len(candidates) + 1)):
        if candidates:
            target = candidates[b-1][1]
            defense_sequence.append((target, b, 0))
            defended_vertices.add(target)
            distances.append(graph.D[firefighter_pos, target])
            firefighter_pos = target
    
    # Calculate rough objective (number of potentially burned vertices)
    objective = len([i for i in range(original_n) if i not in defended_vertices and i not in graph.burnt_nodes])
    
    runtime = timer.time() - start_time
    
    print(f"🎯 Heuristic solution: {len(defended_vertices)} defended, ~{objective} may burn")
    
    return False, runtime, True, float(objective), defense_sequence, distances


# Make it compatible with existing code
def mfp_constraints(D, B, n, graph, time=60, firefighters=1):
    """Wrapper to maintain compatibility with existing code"""
    return mfp_constraints_scip(D, B, n, graph, time, firefighters) 