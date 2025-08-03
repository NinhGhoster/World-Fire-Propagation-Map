import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import COBYLA, SciPyOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer, GoemansWilliamsonOptimizer
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals

# -- Seed for reproducibility --
seed = 4332 # random.randint(0, 9999)
print(seed)
random.seed(seed)
np.random.seed(seed)
rng  = np.random.default_rng(seed)
rng = np.random.RandomState(seed)
algorithm_globals.random_seed = seed
# -- User parameters --
reps          = 1       # QAOA depth
max_iters     = 100      # Optimizer max iterations
n_nodes       = 10       # Number of graph nodes
p_edge        = 0.5      # Random graph edge probability
initial_gamma = 0.1      # Initial gamma
initial_beta  = 0.0      # Initial beta
epsilon       = 0.4      # Relaxation strength

# -- 1. Build problem and classical seed --
G = nx.erdos_renyi_graph(n_nodes, p_edge, seed=seed)
w = nx.to_numpy_array(G)
qp = Maxcut(w).to_quadratic_program()
qubit_op, offset = qp.to_ising()

# Helper: compute cut weight
def compute_cut_weight(bitstr, w):
    cut = 0
    for i in range(len(bitstr)):
        for j in range(i+1, len(bitstr)):
            if bitstr[i] != bitstr[j]:
                cut += w[i, j]
    return cut

# Classical Goemans–Williamson seed (uses its own `seed` arg)
gw_res = GoemansWilliamsonOptimizer(num_cuts=1, seed=seed).solve(qp)
x_gw = np.array(gw_res.x)
# print("Classical before:", compute_cut_weight(x_gw, w))

# Flip last bit to see “after”
# x_gw[-1] = 1 - x_gw[-1]
print("Classical Cut: ", compute_cut_weight(x_gw, w))

# -- 2. Correct relaxation: p_i = 1-ε if z_i=1 else ε --
p_relaxed = np.where(x_gw == 1,
                     1 - epsilon,
                     epsilon)
# print("Relaxed probs p_i:", p_relaxed)

# Map to Ry rotation angles
thetas = [2 * np.arcsin(np.sqrt(p)) for p in p_relaxed]

# -- 3. Build warm-start initial state circuit --
init_state = QuantumCircuit(n_nodes)
for idx, th in enumerate(reversed(thetas)):
    init_state.ry(th, idx)

# -- 4. Build manual XY-mixer operator --
labels, coeffs = [], []
for i, j in G.edges():
    for pa in ("X", "Y"):
        p = ["I"] * n_nodes
        p[i] = p[j] = pa
        labels.append("".join(reversed(p)))
        coeffs.append(0.5)
xy_mixer = SparsePauliOp(labels, coeffs)

# -- 5. Sampling & metrics setup --
sampler  = Sampler(options={'seed':seed})
variants = ["QAOA", "WS-QAOA", "WSXY-QAOA"]
metrics  = {v: {"loss": [], "cut": [], "iter_to_final": None}
            for v in variants}

def get_cut(qc):
    res  = sampler.run(qc, shots=10000,run_options={'seed':seed}).result()
    dist = res.quasi_dists[0]
    best = max(dist, key=dist.get)
    bstr = format(best, f'0{n_nodes}b')
    return compute_cut_weight([int(b) for b in bstr], w)

# -- 6. Early-stop callback machinery --
class EarlyStop(Exception):
    pass

def make_callback(name, qaoa, threshold, tol=1e-8):
    def cb(eval_count, params, mean, std):
        metrics[name]["loss"].append(mean)
        qc  = qaoa.ansatz.bind_parameters(params)
        cut = get_cut(qc)
        metrics[name]["cut"].append(cut)
        # Stop as soon as we hit our threshold
        if cut >= threshold - tol:
            metrics[name]["iter_to_final"] = len(metrics[name]["cut"])
            raise EarlyStop()
    return cb

# -- 7. Choose optimizer: COBYLA or SciPyOptimizer --
use_scipy = True
optimizer = (
    SciPyOptimizer("L-BFGS-B", options={"maxiter": max_iters})
    if use_scipy
    else COBYLA(maxiter=max_iters)
)

init_pt = [initial_gamma] * reps + [initial_beta] * reps

# -- 8. Run QAOA for each variant with early stopping --
results = {}
for name, qargs in [
    ("QAOA",   {"sampler": sampler, "optimizer": optimizer, "reps": reps}),
    ("WS-QAOA",    {"sampler": sampler,
                    "optimizer": optimizer,
                    "reps": reps,
                    "initial_state": init_state,
                    "initial_point": init_pt}),
    ("WSXY-QAOA", {"sampler": sampler,
                    "optimizer": optimizer,
                    "reps": reps,
                    "initial_state": init_state,
                    "mixer": xy_mixer,
                    "initial_point": init_pt})
]:
    qaoa = QAOA(**qargs)
    # Phase 1: get the true final fval
    full_res = MinimumEigenOptimizer(qaoa).solve(qp)
    threshold = full_res.fval

    # Phase 2: re-run with early-stopping callback
    qaoa.callback = make_callback(name, qaoa, threshold)
    try:
        res = MinimumEigenOptimizer(qaoa).solve(qp)
    except EarlyStop:
        # interrupted early; use full result for fval
        res = full_res
    results[name] = res

# Ensure iter_to_final is set if never triggered early-stop
for name in variants:
    if metrics[name]["iter_to_final"] is None:
        metrics[name]["iter_to_final"] = len(metrics[name]["cut"])

# -- 9. Plotting results --
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
final_vals = [results[n].fval for n in variants]
for i, name in enumerate(variants):
  print(f"{name} Cut: ", final_vals[i])

plt.bar(["Classical"] + variants, [compute_cut_weight(x_gw, w)] + final_vals, color=['gray', 'C0', 'C1', 'C2'])
plt.title("Final Objective Value (Maximize)")
plt.ylabel("Objective Value")

# c) Iterations to reach final cut
plt.subplot(1, 2, 2)
iters_needed = [metrics[n]["iter_to_final"] for n in variants]
plt.bar(variants, iters_needed, color=['C0', 'C1', 'C2'])
plt.title("Time To Solution (TTS)")
plt.ylabel("Iterations")

plt.tight_layout()
plt.show()
