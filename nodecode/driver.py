# driver_sor.py
# Purpose: Run steady-state axisymmetric SOR on your mesh with separate relaxation factors.
# Notes:
# - You MUST fill in how to instantiate your Mesh at the TODO below.
# - This driver assumes your Node exposes radius as either .r (preferred) or ._r.
# - It writes iteration history and final fields to CSV in the working directory.

from typing import List, Dict, Any
import csv
import math

# --- Your project modules ---
from mesh import Mesh  # noqa: you will fill in the constructor below
from node import Node  # only to type-hint or inspect attributes; not required strictly
from SORSolver import SORSolver  # the solver class we sketched earlier

# -----------------------------
# Helpers (robust to your Node API)
# -----------------------------
def node_radius(nd: Node) -> float:
    """Return the radius field from a Node, supporting both .r and ._r."""
    if hasattr(nd, "r"):
        return float(getattr(nd, "r"))
    return float(getattr(nd, "_r"))

def uniform_dr(r_sorted: List[float], tol: float = 1e-12) -> float:
    """Verify (near) uniform spacing and return Δr."""
    if len(r_sorted) < 2:
        raise ValueError("Need at least two nodes to define Δr.")
    diffs = [r_sorted[i+1] - r_sorted[i] for i in range(len(r_sorted)-1)]
    dr = sum(diffs) / len(diffs)
    for d in diffs:
        if abs(d - dr) > tol:
            raise ValueError(f"Mesh is not uniform: saw Δr={d}, avg Δr={dr}.")
    return dr

def sort_nodes_inner_to_outer(nodes: List[Node]) -> List[Node]:
    return sorted(nodes, key=lambda n: node_radius(n))

def save_history_csv(history: List[Dict[str, Any]], path: str) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

def save_fields_csv(nodes: List[Node], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "r", "u_r", "u_theta", "p"])
        nodes_sorted = sort_nodes_inner_to_outer(nodes)
        for i, nd in enumerate(nodes_sorted):
            r = node_radius(nd)
            u_r = getattr(nd, "_u_r", None)
            u_t = getattr(nd, "_u_theta", None)
            p   = getattr(nd, "_p", None)
            w.writerow([i, r, u_r, u_t, p])

# -----------------------------
# Initial guesses (replace if you prefer to load from file)
# -----------------------------
def build_initial_fields(nodes: List[Node]) -> Dict[str, List[float]]:
    N = len(nodes)
    # Example initial fields: zeros everywhere
    # You said you will pass initial guesses; if so, replace this function
    # to load from your source and return arrays aligned inner->outer.
    return {
        "u_r":    [0.0 for _ in range(N)],
        "u_theta":[0.0 for _ in range(N)],
        "p":      [0.0 for _ in range(N)],
    }

# -----------------------------
# Main runner
# -----------------------------
def main() -> None:
    # ---- TODO: Instantiate YOUR mesh here ----
    # Adjust this line to match your Mesh signature (e.g., Mesh(r_inner, r_outer, N), or Mesh().build(...))
    # Example (PSEUDO): mesh = Mesh(r_inner=0.02, r_outer=0.10, n_nodes=121)
    # If your Mesh needs a method call to build, call it here before continuing.
    mesh = Mesh()  # <-- REPLACE with your actual constructor / builder
    # ------------------------------------------

    if not hasattr(mesh, "nodes") or not mesh.nodes:
        raise RuntimeError("Mesh has no nodes. Make sure you built it before running the solver.")

    # Sort nodes inner->outer and get Δr (verified uniform)
    mesh.nodes = sort_nodes_inner_to_outer(mesh.nodes)
    radii = [node_radius(nd) for nd in mesh.nodes]
    dr = uniform_dr(radii)

    # Build or load initial guesses (aligned inner->outer)
    init_fields = build_initial_fields(mesh.nodes)

    # Physical properties (example values – set yours)
    rho = 1000.0    # kg/m^3
    nu  = 1.0e-6    # m^2/s  (kinematic viscosity)

    # SOR relaxation factors (set what you want; GS ⇔ ω=1)
    omega_r = 1.5
    omega_t = 1.5
    omega_p = 1.7

    # Tolerance and iteration cap
    tol = 1e-8
    max_iter = 20000

    # Boundary conditions
    # Outer: Dirichlet u_r, u_theta at outermost node
    outer_bc = {"u_r": 0.0, "u_theta": 1.0}  # <--- set your physical BCs here
    # Inner: Dirichlet p at innermost node
    inner_bc = {"p": 0.0}                    # <--- set your physical BC here

    # Create and run the solver
    solver = SORSolver(
        mesh=mesh,
        rho=rho,
        nu=nu,
        dr=dr,
        omega_r=omega_r,
        omega_t=omega_t,
        omega_p=omega_p,
        tol=tol,
        max_iter=max_iter
    )

    result = solver.solve(
        init_fields=init_fields,
        outer_bc=outer_bc,
        inner_bc=inner_bc
    )

    # Report
    iters = result["iterations"]
    final_norm = result["final_norm"]
    print(f"[SOR] Converged in {iters} iterations with max-norm = {final_norm:.3e}")

    # Persist outputs
    save_history_csv(result["history"], "sor_history.csv")
    save_fields_csv(mesh.nodes, "sor_final_fields.csv")
    print("Saved sor_history.csv and sor_final_fields.csv")

if __name__ == "__main__":
    main()
