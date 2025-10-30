# sor_driver.py
from __future__ import annotations
import sys
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np


from mesh import Mesh
from SORSolver import SORSolver

# ---------- Utilities ----------
def require_node_api(mesh) -> None:
    """Fail early with a clear message if Node API expected by SORSolver is missing."""
    needed_attrs = [
    # state mgmt
    "snapshot_prev", "local_deltas", "update_local_residuals",
    # coefficient assembly
    "assemble_coeffs_u_r", "assemble_coeffs_u_theta", "assemble_coeffs_p",
    # SOR updates
    "sor_update_u_r", "sor_update_u_theta", "sor_update_p",
    # boundary controls / neighbors
    "is_inner_bc", "is_outer_bc", "set_neighbors",
    # coefficient holders
    "coeffs_r", "coeffs_t", "coeffs_p",
    # primary unknowns
    "u_r", "u_theta", "p", "r",
    # BC storage used by solver’s _apply_bcs
    "_u_r", "_u_theta", "_p",
    ]
    if not getattr(mesh, "nodes", None):
        raise ValueError("Mesh has no nodes; buildMesh likely did not run.")
    nd = mesh.nodes[0]
    missing: List[str] = [attr for attr in needed_attrs if not hasattr(nd, attr)]
    if missing:
        items = "\n - ".join(missing)
        raise NotImplementedError(
            "Your current Node implementation is missing methods/fields required by SORSolver.\n"
            "Please add the following to node.py (or adapt SORSolver accordingly):\n"
            f" - {items}\n\n"
            "Tip: coeffs_* can be small containers (e.g., dataclass) holding (aP, aE, aW, b). "
            "assemble_coeffs_* should populate those based on your discretization; "
            "sor_update_* should perform the over-relaxed update using omega."
        )


# ---------- Case definition ----------

class SorCase:
    """
    Encapsulates one steady, axisymmetric solve with piecewise-uniform radial mesh.
    """
    def __init__(self,
                r_inner: float = 0.05,
                r_main: float = 0.10,
                r_outer: float = 0.125,
                n_inner: int = 1,
                n_main: int = 5,
                n_outer: int = 1,
                rho: float = 1000.0,
                nu: float = 1.0e-6,
                p0: float = 101325.0,
                u_r_out: float = 0.0,
                u_t_out: float = 1.0,
                omega_r: float = 1.3,
                omega_t: float = 1.3,
                omega_p: float = 1.7,
                tol: float = 1e-8,
                max_iter: int = 10_000):
        
        self.r_inner = float(r_inner)
        self.r_main = float(r_main)
        self.r_outer = float(r_outer)
        self.n_inner = int(n_inner)
        self.n_main = int(n_main)
        self.n_outer = int(n_outer)
        self.rho = float(rho)
        self.nu = float(nu)
        self.p0 = float(p0)
        self.u_r_out = float(u_r_out)
        self.u_t_out = float(u_t_out)
        self.omega_r = float(omega_r)
        self.omega_t = float(omega_t)
        self.omega_p = float(omega_p)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.mesh: Mesh | None = None

    def build_mesh(self) -> None:
        """Use your Mesh builder to create nodes and set BC values on the boundary nodes."""
        self.mesh = Mesh(r_out=self.r_main, r_in=self.r_inner)
        self.mesh.buildMesh(
            p0=self.p0,
            v_r_out=self.u_r_out,
            v_t_out=self.u_t_out,
            n_inner=self.n_inner,
            n_main=self.n_main,
            n_outer=self.n_outer,
            r_outer=self.r_outer
        )

    def _initial_fields(self) -> Dict[str, List[float]]:
        """Provide an initial guess in inner→outer order."""
        assert self.mesh is not None
        N = len(self.mesh.nodes)
        rs = np.array([nd.r for nd in self.mesh.nodes], dtype=float)
        u_r0 = np.zeros(N, dtype=float)
        u_t0 = self.u_t_out * (rs - rs.min()) / (rs.max() - rs.min() + 1e-30)
        p0 = np.full(N, self.p0, dtype=float)
        return {"u_r": u_r0.tolist(), "u_theta": u_t0.tolist(), "p": p0.tolist()}

    def _bcs(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        outer_bc = {"u_r": self.u_r_out, "u_theta": self.u_t_out}
        inner_bc = {"p":   self.p0}
        return outer_bc, inner_bc

    def run(self) -> Dict[str, Any]:
        if self.mesh is None:
            self.build_mesh()
        # Fail early if Node API is incomplete
        require_node_api(self.mesh)

        solver = SORSolver(mesh=self.mesh,
                            rho=self.rho, nu=self.nu,
                            omega_r=self.omega_r, omega_t=self.omega_t, omega_p=self.omega_p,
                            tol=self.tol, max_iter=self.max_iter)
        init_fields = self._initial_fields()
        outer_bc, inner_bc = self._bcs()
        return solver.solve(init_fields=init_fields, outer_bc=outer_bc, inner_bc=inner_bc)
    
    def run_solid_body_rotation(self, N_total: int = 10, Omega: float = 25.0) -> dict:
        """
        Build a uniform-Δr mesh with ~N_total interior nodes and run the solver with:
        u_r(out) = 0
        u_θ(out) = Ω * r_out
        p(inner) = p0
        After convergence, verify dp/dr ≈ ρ u_θ^2 / r and u_r ≈ 0.
        """
        # Choose counts per region that yield ~N_total Interior Nodes
        self.n_inner = max(1, N_total // 4)
        self.n_main  = max(1, N_total // 2)
        self.n_outer = max(1, N_total - self.n_inner - self.n_main)

        # Set BCs for this test
        self.u_r_out = 0.0
        self.u_t_out = Omega * self.r_outer
        # p0 is already a parameter (inner Dirichlet)

        # Build mesh and solve
        self.build_mesh()
        result = self.run()

        # Pull fields in inner->outer order
        rs = [nd.r for nd in self.mesh.nodes]
        ur = [nd.u_r if nd.u_r is not None else 0.0 for nd in self.mesh.nodes]
        ut = [nd.u_theta if nd.u_theta is not None else 0.0 for nd in self.mesh.nodes]
        pp = [nd.p if nd.p is not None else 0.0 for nd in self.mesh.nodes]

        # Diagnostics
        max_grad_err, max_ur = solid_body_rotation_check(
            rs=rs, ur=ur, ut=ut, p=pp, rho=self.rho,
            tol_grad=2e-3, tol_ur=1e-6
        )

        # Stash into result for programmatic assertions
        result.update({
            "rs": rs, "u_r": ur, "u_theta": ut, "p": pp,
            "max_grad_err": max_grad_err, "max_ur": max_ur,
            "Omega": Omega
        })
        return result


def solid_body_rotation_check(rs, ur, ut, p, rho, tol_grad=2e-3, tol_ur=1e-6):
    """
    Post-convergence diagnostic.
    - Checks || dp/dr - rho*(u_t^2/r) ||_∞ over interior nodes.
    - Checks || u_r ||_∞.
    Prints a compact report and returns (max_grad_err, max_ur).
    """
    import numpy as np
    rs = np.asarray(rs, float)
    ur = np.asarray(ur, float)
    ut = np.asarray(ut, float)
    p  = np.asarray(p,  float)

    # central dp/dr on interior, one-sided on ends just for reporting
    dpdr = np.zeros_like(p)
    dpdr[1:-1] = (p[2:] - p[:-2]) / (rs[2:] - rs[:-2])
    dpdr[0]    = (p[1] - p[0]) / (rs[1] - rs[0])
    dpdr[-1]   = (p[-1] - p[-2]) / (rs[-1] - rs[-2])

    rhs = rho * (ut*ut) / np.maximum(rs, 1e-12)
    grad_err = dpdr - rhs
    max_grad_err = float(np.max(np.abs(grad_err[1:-1])))  # evaluate on interior only
    max_ur = float(np.max(np.abs(ur)))

    print("\n=== Solid-Body Rotation Sanity Check ===")
    print(f" max|u_r|                 = {max_ur: .3e} (target ≈ 0)")
    print(f" max|dp/dr - ρ u_θ²/r|    = {max_grad_err: .3e} (target ≈ 0)")
    print(f" thresholds: ur<{tol_ur:.1e}, grad<{tol_grad:.1e}")
    if max_ur < tol_ur and max_grad_err < tol_grad:
        print(" ✅ PASS")
    else:
        print(" ⚠️  CHECK: consider refining grid or revisiting metric/coeff assembly.")

    return max_grad_err, max_ur

# ---------- CLI entry point ----------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver for steady, axisymmetric SOR solve.")
    p.add_argument("--r_inner", type=float, default=0.05)
    p.add_argument("--r_main",  type=float, default=0.10)
    p.add_argument("--r_outer", type=float, default=0.125)
    p.add_argument("--n_inner", type=int,   default=10)
    p.add_argument("--n_main",  type=int,   default=20)
    p.add_argument("--n_outer", type=int,   default=10)

    p.add_argument("--rho",     type=float, default=1000.0)
    p.add_argument("--nu",      type=float, default=1.0e-6)

    p.add_argument("--p0",      type=float, default=0.0)
    p.add_argument("--u_r_out", type=float, default=0.0)
    p.add_argument("--u_t_out", type=float, default=1.0)

    p.add_argument("--omega_r", type=float, default=1.3)
    p.add_argument("--omega_t", type=float, default=1.3)
    p.add_argument("--omega_p", type=float, default=1.7)

    p.add_argument("--tol",      type=float, default=1e-8)
    p.add_argument("--max_iter", type=int,   default=10000)

    p.add_argument("--sanity", action="store_true",
                   help="Run solid-body rotation sanity case instead of the default run.")
    p.add_argument("--N", type=int, default=10,
                   help="Approximate number of interior nodes for sanity case.")
    p.add_argument("--Omega", type=float, default=25.0,
                   help="Angular speed for solid-body rotation sanity case.")

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    case = SorCase(
        r_inner=args.r_inner,
        r_main=args.r_main,
        r_outer=args.r_outer,
        n_inner=args.n_inner,
        n_main=args.n_main,
        n_outer=args.n_outer,
        rho=args.rho,
        nu=args.nu,
        p0=args.p0,
        u_r_out=args.u_r_out,
        u_t_out=args.u_t_out,
        omega_r=args.omega_r,
        omega_t=args.omega_t,
        omega_p=args.omega_p,
        tol=args.tol,
        max_iter=args.max_iter
    )
    if args.sanity:
        res = case.run_solid_body_rotation(N_total=args.N, Omega=args.Omega)
        iters = res.get("iterations", None)
        final = res.get("final_norm", None)
        print(f"\n[Sanity] Converged in {iters} iterations with max-norm Δ = {final:.3e}")
        return 0

    try:
        result = case.run()
    except NotImplementedError as e:
        # Clear, actionable guidance if Node API is not ready
        print("\n[ERROR] Incomplete Node implementation for SORSolver:\n")
        print(str(e))
        return 2
    except ValueError as e:
        print("\n[ERROR] Value error in setup:\n")
        print(str(e))
        return 3
    except Exception as e:
        print("\n[ERROR] Unexpected failure during solve:\n")
        print(repr(e))
        return 4

    # Pretty print results
    iters = result.get("iterations", None)
    final = result.get("final_norm", None)
    print(f"\nConverged in {iters} iterations with max-norm Δ = {final:.3e}")
    # Optional: print last few history entries
    hist = result.get("history", [])
    tail = hist[-5:] if len(hist) > 5 else hist
    for h in tail:
        print(f"  iter {h['iter']:6d}  max_norm {h['max_norm']:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
