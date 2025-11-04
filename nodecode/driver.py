# sor_driver.py
from __future__ import annotations
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
from mesh import Mesh
from SORSolver import SORSolver

class SorCase:
    """
    Encapsulates one steady, axisymmetric solve with piecewise-uniform radial mesh.
    """
    def __init__(self,
                r_inner: float,
                r_main: float,
                r_outer: float,
                n_inner: int,
                n_main: int,
                n_outer: int,
                rho: float,
                nu: float,
                p0: float,
                u_r_out: float,
                u_t_out: float,
                omega_r: float,
                omega_t: float,
                omega_p: float,
                tol: float,
                max_iter: int,
                pseudo_dt: float = 1e-4):
        
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
        self.pseudo_dt = float(pseudo_dt)

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

    def run(self, init_fields = None):
        if self.mesh is None:
            self.build_mesh()
        solver = SORSolver(mesh=self.mesh,
                        rho=self.rho, nu=self.nu,
                        omega_r=self.omega_r, omega_t=self.omega_t, omega_p=self.omega_p,
                        tol=self.tol, max_iter=self.max_iter,
                        pseudo_dt=getattr(self, "pseudo_dt", None))
        # use override if provided
        init_fields = init_fields if init_fields is not None else self._initial_fields()
        outer_bc, inner_bc = self._bcs()
        return solver.solve(init_fields=init_fields, outer_bc=outer_bc, inner_bc=inner_bc)
    

def main(argv: List[str]) -> int:
    case = SorCase(
                r_inner = 0.05,
                r_main  = 0.10,
                r_outer = 0.125,
                n_inner = 1,
                n_main  = 3,
                n_outer = 1,
                rho     = 1000.0,
                nu      = 1.0e-6,
                p0      = 101325.0,
                u_r_out = 5.0,
                u_t_out = 2.0,
                omega_r = 1.0,
                omega_t = 1.0,
                omega_p = 1.3,
                tol     = 1e-8,
                max_iter= 10_000,
                pseudo_dt= 1e-2)
                

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
