from typing import Dict, Any, List, Tuple
from node import EqCoeffs
import math

class SORSolver:
    """
    Steady-state, axisymmetric, uniform-Δr SOR solver.
    - Single pass per iteration: update u_r, u_theta, p (outer -> inner).
    - Dirichlet BCs enforced every sweep:
        * outermost: u_r, u_theta
        * innermost: p
    """
    def __init__(self,
                 mesh,
                 rho: float,
                 nu: float,
                 dr: float,
                 omega_r: float = 1.0,
                 omega_t: float = 1.0,
                 omega_p: float = 1.0,
                 tol: float = 1e-8,
                 max_iter: int = 10000):
        self.mesh = mesh
        self.rho = rho
        self.nu = nu      # kinematic viscosity
        self.dr = dr      # uniform radial spacing
        self.omega_r = omega_r
        self.omega_t = omega_t
        self.omega_p = omega_p
        self.tol = tol
        self.max_iter = max_iter
        self.history: List[Dict[str, Any]] = []

    # --- public API ---
    def solve(self,
              init_fields: Dict[str, List[float]],
              outer_bc: Dict[str, float],
              inner_bc: Dict[str, float]) -> Dict[str, Any]:
        """
        init_fields: {'u_r': [...], 'u_theta': [...], 'p': [...] } in node order (inner->outer or any; we sort)
        outer_bc:    {'u_r': value, 'u_theta': value}
        inner_bc:    {'p': value}
        Returns: {'iterations': N, 'final_norm': val, 'history': [...]}
        """
        self._prepare_mesh_links()
        self._apply_initial_guess(init_fields)
        self._mark_boundaries_and_values(outer_bc, inner_bc)

        # Main SOR loop
        for it in range(1, self.max_iter + 1):
            # Snapshots for Δ
            for nd in self.mesh.nodes:
                nd.snapshot_prev()

            # Assemble coefficients at current iterate
            for nd in self.mesh.nodes:
                nd.assemble_coeffs_u_r(self.rho, self.nu, self.dr)
                nd.assemble_coeffs_u_theta(self.rho, self.nu, self.dr)
                nd.assemble_coeffs_p(self.dr)

            # Enforce BCs before sweep (keeps aP/b consistent for bounded nodes)
            self._apply_bcs()

            # Single pass sweep: OUTER -> INNER
            ordered = sorted(self.mesh.nodes, key=lambda n: n.r, reverse=True)
            for nd in ordered:
                nd.sor_update_u_r(self.omega_r)
                nd.sor_update_u_theta(self.omega_t)
                nd.sor_update_p(self.omega_p)

            # Enforce BCs after sweep as well (Dirichlet hard-set)
            self._apply_bcs()

            # Residuals (optional; useful for debugging)
            for nd in self.mesh.nodes:
                nd.update_local_residuals()

            # Global max-norm over updates (Δ)
            max_delta = 0.0
            for nd in self.mesh.nodes:
                du_r, du_t, dp = nd.local_deltas()
                max_delta = max(max_delta, du_r, du_t, dp)

            self.history.append({'iter': it, 'max_norm': max_delta})

            if max_delta <= self.tol:
                return {'iterations': it, 'final_norm': max_delta, 'history': self.history}

        # If we reach here, not converged within max_iter
        return {'iterations': self.max_iter, 'final_norm': self.history[-1]['max_norm'], 'history': self.history}

    # --- helpers ---
    def _prepare_mesh_links(self) -> None:
        # Prefer using your Mesh helper if present
        if hasattr(self.mesh, "link_radial_neighbors"):
            self.mesh.link_radial_neighbors()
        else:
            # Fallback: derive links by sorting radii
            self.mesh.nodes.sort(key=lambda nd: nd.r)
            for i, nd in enumerate(self.mesh.nodes):
                nd.set_neighbors(
                    inner=self.mesh.nodes[i-1] if i > 0 else None,
                    outer=self.mesh.nodes[i+1] if i < len(self.mesh.nodes)-1 else None
                )

    def _apply_initial_guess(self, init_fields: Dict[str, List[float]]) -> None:
        # Align init arrays with nodes sorted inner->outer
        self.mesh.nodes.sort(key=lambda nd: nd.r)
        N = len(self.mesh.nodes)
        for key in ('u_r','u_theta','p'):
            if key not in init_fields or len(init_fields[key]) != N:
                raise ValueError(f"init_fields['{key}'] must be length {N}.")
        for i, nd in enumerate(self.mesh.nodes):
            nd.u_r = float(init_fields['u_r'][i])
            nd.u_theta = float(init_fields['u_theta'][i])
            nd.p = float(init_fields['p'][i])

    def _mark_boundaries_and_values(self, outer_bc: Dict[str, float], inner_bc: Dict[str, float]) -> None:
        # Identify inner/outer nodes after sort
        self.mesh.nodes.sort(key=lambda nd: nd.r)
        inner = self.mesh.nodes[0]
        outer = self.mesh.nodes[-1]
        # mark flags
        inner.is_inner_bc = True
        outer.is_outer_bc = True
        # store desired values in the node objects (used in _apply_bcs)
        if 'p' not in inner_bc:
            raise ValueError("inner_bc must include 'p'")
        inner._p = float(inner_bc['p'])

        if 'u_r' not in outer_bc or 'u_theta' not in outer_bc:
            raise ValueError("outer_bc must include 'u_r' and 'u_theta'")
        outer._u_r = float(outer_bc['u_r'])
        outer._u_theta = float(outer_bc['u_theta'])

    def _apply_bcs(self) -> None:
        # Hard set BC nodes every pass
        # (Optionally, zero out their equation coefficients to an identity to avoid drift)
        for nd in self.mesh.nodes:
            if nd.is_outer_bc:
                # Outer Dirichlet: u_r, u_theta fixed
                nd.coeffs_r = EqCoeffs(1.0, 0.0, 0.0, nd._u_r if nd._u_r is not None else 0.0)
                nd.coeffs_t = EqCoeffs(1.0, 0.0, 0.0, nd._u_theta if nd._u_theta is not None else 0.0)
            if nd.is_inner_bc:
                # Inner Dirichlet: p fixed
                nd.coeffs_p = EqCoeffs(1.0, 0.0, 0.0, nd._p if nd._p is not None else 0.0)
