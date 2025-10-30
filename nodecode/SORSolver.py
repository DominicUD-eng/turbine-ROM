import numpy as np
from typing import Dict, Any, List
from node import EqCoeffs

class SORSolver:
    """
    Steady-state, axisymmetric, uniform-Δr SOR solver.
    - Single pass per iteration: update u_r, u_theta, p (outer -> inner).
    - Dirichlet BCs enforced every sweep:
        * outermost: u_r, u_theta
        * innermost: p
    """
    def __init__(self, mesh, rho: float, nu: float,
                omega_r: float, omega_t: float, omega_p: float,
                tol: float, max_iter: int) -> None:
        self.mesh = mesh
        self.rho = float(rho)
        self.nu = float(nu)
        self.omega_r = float(omega_r)
        self.omega_t = float(omega_t)
        self.omega_p = float(omega_p)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.history: List[Dict[str, float]] = []


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

        # Connectivity debug
        print("\\n[DEBUG] Mesh Connectivity (inner → outer):")
        for nd in sorted(self.mesh.nodes, key=lambda n: n.r):
            rin = nd.inner.r if nd.inner else None
            rout = nd.outer.r if nd.outer else None
            print(f"  r={nd.r:.6e}  inner={rin}  outer={rout}")
        print("[DEBUG] End connectivity\\n")

        # Main SOR loop
        for it in range(1, self.max_iter + 1):
            # Snapshots for Δ
            for nd in self.mesh.nodes:
                nd.snapshot_prev()

            # Assemble coefficients at current iterate
            for nd in self.mesh.nodes:
                nd.assemble_coeffs_u_r(self.rho, self.nu)
                nd.assemble_coeffs_u_theta(self.rho, self.nu)
                nd.assemble_coeffs_p(self.rho)

            # Enforce BCs before sweep
            self._apply_bcs()

            # Observe-only coefficient validation
            self._debug_validate_coeffs(stage=f"post-assemble+BC it={it}")

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
                if not np.all(np.isfinite([du_r, du_t, dp])):
                    print(f"[WARN] Non-finite Δ at r={nd.r:.6e}: du_r={du_r!r} du_t={du_t!r} dp={dp!r}")
                max_delta = max(max_delta, du_r, du_t, dp)

            self.history.append({'iter': it, 'max_norm': max_delta})
            if max_delta <= self.tol:
                return {'iterations': it, 'final_norm': max_delta, 'history': self.history}

        # If we reach here, not converged within max_iter
        final_norm = self.history[-1]['max_norm'] if self.history else float('nan')
        return {'iterations': self.max_iter, 'final_norm': final_norm, 'history': self.history}

    # --- helpers ---
    def _prepare_mesh_links(self) -> None:
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
        self.mesh.nodes.sort(key=lambda nd: nd.r)
        N = len(self.mesh.nodes)
        for key in ('u_r','u_theta','p'):
            if key not in init_fields or len(init_fields[key]) != N:
                print(f"[WARN] init_fields['{key}'] length != N ({N})")
        for i, nd in enumerate(self.mesh.nodes):
            nd.u_r     = float(init_fields['u_r'][i])
            nd.u_theta = float(init_fields['u_theta'][i])
            nd.p       = float(init_fields['p'][i])

    def _mark_boundaries_and_values(self, outer_bc: Dict[str, float], inner_bc: Dict[str, float]) -> None:
        self.mesh.nodes.sort(key=lambda nd: nd.r)
        inner = self.mesh.nodes[0]
        outer = self.mesh.nodes[-1]

        inner.is_inner_bc = True
        outer.is_outer_bc = True

        inner._p = float(inner_bc['p']) if 'p' in inner_bc else None
        outer._u_r = float(outer_bc['u_r']) if 'u_r' in outer_bc else None
        outer._u_theta = float(outer_bc['u_theta']) if 'u_theta' in outer_bc else None

        print(f"[DBG] BCs: inner r={inner.r:.6e} p*={inner._p!r} | outer r={outer.r:.6e} ur*={outer._u_r!r} ut*={outer._u_theta!r}")
        

    def _apply_bcs(self) -> None:
        for nd in self.mesh.nodes:
            if nd.is_outer_bc:
                # Outer Dirichlet rows
                nd.coeffs_r = EqCoeffs(1.0, 0.0, 0.0, nd._u_r if nd._u_r is not None else 0.0)
                nd.coeffs_t = EqCoeffs(1.0, 0.0, 0.0, nd._u_theta if nd._u_theta is not None else 0.0)
            if nd.is_inner_bc:
                # Inner Dirichlet for pressure
                nd.coeffs_p = EqCoeffs(1.0, 0.0, 0.0, nd._p if nd._p is not None else 0.0)
                
    def _debug_validate_coeffs(self, stage: str) -> None:
        """Print-only validation; NO forcing or raising."""
        for nd in self.mesh.nodes:
            r = nd.r
            for name, C in (("u_r", nd.coeffs_r), ("u_theta", nd.coeffs_t), ("p", nd.coeffs_p)):
                arr = np.array([C.aP, C.aW, C.aE, C.b], dtype=float)
                if not np.all(np.isfinite(arr)):
                    print(f"[COEFF] {stage} r={r:.9e} {name} aP={C.aP!r} aW={C.aW!r} aE={C.aE!r} b={C.b!r} (non-finite)")