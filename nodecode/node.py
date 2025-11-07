from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _nan_debug_dump(phase, r, dW, dE, vals: dict):
    parts = [f"[NAN-DBG] {phase} r={r:.9e} dW={dW:.9e} dE={dE:.9e}"]
    for k, v in vals.items():
        parts.append(f"{k}={v!r}")
    print("  ".join(parts))

#   aw*Ui-1    +   ap*ui    +   ae*Ui+1     =   b
@dataclass
class EqCoeffs:
    aP: float = 0.0
    aW: float = 0.0
    aE: float = 0.0
    b: float = 0.0

@dataclass
class EqResiduals:
    R: float = 1.0   # local residual (e.g., aP*phi - (aW*phi_W + aE*phi_E + b))

# --- Residual history (module-level utilities) --------------------------------
# We store the MAX-norm of the pointwise residuals over all nodes each iteration.
_RES_HIST = {'iter': [], 'ur_max': [], 'ut_max': [], 'p_max': []}

def reset_residual_history() -> None:
    """Clear the stored residual history."""
    for k in _RES_HIST:
        _RES_HIST[k].clear()

def collect_residual_norms(nodes: list["Node"], it: int) -> None:
    """
    Compute and store the max |residual| across all nodes for each equation.
    Non-finite values are ignored; if all are non-finite, we store a tiny epsilon.
    """
    eps = 1e-16

    def _safe_max(vals):
        # vals may be a generator; realize it once
        arr = np.asarray([abs(v) for v in vals if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return eps
        m = float(np.max(arr))
        # clamp to be strictly positive for log plots
        return m if (np.isfinite(m) and m > 0.0) else eps

    ur_max = _safe_max([nd.res_r.R for nd in nodes])
    ut_max = _safe_max([nd.res_t.R for nd in nodes])
    p_max  = _safe_max([nd.res_p.R for nd in nodes])

    _RES_HIST['iter'].append(int(it))
    _RES_HIST['ur_max'].append(ur_max)
    _RES_HIST['ut_max'].append(ut_max)
    _RES_HIST['p_max'].append(p_max)

def plot_residual_history(*, show: bool = True, savepath: str | None = None) -> None:
    if not _RES_HIST['iter']:
        print("[PLOT] No residual history collected; nothing to plot.")
        return

    its = np.asarray(_RES_HIST['iter'], dtype=float)
    ur  = np.asarray(_RES_HIST['ur_max'], dtype=float)
    ut  = np.asarray(_RES_HIST['ut_max'], dtype=float)
    pp  = np.asarray(_RES_HIST['p_max'],  dtype=float)

    eps = 1e-16

    def scrub(arr):
        # Replace non-finite with eps, and clamp non-positives to eps
        out = np.nan_to_num(arr, nan=eps, posinf=np.finfo(float).max/1e6, neginf=eps)
        out[out <= 0.0] = eps
        return out

    ur, ut, pp = scrub(ur), scrub(ut), scrub(pp)

    # Choose safe y-limits
    y_all = np.concatenate([ur, ut, pp])
    y_min = float(np.min(y_all[np.isfinite(y_all)])) if np.any(np.isfinite(y_all)) else eps
    y_max = float(np.max(y_all[np.isfinite(y_all)])) if np.any(np.isfinite(y_all)) else 1.0
    y_min = max(y_min, eps)
    y_max = max(y_max, y_min * 1.1)  # ensure some dynamic range

    fig, ax = plt.subplots()
    ax.set_yscale('log', nonpositive='clip')
    ax.plot(its, ur, label=r"$u_r$ residual (max)")
    ax.plot(its, ut, label=r"$u_\theta$ residual (max)")
    ax.plot(its, pp, label=r"$p$ residual (max)")
    ax.set_ylim(y_min/10.0, y_max*10.0)  # force safe bounds
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |residual|")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
# -------------------------------------------------------------------------------

class Node:
    def __init__(self, radius: float) -> None:
        # Primary unknowns
        self._u_r: Optional[float] = None
        self._u_theta: Optional[float] = None
        self._p: Optional[float] = None

        # Geometry
        self._r: float = float(radius)
        self._region: Optional[str] = None

        # Radial neighbor links (inner = west, outer = east)
        self.inner: Optional["Node"] = None
        self.outer: Optional["Node"] = None

        # Per-equation coefficient storage
        self.coeffs_r: EqCoeffs = EqCoeffs() # u_r equation
        self.coeffs_t: EqCoeffs = EqCoeffs() # u_theta equation
        self.coeffs_p: EqCoeffs = EqCoeffs() # pressure equation

        # Residuals (Diagnostics only)
        self.res_r: EqResiduals = EqResiduals()
        self.res_t: EqResiduals = EqResiduals()
        self.res_p: EqResiduals = EqResiduals()

        # Previous-iteration snapshots (for Δ max-norm)
        self._u_r_prev: Optional[float] = None
        self._u_theta_prev: Optional[float] = None
        self._p_prev: Optional[float] = None

        # Boundary flags (enforced by solver)
        self.is_outer_bc: bool = False # Dirichlet: u_r, u_theta
        self.is_inner_bc: bool = False # Dirichlet: p

    # --- neighbor linking helper ---
    def set_neighbors(self, inner: Optional["Node"], outer: Optional["Node"]) -> None:
        self.inner = inner
        self.outer = outer
    
    # --- iteration-scope snapshots ---
    def snapshot_prev(self) -> None:
        self._u_r_prev = self._u_r
        self._u_theta_prev = self._u_theta
        self._p_prev = self._p

    # --- local spacing helpers (supports non-uniform Δr) ---
    def _drW(self) -> float:
        return float('nan') if self.inner is None else float(self._r - self.inner._r)

    def _drE(self) -> float:
        return float('nan') if self.outer is None else float(self.outer._r - self._r)


    # --- local Δ magnitudes for convergence tracking ---
    def local_deltas(self) -> Tuple[float, float, float]:
        du_r = 0.0 if self._u_r_prev is None or self._u_r is None else abs(self._u_r - self._u_r_prev)
        du_t = 0.0 if self._u_theta_prev is None or self._u_theta is None else abs(self._u_theta - self._u_theta_prev)
        dp = 0.0 if self._p_prev is None or self._p is None else abs(self._p - self._p_prev)
        return du_r, du_t, dp

    # --- coefficient assembly (second-order, non-uniform central in r) ---
    def assemble_coeffs_u_r(self, rho: float, nu: float, mass_coeff: float = 0.0) -> None:
        W, E = self.inner, self.outer

        # If this node is a Dirichlet boundary, leave to the BC handler.
        if W is None or E is None:
            self.coeffs_r = EqCoeffs(aP=1.0, aW=0.0, aE=0.0, b=0.0)
            return

        # ----------------------------
        # Geometry & spacings (explicit names)
        # ----------------------------
        r_i: float      = float(self._r)
        dr_W: float     = float(self._r - W._r)        # Δr_W = r_i   − r_{i-1}
        dr_E: float     = float(E._r - self._r)        # Δr_E = r_{i+1} − r_i
        d_r:  float     = 0.5 * (dr_W + dr_E)          # centered Δr for the stencil

        # Guard against pathologies
        eps = 1e-12
        if not (np.isfinite(d_r) and d_r > eps and r_i > eps):
            self.coeffs_r = EqCoeffs(aP=np.nan, aW=np.nan, aE=np.nan, b=np.nan)
            return

        rinv      = 1.0 / r_i
        rinv2     = rinv * rinv
        inv_dr    = 1.0 / d_r
        inv_dr2   = inv_dr * inv_dr

        # ----------------------------
        # Coefficients aW, aE, aP  (implicit diffusion + metric sink on diag)
        # ----------------------------
        aW: float = -nu * (  inv_dr2 - 0.5 * rinv * inv_dr )
        aE: float = -nu * (  inv_dr2 + 0.5 * rinv * inv_dr )
        aP: float =  mass_coeff + aW + aE + nu * rinv2    # rho/Δt + aW + aE + ν/r_i^2

        # ----------------------------
        # Explicit RHS b_i (pressure grad + convection + centrifugal)
        #   All “old/prev” values are taken from the previous iterate.
        # ----------------------------
        # Pressures
        p_im1: float = 0.0 if W._p       is None else float(W._p)
        p_ip1: float = 0.0 if E._p       is None else float(E._p)

        # Radial velocities (previous iterate)
        ur_im1_old: float = 0.0 if W._u_r_prev     is None else float(W._u_r_prev)
        ur_i_old:   float = 0.0 if self._u_r_prev  is None else float(self._u_r_prev)
        ur_ip1_old: float = 0.0 if E._u_r_prev     is None else float(E._u_r_prev)

        # Tangential velocity (previous iterate)
        ut_i_old:   float = 0.0 if self._u_theta_prev is None else float(self._u_theta_prev)

        # Centered derivatives that appear in the explicit pieces
        dp_dr_centered: float   = (p_ip1 - p_im1) * 0.5 * inv_dr
        dur_dr_centered_old: float = (ur_ip1_old - ur_im1_old) * 0.5 * inv_dr

        b_press: float   = -(1.0 / rho) * dp_dr_centered
        b_conv:  float   = -rho * ur_i_old * dur_dr_centered_old
        b_cent:  float   =  rho * (ut_i_old * ut_i_old) * rinv

        b_i: float = b_press + b_conv + b_cent


        # Save row
        self.coeffs_r = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=float(b_i))

    def assemble_coeffs_u_theta(self, rho: float, nu: float, mass_coeff: float = 0.0) -> None:
        W, E = self.inner, self.outer

        # Dirichlet nodes are handled elsewhere
        if W is None or E is None:
            self.coeffs_t = EqCoeffs(aP=1.0, aW=0.0, aE=0.0, b=0.0)
            return

        # ----------------------------
        # Geometry & spacings
        # ----------------------------
        r_i: float  = float(self._r)
        dr_W: float = float(self._r - W._r)     # Δr_W = r_i - r_{i-1}
        dr_E: float = float(E._r - self._r)     # Δr_E = r_{i+1} - r_i
        d_r:  float = 0.5 * (dr_W + dr_E)       # centered Δr

        eps = 1e-12
        if not (np.isfinite(d_r) and d_r > eps and r_i > eps):
            self.coeffs_t = EqCoeffs(aP=np.nan, aW=np.nan, aE=np.nan, b=np.nan)
            return

        rinv    = 1.0 / r_i
        rinv2   = rinv * rinv
        inv_dr  = 1.0 / d_r
        inv_dr2 = inv_dr * inv_dr

        # ----------------------------
        # Implicit diffusion (same operator as u_r)
        # ----------------------------
        aW: float = -nu * (  inv_dr2 - 0.5 * rinv * inv_dr )
        aE: float = -nu * (  inv_dr2 + 0.5 * rinv * inv_dr )
        aP: float =  mass_coeff + aW + aE + nu * rinv2

        # ----------------------------
        # Explicit RHS (previous iterate values)
        # ----------------------------
        # Velocities
        ut_im1_old: float = 0.0 if W._u_theta_prev    is None else float(W._u_theta_prev)
        ut_i_old:   float = 0.0 if self._u_theta_prev is None else float(self._u_theta_prev)
        ut_ip1_old: float = 0.0 if E._u_theta_prev    is None else float(E._u_theta_prev)

        ur_i_old:   float = 0.0 if self._u_r_prev     is None else float(self._u_r_prev)

        # Centered derivative of u_theta
        dut_dr_centered_old: float = (ut_ip1_old - ut_im1_old) * 0.5 * inv_dr

        # b pieces
        b_conv:   float = -rho * ( ur_i_old * dut_dr_centered_old )
        b_metric: float = -rho * ( ur_i_old * ut_i_old * rinv )
        b_mass:   float =  mass_coeff * ut_i_old

        b_i: float = b_conv + b_metric + b_mass

        # Save row
        self.coeffs_t = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=float(b_i))

    def assemble_coeffs_p(self, rho: float, nu: float) -> None:
        W, E = self.inner, self.outer

        # If boundary, leave for BC handler (Dirichlet/Neumann applied elsewhere)
        if W is None or E is None:
            self.coeffs_p = EqCoeffs(aP=1.0, aW=0.0, aE=0.0, b=0.0)
            return

        # ----------------------------
        # Geometry & spacings (explicit names)
        # ----------------------------
        r_i: float  = float(self._r)
        r_W: float  = float(W._r)
        r_E: float  = float(E._r)

        dr_W: float = r_i - r_W          # Δr_W
        dr_E: float = r_E - r_i          # Δr_E
        d_r:  float = 0.5 * (dr_W + dr_E)  # centered Δr

        eps = 1e-12
        if not (np.isfinite(d_r) and d_r > eps and r_i > eps and dr_W > eps and dr_E > eps):
            self.coeffs_p = EqCoeffs(aP=np.nan, aW=np.nan, aE=np.nan, b=np.nan)
            return

        rinv    = 1.0 / r_i
        rinv2   = rinv * rinv
        inv_dr  = 1.0 / d_r
        inv_dr2 = inv_dr * inv_dr

        # Face radii
        r_Wf: float = 0.5 * (r_i + r_W)
        r_Ef: float = 0.5 * (r_E + r_i)

        # ----------------------------
        # LHS coefficients (r-weighted Poisson)
        # ----------------------------
        aW: float =  r_Wf / dr_W
        aE: float =  r_Ef / dr_E
        aP: float =  aW + aE

        # ----------------------------
        # RHS b_i (explicit from previous iterate)
        # ----------------------------
        # Previous-iterate velocities
        ur_im1_old: float = 0.0 if W._u_r_prev        is None else float(W._u_r_prev)
        ur_i_old:   float = 0.0 if self._u_r_prev     is None else float(self._u_r_prev)
        ur_ip1_old: float = 0.0 if E._u_r_prev        is None else float(E._u_r_prev)
        ut_i_old:   float = 0.0 if self._u_theta_prev is None else float(self._u_theta_prev)

        # (1) Divergence term: (1/r) * d(r u_r)/dr (centered)
        q_im1 = r_W * ur_im1_old
        q_i   = r_i * ur_i_old
        q_ip1 = r_E * ur_ip1_old
        d_q_dr_centered = (q_ip1 - q_im1) * 0.5 * inv_dr
        div_term = rinv * d_q_dr_centered
        b_div   = rho * div_term

        # (2) Swirl sink: -(u_theta^2)/r^2
        b_swirl = - rho * (ut_i_old * ut_i_old) * rinv2

        # (3) Radial viscous coupling of u_r: -ν [ u_r'' + (1/r) u_r' ]
        du_dr_centered  = (ur_ip1_old - ur_im1_old) * 0.5 * inv_dr
        d2u_dr2_centered = (ur_ip1_old - 2.0*ur_i_old + ur_im1_old) * inv_dr2
        visc_couple = d2u_dr2_centered + rinv * du_dr_centered
        b_visc  = - nu * visc_couple

        b_i: float = b_div + b_swirl + b_visc

        # Save row
        self.coeffs_p = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=float(b_i))
    
    def compute_continuity_residual(self) -> float:
        W, E = self.inner, self.outer
        if W is None or E is None:
            # No proper control volume (boundary node) → no continuity residual
            return 0.0

        eps = 1e-14
        rW = float(W._r)
        rP = float(self._r)
        rE = float(E._r)

        # Face radii
        rw = 0.5 * (rW + rP)
        re = 0.5 * (rP + rE)

        # Control-volume width for node-centered scheme
        dr_i = re - rw
        if dr_i <= eps or rP <= eps:
            return 0.0

        # Cell-center velocities (current iterate)
        uW = 0.0 if W._u_r is None else float(W._u_r)
        uP = 0.0 if self._u_r is None else float(self._u_r)
        uE = 0.0 if E._u_r is None else float(E._u_r)

        # Face velocities (second order)
        u_w = 0.5 * (uW + uP)   # u_{r, i-1/2}
        u_e = 0.5 * (uP + uE)   # u_{r, i+1/2}

        # Signed discrete continuity defect at node i
        return (re * u_e - rw * u_w) / (rP * dr_i)
    
    
    # --- local residuals (diagnostics) ---
    def update_local_residuals(self) -> None:
        # ---- u_r residual ----
        if self.inner is not None and self.outer is not None and self._u_r is not None:
            cr = self.coeffs_r
            phiW = 0.0 if self.inner._u_r is None else float(self.inner._u_r)
            phiE = 0.0 if self.outer._u_r is None else float(self.outer._u_r)

            lhs = cr.aP * float(self._u_r)
            rhs = cr.aW * phiW + cr.aE * phiE + cr.b

            self.res_r.R = lhs - rhs
        else:
            self.res_r.R = 0.0

        # ---- u_theta residual ----
        if self.inner is not None and self.outer is not None and self._u_theta is not None:
            ct = self.coeffs_t
            phiW = 0.0 if self.inner._u_theta is None else float(self.inner._u_theta)
            phiE = 0.0 if self.outer._u_theta is None else float(self.outer._u_theta)

            lhs = ct.aP * float(self._u_theta)
            rhs = ct.aW * phiW + ct.aE * phiE + ct.b

            self.res_t.R = lhs - rhs
        else:
            self.res_t.R = 0.0

        # ---- "p" residual = continuity residual (mass conservation) ----
        if self.inner is not None and self.outer is not None and self._p is not None:
            cp = self.coeffs_p
            pW = 0.0 if self.inner._p is None else float(self.inner._p)
            pE = 0.0 if self.outer._p is None else float(self.outer._p)

            lhs = cp.aP * float(self._p)
            rhs = cp.aW * pW + cp.aE * pE + cp.b

            self.res_p.R = lhs - rhs
        else:
            self.res_p.R = 0.0

    # --- SOR Updates (GS if w == 1)
    def sor_update_u_r(self, omega_r: float) -> None:
        if self.inner is None or self.outer is None or self.is_outer_bc or self.is_inner_bc:
            return
        cr = self.coeffs_r
        if (not np.all(np.isfinite([cr.aP, cr.aW, cr.aE, cr.b]))) or (cr.aP == 0.0):
            return
        uW = 0.0 if self.inner._u_r is None else self.inner._u_r
        uE = 0.0 if self.outer._u_r is None else self.outer._u_r
        num = cr.aW*uW + cr.aE*uE + cr.b
        if not np.isfinite(num):
            return
        u_new = num / cr.aP
    
        if np.isfinite(u_new):
            if self._u_r is None:
                self._u_r = 0.0
            self._u_r = (1.0 - omega_r)*self._u_r + omega_r*u_new


    def sor_update_u_theta(self, omega_t: float) -> None:
        if self.inner is None or self.outer is None or self.is_outer_bc or self.is_inner_bc:
            return
        ct = self.coeffs_t
        if (not np.all(np.isfinite([ct.aP, ct.aW, ct.aE, ct.b]))) or (ct.aP == 0.0):
            return
        tW = 0.0 if self.inner._u_theta is None else self.inner._u_theta
        tE = 0.0 if self.outer._u_theta is None else self.outer._u_theta
        num = ct.aW*tW + ct.aE*tE + ct.b
        if not np.isfinite(num):
            return
        t_new = num / ct.aP
        if np.isfinite(t_new):
            if self._u_theta is None:
                self._u_theta = 0.0
            self._u_theta = (1.0 - omega_t)*self._u_theta + omega_t*t_new


    def sor_update_p(self, omega_p: float) -> None:
        if self.inner is None or self.outer is None:
            return
        if self.is_inner_bc:
            return
        cp = self.coeffs_p
        if (not np.all(np.isfinite([cp.aP, cp.aW, cp.aE, cp.b]))) or (cp.aP == 0.0):
            return
        pW = 0.0 if self.inner._p is None else self.inner._p
        pE = 0.0 if self.outer._p is None else self.outer._p
        num = cp.aW*pW + cp.aE*pE + cp.b
        if not np.isfinite(num):
            return
        p_new = num / cp.aP
        if np.isfinite(p_new):
            if self._p is None:
                self._p = 0.0
            self._p = (1.0 - omega_p)*self._p + omega_p*p_new

    #region Properties 
   # ==== Properties kept minimal and consistent with usage ====
    @property
    def region(self) -> Optional[str]:
        return self._region

    @region.setter
    def region(self, value: str) -> None:
        self._region = value

    @property
    def u_r(self):
        return self._u_r

    @u_r.setter
    def u_r(self, v):
        self._u_r = v

    @property
    def u_theta(self):
        return self._u_theta

    @u_theta.setter
    def u_theta(self, v):
        self._u_theta = v

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, v):
        self._p = v

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, v):
        self._r = float(v)

    #endregion