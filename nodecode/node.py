from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

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
    R: float = 0.0   # local residual (e.g., aP*phi - (aW*phi_W + aE*phi_E + b))

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
    def assemble_coeffs_u_r(self, rho: float, nu: float) -> None:
        W, E = self.inner, self.outer
        if W is None or E is None:
            self.coeffs_r = EqCoeffs(1.0, 0.0, 0.0, 0.0)
            return

        r  = float(self._r)
        dW = self._drW()
        dE = self._drE()
        eps = 1e-12

        if (not np.isfinite(dW)) or (not np.isfinite(dE)) or (dW <= eps) or (dE <= eps) or ((dW + dE) <= eps):
            _nan_debug_dump("u_r bad spacing", r, dW, dE, {})
            # Produce NaN row to signal invalid geometry; SOR will skip this row.
            self.coeffs_r = EqCoeffs(np.nan, np.nan, np.nan, np.nan)
            return
        
        rinv  = 1.0 / (r if r > eps else eps)
        rinv2 = rinv * rinv

        # operators
        alphaW = - dE / (dW * (dW + dE))
        alphaE =   dW / (dE * (dW + dE))
        alphaP = (dE - dW) / (dW * dE)

        betaW =  2.0 / ((dE + dW) * dW)
        betaE =  2.0 / ((dE + dW) * dE)
        betaP = - (betaW + betaE)

        # debug operators
        
        if not np.all(np.isfinite([alphaW, alphaP, alphaE, betaW, betaP, betaE])):
            _nan_debug_dump("u_r operators non-finite", r, dW, dE, dict(alphaW=alphaW, alphaP=alphaP, alphaE=alphaE,
                                 betaW=betaW, betaP=betaP, betaE=betaE))

        # current iterate values
        uW = 0.0 if W._u_r is None else float(W._u_r)
        uP = 0.0 if self._u_r is None else float(self._u_r)
        uE = 0.0 if E._u_r is None else float(E._u_r)
        tP = 0.0 if self._u_theta is None else float(self._u_theta)
        pW = 0.0 if W._p is None else float(W._p)
        pP = 0.0 if self._p is None else float(self._p)
        pE = 0.0 if E._p is None else float(E._p)

        # diffusion + metric
        aW_diff = nu * (betaW + rinv * alphaW)
        aE_diff = nu * (betaE + rinv * alphaE)
        aP_diff = nu * (betaP + rinv * alphaP)

        # convection (linearized)
        aW_conv =  rho * uP * alphaW
        aE_conv =  rho * uP * alphaE
        aP_conv =  rho * uP * alphaP

        # totals
        aW = aW_diff + aW_conv
        aE = aE_diff + aE_conv
        aP = (-(aW + aE)) + (aP_diff + aP_conv) - (nu*rinv2)

        # RHS
        dp_dr_i = alphaW * pW + alphaP * pP + alphaE * pE
        b = - dp_dr_i + rho * (tP * tP) * rinv
        
        # Observe-only debug
        if not np.all(np.isfinite([aW, aE, aP, b])):
            _nan_debug_dump("u_r coeffs non-finite", r, dW, dE,
                            dict(aW=aW, aE=aE, aP=aP, b=b,
                                 aW_diff=aW_diff, aE_diff=aE_diff, aP_diff=aP_diff,
                                 aW_conv=aW_conv, aE_conv=aE_conv, aP_conv=aP_conv,
                                 dp_dr_i=dp_dr_i, tP=tP))

        self.coeffs_r = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=float(b))

    def assemble_coeffs_u_theta(self, rho: float, nu: float) -> None:
        W, E = self.inner, self.outer
        if W is None or E is None:
            self.coeffs_t = EqCoeffs(1.0, 0.0, 0.0, 0.0)
            return

        r  = float(self._r)
        dW = self._drW()
        dE = self._drE()
        eps = 1e-12

        if (not np.isfinite(dW)) or (not np.isfinite(dE)) or (dW <= eps) or (dE <= eps) or ((dW + dE) <= eps):
            _nan_debug_dump("u_theta bad spacing", r, dW, dE, {})
            self.coeffs_t = EqCoeffs(np.nan, np.nan, np.nan, np.nan)
            return
        
        rinv  = 1.0 / (r if r > eps else eps)
        rinv2 = rinv * rinv

        alphaW = - dE / (dW * (dW + dE))
        alphaE =   dW / (dE * (dW + dE))
        alphaP = (dE - dW) / (dW * dE)

        betaW =  2.0 / ((dE + dW) * dW)
        betaE =  2.0 / ((dE + dW) * dE)
        betaP = - (betaW + betaE)

        if not np.all(np.isfinite([alphaW, alphaP, alphaE, betaW, betaP, betaE])):
            _nan_debug_dump("u_theta operators non-finite", r, dW, dE,
                            dict(alphaW=alphaW, alphaP=alphaP, alphaE=alphaE,
                                 betaW=betaW, betaP=betaP, betaE=betaE))

        ur = 0.0 if self._u_r     is None else float(self._u_r)
        tW = 0.0 if W._u_theta    is None else float(W._u_theta)
        tP = 0.0 if self._u_theta is None else float(self._u_theta)
        tE = 0.0 if E._u_theta    is None else float(E._u_theta)

        aW_diff = nu * (betaW + rinv * alphaW)
        aE_diff = nu * (betaE + rinv * alphaE)
        aP_diff = nu * (betaP + rinv * alphaP)

        aW_conv =  rho * ur * alphaW
        aE_conv =  rho * ur * alphaE
        aP_conv =  rho * ur * alphaP
        aP_metric = rho * (ur * rinv)

        aW = aW_diff + aW_conv
        aE = aE_diff + aE_conv
        aP = (-(aW + aE)) + (aP_diff + aP_conv + aP_metric) - (nu * rinv2)

        if not np.all(np.isfinite([aW, aE, aP])):
            _nan_debug_dump("u_theta coeffs non-finite", r, dW, dE,
                            dict(aW=aW, aE=aE, aP=aP,
                                 aW_diff=aW_diff, aE_diff=aE_diff, aP_diff=aP_diff,
                                 aW_conv=aW_conv, aE_conv=aE_conv, aP_conv=aP_conv,
                                 aP_metric=aP_metric, ur=ur))
            
        self.coeffs_t = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=0.0)

    def assemble_coeffs_p(self, rho: float) -> None:
        W, E = self.inner, self.outer
        if W is None or E is None:
            self.coeffs_p = EqCoeffs(1.0, 0.0, 0.0, 0.0)
            return

        r  = float(self._r)
        dW = self._drW()
        dE = self._drE()
        eps = 1e-12

        if (not np.isfinite(dW)) or (not np.isfinite(dE)) or (dW <= eps) or (dE <= eps) or ((dW + dE) <= eps):
            _nan_debug_dump("p bad spacing", r, dW, dE, {})
            self.coeffs_p = EqCoeffs(np.nan, np.nan, np.nan, np.nan)
            return
        
        rinv  = 1.0 / (r if r > eps else eps)

        alphaW = - dE / (dW * (dW + dE))
        alphaE =   dW / (dE * (dW + dE))
        alphaP = (dE - dW) / (dW * dE)

        betaW =  2.0 / ((dE + dW) * dW)
        betaE =  2.0 / ((dE + dW) * dE)
        betaP = - (betaW + betaE)

        if not np.all(np.isfinite([alphaW, alphaP, alphaE, betaW, betaP, betaE])):
            _nan_debug_dump("p operators non-finite", r, dW, dE,
                            dict(alphaW=alphaW, alphaP=alphaP, alphaE=alphaE,
                                 betaW=betaW, betaP=betaP, betaE=betaE))

        aW = betaW + rinv * alphaW
        aE = betaE + rinv * alphaE
        aP = (-(aW + aE)) + (rinv * alphaP)

        uW = 0.0 if W._u_r is None else float(W._u_r)
        uP = 0.0 if self._u_r is None else float(self._u_r)
        uE = 0.0 if E._u_r is None else float(E._u_r)

        qW, qP, qE = (float(W._r) * uW), (r * uP), (float(E._r) * uE)
        d_q_dr_i = alphaW * qW + alphaP * qP + alphaE * qE
        Di = rinv * d_q_dr_i
        b = rho * Di

        if not np.all(np.isfinite([aW, aE, aP, b])):
            _nan_debug_dump("p coeffs non-finite", r, dW, dE,
                            dict(aW=aW, aE=aE, aP=aP, b=b,
                                 qW=qW, qP=qP, qE=qE, d_q_dr_i=d_q_dr_i, Di=Di))

        self.coeffs_p = EqCoeffs(aP=float(aP), aW=float(aW), aE=float(aE), b=float(b))

    # --- local residuals (diagnostics) ---
    def update_local_residuals(self) -> None:
        # u_r residual
        if self.inner and self.outer and self._u_r is not None:
            phiW = self.inner._u_r
            phiE = self.outer._u_r
            cr = self.coeffs_r
            lhs = cr.aP * self._u_r
            rhs = (cr.aW * (0.0 if phiW is None else phiW)
            + cr.aE * (0.0 if phiE is None else phiE) + cr.b)

            self.res_r.R = lhs - rhs
        else:
            self.res_r.R = 0.0

        # u_theta residual
        if self.inner and self.outer and self._u_theta is not None:
            phiW = self.inner._u_theta
            phiE = self.outer._u_theta
            ct = self.coeffs_t
            lhs = ct.aP * self._u_theta
            rhs = (ct.aW * (0.0 if phiW is None else phiW)
            + ct.aE * (0.0 if phiE is None else phiE) + ct.b)

            self.res_t.R = lhs - rhs
        else:
            self.res_t.R = 0.0

        # p residual
        if self.inner and self.outer and self._p is not None:
            phiW = self.inner._p
            phiE = self.outer._p
            cp = self.coeffs_p
            lhs = cp.aP * self._p

            rhs = (cp.aW * (0.0 if phiW is None else phiW)
            + cp.aE * (0.0 if phiE is None else phiE) + cp.b)
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