# node.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

@dataclass
class EqCoeffs:
    aP: float = 0.0
    aW: float = 0.0
    aE: float = 0.0
    b:  float = 0.0

@dataclass
class EqResiduals:
    R: float = 0.0   # local residual (e.g., aP*phi - (aW*phi_W + aE*phi_E + b))

class Node:
    def __init__(self, radius: float):

        # Velocity Components
        self._u_r: Optional[float] = None
        self._u_r_i: Optional[float] = None
        self._u_theta: Optional[float] = None
        self._u_theta_i: Optional[float] = None
        self._u_z: Optional[float] = None
        self._u_z_i: Optional[float] = None
        self._u_r_half: Optional[float] = None
        self._u_theta_half: Optional[float] = None

        # Pressure
        self._p: Optional[float] = None
        self._p_i: Optional[float] = None

        # Local Geometry
        self._r: float = float(radius)
        self._r_i: Optional[float] = None

        # Region label
        self._region: Optional[str] = None

        #Fluid Information
        self._dv: Optional[float] = None
        self._density: Optional[float] = None

        #Extra Values
        self.convergence = 1e-8

        # --- NEW: neighbor links (inner/west, outer/east) ---
        self.inner: Optional["Node"] = None  # r - Δr
        self.outer: Optional["Node"] = None  # r + Δr

        # --- NEW: per-equation coefficient storage ---
        self.coeffs_r: EqCoeffs = EqCoeffs()   # for u_r equation
        self.coeffs_t: EqCoeffs = EqCoeffs()   # for u_θ equation
        self.coeffs_p: EqCoeffs = EqCoeffs()   # for pressure equation (from continuity)

        # --- NEW: per-equation residuals ---
        self.res_r: EqResiduals = EqResiduals()
        self.res_t: EqResiduals = EqResiduals()
        self.res_p: EqResiduals = EqResiduals()

        # --- NEW: previous-iteration snapshots (for Δ max-norm & diagnostics) ---
        self._u_r_prev: Optional[float] = None
        self._u_theta_prev: Optional[float] = None
        self._p_prev: Optional[float] = None

        # --- NEW: optional flags for BCs (so apply_bcs() is simple & explicit) ---
        self.is_outer_bc: bool = False  # Dirichlet u_r, u_θ
        self.is_inner_bc: bool = False  # Dirichlet p

    
    # --- NEW: neighbor linking helper ---
    def set_neighbors(self, inner: Optional["Node"], outer: Optional["Node"]) -> None:
        self.inner = inner
        self.outer = outer
    
    # --- NEW: iteration-scope snapshots ---
    def snapshot_prev(self) -> None:
        self._u_r_prev = self._u_r
        self._u_theta_prev = self._u_theta
        self._p_prev = self._p

    # --- NEW: local Δ magnitudes for convergence tracking ---
    def local_deltas(self) -> Tuple[float, float, float]:
        du_r = 0.0 if self._u_r_prev is None or self._u_r is None else abs(self._u_r - self._u_r_prev)
        du_t = 0.0 if self._u_theta_prev is None or self._u_theta is None else abs(self._u_theta - self._u_theta_prev)
        dp   = 0.0 if self._p_prev is None or self._p is None else abs(self._p - self._p_prev)
        return du_r, du_t, dp

    # --- NEW: assembly of second-order coefficients (central in r) ---
    def assemble_coeffs_u_r(self, rho: float, nu: float, dr: float) -> None:
        r = self._r
        W = self.inner
        E = self.outer

        # Guard for boundaries
        if W is None or E is None:
            self.coeffs_r = EqCoeffs(1.0, 0.0, 0.0, 0.0)  # overwritten by BCs in apply_bcs()
            return

        # TODO: Replace the following placeholders with your PDF’s exact 2nd-order, axisymmetric discrete form.
        # Example skeleton (diffusion central + simple convection central; axisymmetric metric terms folded into aP/aW/aE):
        # NOTE: These are placeholders for structure only.
        aW = nu / (dr*dr)
        aE = nu / (dr*dr)
        aP = aW + aE  # + convective / metric contributions as needed
        b  = 0.0

        self.coeffs_r = EqCoeffs(aP=aP, aW=aW, aE=aE, b=b)

     # --- NEW: local residuals (for debugging/monitoring) ---
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

        # pressure residual
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

    # --- NEW: SOR updates. (GS if ω==1) ---
    def sor_update_u_r(self, omega_r: float) -> None:
        if self.inner is None or self.outer is None or self.is_outer_bc or self.is_inner_bc:
            return  # boundaries handled in apply_bcs()
        cr = self.coeffs_r
        if cr.aP == 0.0: return
        uW = 0.0 if self.inner._u_r is None else self.inner._u_r
        uE = 0.0 if self.outer._u_r is None else self.outer._u_r
        u_new = (cr.aW*uW + cr.aE*uE + cr.b) / cr.aP
        if self._u_r is None: self._u_r = 0.0
        self._u_r = (1.0 - omega_r)*self._u_r + omega_r*u_new

    def sor_update_u_theta(self, omega_t: float) -> None:
        if self.inner is None or self.outer is None or self.is_outer_bc or self.is_inner_bc:
            return
        ct = self.coeffs_t
        if ct.aP == 0.0: return
        tW = 0.0 if self.inner._u_theta is None else self.inner._u_theta
        tE = 0.0 if self.outer._u_theta is None else self.outer._u_theta
        t_new = (ct.aW*tW + ct.aE*tE + ct.b) / ct.aP
        if self._u_theta is None: self._u_theta = 0.0
        self._u_theta = (1.0 - omega_t)*self._u_theta + omega_t*t_new

    def sor_update_p(self, omega_p: float) -> None:
        if self.inner is None or self.outer is None:
            return
        cp = self.coeffs_p
        if cp.aP == 0.0 or self.is_inner_bc:  # inner p is Dirichlet
            return
        pW = 0.0 if self.inner._p is None else self.inner._p
        pE = 0.0 if self.outer._p is None else self.outer._p
        p_new = (cp.aW*pW + cp.aE*pE + cp.b) / cp.aP
        if self._p is None: self._p = 0.0
        self._p = (1.0 - omega_p)*self._p + omega_p*p_new

    #region G&S  
    # ==== G & S ====
    @property
    def density(self): return self._density
    @density.setter
    def density(self, den): self._density = den

    @property
    def dv(self): return self._dv
    @dv.setter
    def dv(self, dv): self._dv = dv

    @property
    def region(self) -> Optional[str]:
        return self._region
    @region.setter
    def region(self, value: str) -> None:
        self._region = value
        
    @property
    def u_r(self): return self._u_r
    @u_r.setter
    def u_r(self, v): self._u_r = v

    @property
    def u_r_i(self): return self._u_r_i
    @u_r_i.setter
    def u_r_i(self, v): self._u_r_i = v

    @property
    def u_theta(self): return self._u_theta
    @u_theta.setter
    def u_theta(self, v): self._u_theta = v

    @property
    def u_theta_i(self): return self._u_theta_i
    @u_theta_i.setter
    def u_theta_i(self, v): self._u_theta_i = v

    @property
    def u_z(self): return self._u_z
    @u_z.setter
    def u_z(self, v): self._u_z = v

    @property
    def u_z_i(self): return self._u_z_i
    @u_z_i.setter
    def u_z_i(self, v): self._u_z_i = v

    @property
    def u_r_half(self): return self._u_r_half
    @u_r_half.setter
    def u_r_half(self, v): self._u_r_half = v

    @property
    def u_theta_half(self): return self._u_theta_half
    @u_theta_half.setter
    def u_theta_half(self, v): self._u_theta_half = v

    @property
    def p(self): return self._p
    @p.setter
    def p(self, v): self._p = v

    @property
    def p_i(self): return self._p_i
    @p_i.setter
    def p_i(self, v): self._p_i = v

    @property
    def r(self): return self._r
    @r.setter
    def r(self, v): self._r = v

    @property
    def r_i(self): return self._r_i
    @r_i.setter
    def r_i(self, v): self._r_i = v

    #endregion