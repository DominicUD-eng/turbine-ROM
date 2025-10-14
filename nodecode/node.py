'''
Add Region String

'''
# node.py
from __future__ import annotations
from typing import Optional

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

    def set_previous(self, previous: "Node") -> None:
        self.r_i       = previous.r
        self.u_r_i     = previous.u_r
        self.u_theta_i = previous.u_theta
        self.u_z_i     = previous.u_z
        self.p_i       = previous.p



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