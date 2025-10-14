"""
nodecode package
================

This package provides tools for modeling radial inflow on a rotating disk using
a 1-D thin-film approximation.

Modules included:
- Node.py       : Node dataclass (r, dr, vr, vθ, p, ρ, μ)
- Mesh.py       : Mesh builder and geometry helpers
- MeshSolver.py : Solver for continuity, swirl decay, pressure integration, torque
- CaseRunner.py : Predefined fluid property cases
- NodeCodeDriver.py : Example driver script to run cases
"""

from nodecode.node import Node
from nodecode.mesh import Mesh


__all__ = ["Node", "Mesh", "SteadyCFDSolver", "DiskProperties"]