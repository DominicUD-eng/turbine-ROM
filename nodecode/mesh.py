# mesh.py
import numpy as np
from node import Node

class Mesh:
    def __init__(self, r_out: float, r_in: float) -> None:
        assert r_out > r_in > 0.0, "Require r_out > r_in > 0"
        self.r_main = float(r_out)
        self.r_in = float(r_in)
        # Populated by buildMesh()
        self.nodes: list[Node] = []
        self.rs: np.ndarray | None = None
        self.tags: list[str] | None = None

    def buildMesh(
        self,
        p0: float,
        v_r_out: float,
        v_t_out: float,
        n_inner: int,
        n_main: int,
        n_outer: int,
        r_outer: float | None = None,
    ) -> None:
        
        r_main = self.r_main
        r_inner = self.r_in

        if r_outer is None:
            r_outer = r_main * 1.25 # explicit default choice
        r_outer = float(r_outer)
        assert r_outer > r_main > r_inner > 0.0, "Require r_outer > r_main > r_inner > 0."

        def interior_even(a: float, b: float, n: int) -> np.ndarray:
            if n <= 0:
                return np.empty((0,), dtype=float)
            return np.linspace(a, b, n + 2, dtype=float)[1:-1]

        # Region radii (ascending per region), then reverse each to get global outer→inner
        rs_outer = interior_even(r_outer, r_main, n_outer)
        rs_main = interior_even(r_main, r_inner, n_main)
        rs_inner = interior_even(r_inner, self.r_in, n_inner)

        outer_nodes = [Node(radius=float(r)) for r in rs_outer[::-1]]
        for nd in outer_nodes:
            nd.region = "OUTER"

        main_nodes = [Node(radius=float(r)) for r in rs_main[::-1]]
        for nd in main_nodes:
            nd.region = "MAIN"

        inner_nodes = [Node(radius=float(r)) for r in rs_inner[::-1]]
        for nd in inner_nodes:
            nd.region = "INNER"

        # Combined list: index 0 is innermost
        self.nodes = [*inner_nodes, *main_nodes, *outer_nodes]
        if not self.nodes:
            raise ValueError("Mesh has zero interior nodes across all regions.")

        # Boundary conditions
        self.nodes[-1].u_r = float(v_r_out)
        self.nodes[-1].u_theta = float(v_t_out)
        self.nodes[0].p = float(p0)

        # --- ADD: ensure strictly increasing radii & remove duplicates ---
        self.nodes.sort(key=lambda nd: nd.r)

        tol = 1e-12
        dedup = [self.nodes[0]]
        for nd in self.nodes[1:]:
            if abs(nd.r - dedup[-1].r) > tol:
                dedup.append(nd)
        self.nodes = dedup

        rs = np.array([nd.r for nd in self.nodes], float)
        dr = np.diff(rs)
        if np.any(dr <= 0.0):
            raise ValueError(f"Non-increasing radii in mesh. min Δr = {dr.min():.3e}")
        # --- END ADD ---

        # Link neighbors inner↔outer (sorted by radius, robust to non-uniform Δr)
        self.link_radial_neighbors()

        # Mark BC flags for clarity
        self.nodes[0].is_inner_bc = True
        self.nodes[-1].is_outer_bc = True

        # Convenience arrays
        self.rs = np.array([nd.r for nd in self.nodes], dtype=float)
        self.tags = [nd.region for nd in self.nodes]

        self.nodes.sort(key=lambda nd: nd.r)

        tol = 1e-12
        dedup = [self.nodes[0]]
        for nd in self.nodes[1:]:
            if abs(nd.r - dedup[-1].r) > tol:
                dedup.append(nd)
        self.nodes = dedup

        rs = np.array([nd.r for nd in self.nodes], float)
        dr = np.diff(rs)
        if np.any(dr <= 0.0):
            raise ValueError(f"Non-increasing radii in mesh. min Δr = {dr.min():.3e}")

        self.link_radial_neighbors()
        print("[MESH DBG] min r =", min(nd.r for nd in self.nodes))
        print("[MESH DBG] inner flag =", self.nodes[0].is_inner_bc, "outer flag =", self.nodes[-1].is_outer_bc)
        print("[MESH DBG] any r<r_inner? ->", any(nd.r < self.r_in for nd in self.nodes))


    def link_radial_neighbors(self) -> None:
        """Sort nodes by r (inner→outer) and establish neighbor pointers."""
        if not hasattr(self, "nodes"):
            raise AttributeError("Mesh has no 'nodes' list.")
        self.nodes.sort(key=lambda nd: nd.r)
        for i, nd in enumerate(self.nodes):
            nd.set_neighbors(
                inner=self.nodes[i-1] if i > 0 else None,
                outer=self.nodes[i+1] if i < len(self.nodes)-1 else None,
            )