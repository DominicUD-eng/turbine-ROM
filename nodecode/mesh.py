# Build Mesh

# "Think CSV file rows where each colomn is a node and each row is an interation"
'''
    1) Build Mesh
    2) Think CSV file rows where each colomn is a node and each row is an interation
        A) every iteration (Row)
            a) Call each node and update values with relazation factor (Solve will be in the node object)
            b) Check for convergence of every node, not just a single
            c) If all nodes have reached convergence, stop iteration. 
            d) Output values
    3) Explicit solver

    MEETING NOTES
    1) Every tangential is the same. So to solve inner -> outer you have two BC's. Presssure and U_theta
    2) Next Steps. Build Mesh nice, add region gaps, graph accordingly COnfirm approprietyl built mesh

'''

# mesh.py
import numpy as np
from node import Node  # or from .node import Node if packaged

class Mesh:
    def __init__(self, r_out: float, r_in: float, rho: float, v: float):
        assert r_out > r_in > 0.0, "Require r_out > r_in > 0"
        self.r_main = float(r_out)
        self.r_in   = float(r_in)
        self.dens   = float(rho)
        self.kin_vis = float(v)

    def buildMesh(
        self,
        p0: float,
        v_r_out: float,
        v_t_out: float,
        n_inner: int,
        n_main: int,
        n_outer: int,
        r_outer: float | None = None
    ):
        r_main  = self.r_main
        r_inner = self.r_in

        if r_outer is None:
            r_outer = r_main * 1.25  # explicit default choice

        r_outer = float(r_outer)

        assert r_outer > r_main > r_inner > 0.0, "Require r_outer > r_main > r_inner > 0."

        def interior_even(a: float, b: float, n: int) -> np.ndarray:
            """Return n evenly spaced interior points strictly between a and b."""
            if n <= 0:
                return np.empty((0,), dtype=float)
            return np.linspace(a, b, n + 2, dtype=float)[1:-1]

        # Region radii (ascending per region), then reverse to get global outer→inner
        rs_outer = interior_even(r_outer,  r_main, n_outer)  # just outside the donut
        rs_main  = interior_even(r_main, r_inner,  n_main)   # the donut
        rs_inner = interior_even(r_inner, 0.0, n_inner)  # inside the hole

        outer_nodes = [Node(radius=float(r)) for r in rs_outer[::-1]]  # descending
        for nd in outer_nodes: nd.region = "OUTER"

        main_nodes  = [Node(radius=float(r)) for r in rs_main[::-1]]
        for nd in main_nodes: nd.region = "MAIN"

        inner_nodes = [Node(radius=float(r)) for r in rs_inner[::-1]]
        for nd in inner_nodes: nd.region = "INNER"

        # Combined list: index 0 is Innnermost
        self.nodes: list[Node] = [*inner_nodes, *main_nodes, *outer_nodes]

        if not self.nodes:
            raise ValueError("Mesh has zero interior nodes across all regions.")

        # Boundary conditions
        self.nodes[len(self.nodes)-1].u_r = float(v_r_out)
        self.nodes[len(self.nodes)-1].u_theta = float(v_t_out)
        self.nodes[0].p = float(p0)

        # Link previous (outer→inner)
        for i in range(1, len(self.nodes)):
            self.nodes[i].set_previous(self.nodes[i - 1])

        for ind_n in self.nodes:
            ind_n.density = self.dens
            ind_n.dv = self.kin_vis

        # Optionally keep convenience arrays
        self.rs   = np.array([nd.r for nd in self.nodes], dtype=float)
        self.tags = [nd.region for nd in self.nodes]

    # --- ADD to your Mesh class (non-breaking helper) ---
    def link_radial_neighbors(self) -> None:
        """Assumes nodes are uniformly spaced in r and sorted ascending by radius (inner -> outer)."""
        if not hasattr(self, "nodes"): 
            raise AttributeError("Mesh has no 'nodes' list.")
        # Sort defensively by radius (stable if already sorted)
        self.nodes.sort(key=lambda nd: nd.r)
        for i, nd in enumerate(self.nodes):
            nd.set_neighbors(
                inner=self.nodes[i-1] if i > 0 else None,
                outer=self.nodes[i+1] if i < len(self.nodes)-1 else None
            )


'''
Step 1) Build Mesh -> Create Node for each radius
Step 2) Solve outmost node with initial conditions 
        a) might have to be innermost.
step 3) Set node-1 conditions in next node node -> solve node -> repeat for following node

hard code con in node, pass con back to mesh to check all for convergence, iff all then good
Three residuals (convergence) one for each equation


'''