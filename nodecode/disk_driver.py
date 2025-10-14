import matplotlib.pyplot as plt 
import numpy as np

try:
    from mesh import Mesh
except Exception as e:
    raise RuntimeError(
        "Failed to import Mesh from mesh.py. Make sure your fixed mesh.py is in the same directory."
    ) from e
    
def plot_mesh_radii(mesh, title="Mesh Radii (outer → inner)"):
    """
    Plot nodes colored by region and vertical lines at r_in, r_main, r_outer.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Assign color per region
    region_colors = {
        "INNER": "tab:blue",
        "MAIN": "tab:orange",
        "OUTER": "tab:green"
    }

    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot each region separately
    for region_name, color in region_colors.items():
        xs = [nd.r for nd in mesh.nodes if getattr(nd, "region", None) == region_name]
        ys = np.zeros(len(xs))
        if xs:
            ax.scatter(xs, ys, color=color, label=f"{region_name} nodes", zorder=3)

    # Draw vertical lines at region boundaries
    ax.axvline(mesh.r_in,   color="tab:blue",   linestyle="--", label="Inner radius",  zorder=2)
    ax.axvline(mesh.r_main, color="tab:orange", linestyle="--", label="Main radius",   zorder=2)
    if getattr(mesh, "r_outer", None) is not None:
        ax.axvline(mesh.r_outer, color="tab:green", linestyle="--", label="Outer radius", zorder=2)

    # Labels and formatting
    ax.set_xlabel("Radius (m)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # ---- Example parameters (adjust to your case) ----
    # Geometry
    r_in   = 0.40   # inner radius (hole radius)
    r_main = 1.00   # main (donut outer) radius
    r_outer= None   # far-field radius for plotting/mesh

    # Fluid properties (not used in this plot, but needed for Mesh init)
    rho = 1000.0
    nu  = 1e-6

    # Mesh resolution (per-region interior nodes)
    n_inner = 1
    n_main  = 3
    n_outer = 1

    # Boundary conditions
    p0      = 101325.0  # pressure at innermost node
    v_r_out = 0.0       # radial velocity at outermost node
    v_t_out = 10.0      # tangential velocity at outermost node

    # ---- Build mesh ----
    mesh = Mesh(r_out=r_main, r_in=r_in, rho=rho, v=nu)

    # IMPORTANT: buildMesh takes r_outer as an argument; we also stash it on the mesh for plotting.
    mesh.buildMesh(p0, v_r_out, v_t_out, n_inner, n_main, n_outer, r_outer=r_outer)
    mesh.r_outer = r_outer  # keep for plotting

    # ---- Plot ----
    fig, ax = plot_mesh_radii(mesh, title="Mesh Radii (outer → inner)")
    fig.savefig("mesh_radii.png", dpi=150)
    plt.show()

    print("N nodes:", len(mesh.nodes))
    print("r (inner->outer):", mesh.rs)
    print("tags:", mesh.tags)
    print("BC outer (u_r, u_theta):", mesh.nodes[len(mesh.nodes)-1].u_r, mesh.nodes[len(mesh.nodes)-1].u_theta)
    print("BC inner (p):", mesh.nodes[0].p)