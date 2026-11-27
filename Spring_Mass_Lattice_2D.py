import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
import ipywidgets as widgets
from IPython.display import display

# Font: Soft

##########################################################################
                                                  
# ,--.   ,--.,--.                       ,--.        
#  \  `.'  / `--' ,---. ,--.,--. ,--,--.|  | ,---.  
#   \     /  ,--.(  .-' |  ||  |' ,-.  ||  |(  .-'  
#    \   /   |  |.-'  `)'  ''  '\ '-'  ||  |.-'  `) 
#     `-'    `--'`----'  `----'  `--`--'`--'`----'  
                                                
##########################################################################

def draw_spring(ax, p1, p2, coils=8, width=0.08):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    d = p2 - p1
    length = np.linalg.norm(d)

    if length == 0:
        return

    direction = d / length
    normal = np.array([-direction[1], direction[0]])

    t = np.linspace(0, 1, 2 * coils + 3)

    offset = np.zeros(len(t))
    offset[1:-1] = width * (-1.0)**np.arange(len(t)-2)

    points = (
        p1[None, :]
        + t[:, None] * d[None, :]
        + offset[:, None] * normal[None, :]
    )

    ax.plot(points[:, 0], points[:, 1],
            color="gray", linewidth=1.2)


def draw_lattice(Nx, Ny, a=1, boundary="fixed",
                 ux=None, uy=None):

    fig, ax = plt.subplots(figsize=(7, 7))

    # Equilibrium positions
    X, Y = np.meshgrid(
        np.arange(Nx) * a,
        np.arange(Ny) * a,
        indexing="ij"
    )

    # Displacements from equilibrium
    if ux is None:
        ux = np.zeros((Nx, Ny))

    if uy is None:
        uy = np.zeros((Nx, Ny))

    # Actual particle positions
    X = X + ux
    Y = Y + uy

    # Nearest-neighbor springs
    for i in range(Nx):
        for j in range(Ny):

            p = (X[i, j], Y[i, j])

            # Horizontal spring
            if i < Nx - 1:
                draw_spring(
                    ax, p,
                    (X[i+1, j], Y[i+1, j])
                )

            # Vertical spring
            if j < Ny - 1:
                draw_spring(
                    ax, p,
                    (X[i, j+1], Y[i, j+1])
                )

    # Fixed boundary springs
    if boundary == "fixed":

        for j in range(Ny):
            draw_spring(
                ax, (-a, j*a),
                (X[0, j], Y[0, j])
            )

            draw_spring(
                ax, (X[-1, j], Y[-1, j]),
                (Nx*a, j*a)
            )

        for i in range(Nx):
            draw_spring(
                ax, (i*a, -a),
                (X[i, 0], Y[i, 0])
            )

            draw_spring(
                ax, (X[i, -1], Y[i, -1]),
                (i*a, Ny*a)
            )

        # Rigid walls
        walls = plt.Rectangle(
            (-a, -a),
            (Nx+1)*a,
            (Ny+1)*a,
            fill=False,
            color="black",
            linewidth=3
        )

        ax.add_patch(walls)

    # Draw particles
    ax.scatter(
        X, Y, s=100,
        color="royalblue",
        edgecolors="black",
        zorder=5
    )

    ax.set_aspect("equal")
    ax.set_title(f"{Nx} × {Ny} Harmonic Lattice")
    ax.axis("off")
    plt.show()


def plot_potential_field(ux, uy, k):

    Ex = 0.5 * k * np.diff(ux, axis=0)**2
    Ey = 0.5 * k * np.diff(uy, axis=1)**2

    # Local potential energy
    V = np.zeros_like(ux, dtype=float)

    V[:-1, :] += Ex / 2
    V[1:, :]  += Ex / 2
    V[:, :-1] += Ey / 2
    V[:, 1:]  += Ey / 2

    X, Y = np.indices(ux.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, V, cmap="jet",
                           edgecolor="black")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Potential Energy")
    ax.set_title("Potential Energy Distribution")

    fig.colorbar(surf, ax=ax, shrink=0.6)

    plt.show()


def plot_eff_potential_field(k):
    a = 1  # Equilibrium lattice spacing

    x = np.linspace(-a/2, (3-0.5)*a, 100)
    y = np.linspace(-a/2, (3-0.5)*a, 100)

    X, Y = np.meshgrid(x, y)

    # Nearest equilibrium lattice site
    I = np.clip(np.round(X/a), 0, 3-1).astype(int)
    J = np.clip(np.round(Y/a), 0, 3-1).astype(int)

    # Displacements from equilibrium
    Ux = X - I*a
    Uy = Y - J*a

    # Number of connected springs in each direction
    nx = (I > 0).astype(int) + (I < 3-1).astype(int)
    ny = (J > 0).astype(int) + (J < 3-1).astype(int)

    # Effective potential with frozen neighbors
    V = 0.5*k*(nx*Ux**2 + ny*Uy**2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, V,
        cmap="jet",
        edgecolor="black",
        linewidth=0.1,
        rcount=150,
        ccount=150
    )

    # Particles at equilibrium
    Xp, Yp = np.meshgrid(np.arange(3)*a, np.arange(3)*a)

    ax.scatter(Xp, Yp, np.zeros_like(Xp), color="black", s=30)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("V(x,y)")
    ax.set_title("Effective Potential Landscape at Equilibrium for a 3 x 3 Lattice")

    fig.colorbar(surf, ax=ax, shrink=0.6)
    plt.show()


def draw_solution(sol, positions, a):

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(7, 7))

    def get_bonds(R):
        horizontal = np.stack(
            [R[:-1, :, :], R[1:, :, :]], axis=-2
        ).reshape(-1, 2, 2)

        vertical = np.stack(
            [R[:, :-1, :], R[:, 1:, :]], axis=-2
        ).reshape(-1, 2, 2)

        return np.concatenate([horizontal, vertical])


    R0 = positions[0]

    particles = ax.scatter(
        R0[..., 0], R0[..., 1],
        s=60, color="royalblue", zorder=3
    )

    springs = LineCollection(
        get_bonds(R0), colors="gray", linewidths=1.5
    )

    ax.add_collection(springs)

    ax.set_xlim(positions[..., 0].min()-a,
                positions[..., 0].max()+a)

    ax.set_ylim(positions[..., 1].min()-a,
                positions[..., 1].max()+a)

    ax.set_aspect("equal")
    ax.set_title(f"Time = {sol.t[0]:.2f}")

    def update(frame):
        Rt = positions[frame]

        particles.set_offsets(Rt.reshape(-1, 2))
        springs.set_segments(get_bonds(Rt))

        ax.set_title(f"Time = {sol.t[frame]:.2f}")
        fig.canvas.draw_idle()


    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(sol.t)-1,
        step=1,
        description="Frame:",
        continuous_update=True,
        layout=widgets.Layout(width="80%")
    )

    slider.observe(
        lambda change: update(change["new"]),
        names="value"
    )

    display(widgets.VBox([slider, fig.canvas]))