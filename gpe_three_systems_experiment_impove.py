import numpy as np
import matplotlib.pyplot as plt

from gpe_three_systems_experiment import (
    DuffingParams,
    duffing_dynamics,
    collect_data_g_pe,
    poly_features_2d,
)


def run_demo():
    """Collect G-PE data for the Duffing system and visualise targets."""
    params = DuffingParams()
    dyn = lambda x, u: duffing_dynamics(x, u, params)
    dt = 0.01
    L_seg = 20
    N_max = 120
    dims = 2
    grid_sizes = (12, 24, 36)
    gamma = 1e-2
    rho0 = 6.0

    X, U, log = collect_data_g_pe(
        dyn=dyn,
        p=params,
        dt=dt,
        L_seg=L_seg,
        max_segments=N_max,
        state_dim=dims,
        feature_fn=lambda x: poly_features_2d(x, degree=3),
        dims=dims,
        grid_sizes=grid_sizes,
        gamma=gamma,
        rho0=rho0,
    )

    targets = log["targets"]
    counts = np.asarray(log["counts"])

    plt.figure(figsize=(6, 6))
    plt.scatter(targets[:, 0], targets[:, 1], c=counts, cmap="viridis", marker="x", label="targets")
    plt.plot(X[:, 0], X[:, 1], "k-", label="trajectory")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.title("G-PE trajectory and Poisson targets")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_demo()
