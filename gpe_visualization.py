import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_phase_portraits(system_name, X_ape, X_gpe, save_dir="figures"):
    """Plot phase portraits for A-PE and G-PE datasets."""
    save_dir = _ensure_dir(save_dir)
    dims = X_ape.shape[1]
    if dims == 2:
        plt.figure(figsize=(5, 4))
        plt.plot(X_ape[:, 0], X_ape[:, 1], ".", ms=1.0, alpha=0.5, label="A-PE")
        plt.plot(X_gpe[:, 0], X_gpe[:, 1], ".", ms=1.0, alpha=0.5, label="G-PE")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(f"{system_name} phase portrait")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
    elif dims == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_ape[:, 0], X_ape[:, 1], X_ape[:, 2], s=2, alpha=0.4, label="A-PE")
        ax.scatter(X_gpe[:, 0], X_gpe[:, 1], X_gpe[:, 2], s=2, alpha=0.4, label="G-PE")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax.set_title(f"{system_name} phase portrait")
        ax.legend()
    else:
        raise ValueError("Unsupported dimension for phase portrait")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{system_name}_phase_portrait.png"), dpi=160)
    plt.close()


def plot_lambda_history(system_name, lam_hist, gamma, save_dir="figures"):
    """Plot lambda_min evolution for G-PE."""
    save_dir = _ensure_dir(save_dir)
    seg = np.arange(1, len(lam_hist) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(seg, np.array(lam_hist) + 1e-12, "o-")
    plt.axhline(gamma, color="k", ls="--", label=r"$\gamma$")
    plt.yscale("log")
    plt.xlabel("Greedy segments")
    plt.ylabel(r"$\lambda_{\min}(\Sigma_x)$")
    plt.title(f"{system_name} lambda history")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{system_name}_lambda_history.png"), dpi=160)
    plt.close()


def plot_ratio_history(system_name, ratio_hist, rho0, save_dir="figures"):
    """Plot multi-scale non-clustering ratios."""
    save_dir = _ensure_dir(save_dir)
    ratio_arr = np.array(ratio_hist)
    seg = np.arange(1, ratio_arr.shape[0] + 1)
    plt.figure(figsize=(6, 4))
    for i in range(ratio_arr.shape[1]):
        plt.plot(seg, ratio_arr[:, i], "o-", label=f"grid {i}")
    plt.axhline(rho0, color="k", ls="--", label=r"$\rho_0$")
    plt.yscale("log")
    plt.xlabel("Greedy segments")
    plt.ylabel("max-count / expected")
    plt.title(f"{system_name} non-clustering ratios")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{system_name}_ratio_history.png"), dpi=160)
    plt.close()


def plot_tracking(system_name, ref_traj, traj_ape, traj_gpe, dt, save_dir="figures"):
    """Plot reference tracking for A-PE and G-PE models."""
    save_dir = _ensure_dir(save_dir)
    t = np.arange(ref_traj.shape[0]) * dt
    dims = ref_traj.shape[1]
    plt.figure(figsize=(8, 3 * dims))
    for d in range(dims):
        ax = plt.subplot(dims, 1, d + 1)
        ax.plot(t, ref_traj[:, d], "k--", lw=1.5, label="ref")
        ax.plot(t, traj_ape[:, d], label="A-PE")
        ax.plot(t, traj_gpe[:, d], label="G-PE")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"$x_{d+1}$")
        ax.grid(True)
        if d == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{system_name}_tracking.png"), dpi=160)
    plt.close()
