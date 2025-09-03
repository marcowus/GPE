import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# -------------------------------------------------------------
# Lorenz system with control u in R^3: dx/dt = f(x) + u
# -------------------------------------------------------------

def lorenz(x: np.ndarray, u: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    x1, x2, x3 = x
    u1, u2, u3 = u
    dx1 = sigma * (x2 - x1) + u1
    dx2 = x1 * (rho - x3) - x2 + u2
    dx3 = x1 * x2 - beta * x3 + u3
    return np.array([dx1, dx2, dx3])


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, dyn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * dt * k1, u)
    k3 = dyn(x + 0.5 * dt * k2, u)
    k4 = dyn(x + dt * k3, u)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

# -------------------------------------------------------------
# Feature map for EDMDc: quadratic monomials
# -------------------------------------------------------------

def poly_features_3d(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = x
    return np.array([
        x1, x2, x3,
        x1**2, x2**2, x3**2,
        x1*x2, x1*x3, x2*x3,
        1.0,
    ])


def build_phi_matrix(X: np.ndarray) -> np.ndarray:
    return np.stack([poly_features_3d(x) for x in X], axis=1)

# -------------------------------------------------------------
# EDMDc fit and predict
# -------------------------------------------------------------

def edmdc_fit(Phi: np.ndarray, Phi_next: np.ndarray, U: np.ndarray, reg: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.vstack([Phi, U])  # (q+m, N)
    G = Z @ Z.T
    K = Phi_next @ Z.T @ np.linalg.inv(G + reg * np.eye(G.shape[0]))
    q = Phi.shape[0]
    A_phi = K[:, :q]
    B_phi = K[:, q:]
    return A_phi, B_phi


def edmdc_predict(A_phi: np.ndarray, B_phi: np.ndarray, x0: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    x = x0.copy()
    traj = [x.copy()]
    for u in U:
        phi = poly_features_3d(x)
        phi_next = A_phi @ phi + B_phi @ u
        x = phi_next[:3]
        traj.append(x.copy())
    return np.array(traj)

# -------------------------------------------------------------
# Data collection methods
# -------------------------------------------------------------

def collect_full_dim_data(T: int, dt: float, u_max: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=3)
    X = [x.copy()]
    U = []
    for _ in range(T):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        u = u_max * direction
        x = rk4_step(x, u, dt, lorenz)
        X.append(x.copy()); U.append(u)
    return np.array(X), np.array(U)


def collect_hankel_data(T: int, dt: float, u_max: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5,5,size=3)
    X = [x.copy()]
    U = []
    u_seq = rng.uniform(-u_max, u_max, size=T)
    for u_scalar in u_seq:
        u_vec = np.array([u_scalar, 0.0, 0.0])
        x = rk4_step(x, u_vec, dt, lorenz)
        X.append(x.copy()); U.append(u_vec)
    # Hankel matrix to check rank
    hankel_order = 3
    H = np.lib.stride_tricks.sliding_window_view(u_seq, hankel_order)
    rank = np.linalg.matrix_rank(H)
    return np.array(X), np.array(U), rank

# -------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------

def main():
    dt = 0.01
    T = 4000
    u_max = 5.0

    # Full-dimension excitation
    X_fd, U_fd = collect_full_dim_data(T, dt, u_max, seed=1)
    Phi = build_phi_matrix(X_fd[:-1])
    Phi_next = build_phi_matrix(X_fd[1:])
    U_row = U_fd.T
    A_fd, B_fd = edmdc_fit(Phi, Phi_next, U_row)

    # Hankel-based scalar excitation
    X_hk, U_hk, rank = collect_hankel_data(T, dt, u_max, seed=2)
    Phi_hk = build_phi_matrix(X_hk[:-1])
    Phi_next_hk = build_phi_matrix(X_hk[1:])
    U_row_hk = U_hk.T
    A_hk, B_hk = edmdc_fit(Phi_hk, Phi_next_hk, U_row_hk)

    print(f"Hankel matrix rank: {rank}")

    # Evaluation on new trajectory with zero input
    eval_T = 1000
    x0 = np.array([1.0, 1.0, 1.0])
    U_zero = np.zeros((eval_T, 3))
    true_traj = [x0.copy()]
    x = x0.copy()
    for _ in range(eval_T):
        x = rk4_step(x, np.zeros(3), dt, lorenz)
        true_traj.append(x.copy())
    true_traj = np.array(true_traj)

    pred_fd = edmdc_predict(A_fd, B_fd, x0, U_zero, dt)
    pred_hk = edmdc_predict(A_hk, B_hk, x0, U_zero, dt)

    mse_fd = np.mean((true_traj - pred_fd)**2)
    mse_hk = np.mean((true_traj - pred_hk)**2)
    print(f"Full-dimension MSE: {mse_fd:.4f}")
    print(f"Hankel MSE: {mse_hk:.4f}")

    t = np.arange(eval_T+1) * dt
    fig, axs = plt.subplots(3,1, figsize=(8,9))
    labels = ['x', 'y', 'z']
    for i in range(3):
        axs[i].plot(t, true_traj[:,i], 'k', label='true')
        axs[i].plot(t, pred_fd[:,i], 'r--', label='full-dim')
        axs[i].plot(t, pred_hk[:,i], 'b:', label='hankel')
        axs[i].set_ylabel(labels[i])
    axs[2].set_xlabel('time')
    axs[0].legend()
    fig.tight_layout()
    plt.savefig('lorenz_edmdc_comparison.png')

if __name__ == "__main__":
    main()
