import numpy as np
from typing import Callable
from scipy.linalg import solve_discrete_are

from gpe_three_systems_experiment import (
    DuffingParams, LorenzParams, VanDerPolParams,
    duffing_dynamics, lorenz_dynamics, vdp_dynamics,
    rk4_step, collect_data_a_pe, collect_data_g_pe,
    build_phi_matrix, edmdc_fit,
    poly_features_2d, poly_features_3d
)

# -----------------------------------------------------------------------------
# LQR helper
# -----------------------------------------------------------------------------

def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete-time LQR via solution to the Riccati equation."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K


def simulate_closed_loop(
    dyn: Callable[[np.ndarray, float], np.ndarray],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    u_max: float,
    dt: float,
    steps: int,
) -> tuple:
    """Simulate Koopman-LQR closed loop."""
    x = x0.copy()
    xs = [x.copy()]
    us = []
    for _ in range(steps):
        phi = feature_fn(x)
        u = float(-(K @ phi))
        u = float(np.clip(u, -u_max, u_max))
        x = rk4_step(x, u, dt, dyn)
        xs.append(x.copy())
        us.append(u)
    return np.array(xs), np.array(us)


def build_koopman(dyn, params, feature_fn, dims, dt=0.01, L_seg=20, N_max=200, gamma=1e-2, rho0=6.0):
    """Collect A-PE and G-PE data and fit Koopman models."""
    # data A-PE
    X_ape, U_ape = collect_data_a_pe(dyn, params, dt, L_seg, N_max, dims)
    Phi_ape = build_phi_matrix(X_ape[:-1], feature_fn)
    Phi_next_ape = build_phi_matrix(X_ape[1:], feature_fn)
    U_row_ape = U_ape.reshape(1, -1)
    A_phi_ape, B_phi_ape = edmdc_fit(Phi_ape, Phi_next_ape, U_row_ape)
    # data G-PE
    grid_sizes = (12, 24, 36) if dims == 2 else (8, 12, 16)
    X_gpe, U_gpe, _ = collect_data_g_pe(
        dyn=dyn, p=params, dt=dt, L_seg=L_seg, max_segments=N_max, state_dim=dims,
        feature_fn=feature_fn, dims=dims, grid_sizes=grid_sizes,
        gamma=gamma, rho0=rho0, cov_window=2000, ratio_window=2000)
    Phi_gpe = build_phi_matrix(X_gpe[:-1], feature_fn)
    Phi_next_gpe = build_phi_matrix(X_gpe[1:], feature_fn)
    U_row_gpe = U_gpe.reshape(1, -1)
    A_phi_gpe, B_phi_gpe = edmdc_fit(Phi_gpe, Phi_next_gpe, U_row_gpe)
    return (A_phi_ape, B_phi_ape), (A_phi_gpe, B_phi_gpe)


def lqr_for_system(system_name: str):
    dt = 0.01
    L_seg = 20
    N_max = 200
    gamma = 1e-2
    rho0 = 6.0
    if system_name == 'duffing':
        params = DuffingParams()
        dyn = lambda x, u: duffing_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2
    elif system_name == 'lorenz':
        params = LorenzParams()
        dyn = lambda x, u: lorenz_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_3d(x, degree=2)
        dims = 3
    elif system_name == 'vdp':
        params = VanDerPolParams()
        dyn = lambda x, u: vdp_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2
    else:
        raise ValueError('unknown system')
    (A_ape, B_ape), (A_gpe, B_gpe) = build_koopman(dyn, params, feature_fn, dims, dt, L_seg, N_max, gamma, rho0)
    C = np.zeros((dims, A_ape.shape[0]))
    C[:, :dims] = np.eye(dims)
    Q = np.eye(dims)
    R = np.array([[1.0]])
    Q_lift = C.T @ Q @ C
    K_ape = dlqr(A_ape, B_ape, Q_lift, R)
    K_gpe = dlqr(A_gpe, B_gpe, Q_lift, R)
    x0 = np.random.randn(dims)
    xs_ape, us_ape = simulate_closed_loop(dyn, feature_fn, A_ape, B_ape, K_ape, x0, params.u_max, dt, 100)
    xs_gpe, us_gpe = simulate_closed_loop(dyn, feature_fn, A_gpe, B_gpe, K_gpe, x0, params.u_max, dt, 100)
    rmse_ape = np.sqrt(np.mean(xs_ape**2))
    rmse_gpe = np.sqrt(np.mean(xs_gpe**2))
    energy_ape = np.sum(us_ape**2)
    energy_gpe = np.sum(us_gpe**2)
    print(f"\n{system_name.upper()} Koopman-LQR")
    print(f"RMSE A-PE: {rmse_ape:.3f}, G-PE: {rmse_gpe:.3f}")
    print(f"Energy A-PE: {energy_ape:.3f}, G-PE: {energy_gpe:.3f}")


if __name__ == '__main__':
    for sys in ['duffing', 'lorenz', 'vdp']:
        lqr_for_system(sys)
