import numpy as np
from typing import Callable, Tuple, Dict

from gpe_three_systems_experiment import (
    DuffingParams,
    LorenzParams,
    VanDerPolParams,
    duffing_dynamics,
    lorenz_dynamics,
    vdp_dynamics,
    rk4_step,
    collect_data_a_pe,
    collect_data_g_pe,
    build_phi_matrix,
    edmdc_fit,
    lambda_min_cov,
    poly_features_2d,
    poly_features_3d,
)
import gpe_visualization as viz


# -----------------------------------------------------------------------------
# Prediction matrices and MPC-like controller
# -----------------------------------------------------------------------------

def _prediction_matrices(A_phi: np.ndarray, B_phi: np.ndarray, C: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute lifted-space prediction matrices for a finite horizon."""
    n_phi = A_phi.shape[0]
    dims = C.shape[0]
    F = np.zeros((dims * horizon, n_phi))
    G = np.zeros((dims * horizon, horizon))
    A_power = np.eye(n_phi)
    for i in range(horizon):
        A_power = A_power @ A_phi
        F[dims * i : dims * (i + 1), :] = C @ A_power
        for j in range(i + 1):
            A_power_j = np.linalg.matrix_power(A_phi, i - j)
            G[dims * i : dims * (i + 1), j] = (C @ A_power_j @ B_phi).flatten()
    return F, G


def run_tracking_controller(
    dyn: Callable[[np.ndarray, float], np.ndarray],
    A_phi: np.ndarray,
    B_phi: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    dims: int,
    dt: float,
    ref_traj: np.ndarray,
    noise_seq: np.ndarray,
    horizon: int = 10,
    reg_u: float = 1e-3,
    disturb_step: int = None,
    disturb: np.ndarray = None,
    return_traj: bool = False,
    x0: np.ndarray = None,
) -> Dict[str, float]:
    """Run a simple MPC-like controller on the Koopman model."""
    n_steps = ref_traj.shape[0] - 1
    n_phi = A_phi.shape[0]
    C = np.zeros((dims, n_phi))
    C[:, :dims] = np.eye(dims)
    F, G = _prediction_matrices(A_phi, B_phi, C, horizon)

    if x0 is None:
        x = ref_traj[0].copy()
    else:
        x = x0.copy()
    X_hist = [x.copy()]
    U_hist = []
    for k in range(n_steps):
        phi0 = feature_fn(x)
        r_seg = ref_traj[k + 1 : k + horizon + 1]
        if r_seg.shape[0] < horizon:
            pad = np.repeat(r_seg[-1:, :], horizon - r_seg.shape[0], axis=0)
            r_seg = np.vstack([r_seg, pad])
        r_stack = r_seg.reshape(-1)
        x_pred0 = F @ phi0
        H = G.T @ G + reg_u * np.eye(horizon)
        b = G.T @ (r_stack - x_pred0)
        U_seq = np.linalg.solve(H, b)
        u = float(U_seq[0])
        x = rk4_step(x, u, dt, dyn)
        x += noise_seq[k]
        if disturb_step is not None and k == disturb_step:
            x += disturb
        X_hist.append(x.copy())
        U_hist.append(u)

    X_hist = np.array(X_hist)
    ref_used = ref_traj[: X_hist.shape[0]]
    err = X_hist - ref_used
    rmse = float(np.sqrt(np.mean(err**2)))
    energy = float(np.sum(np.array(U_hist) ** 2))
    if disturb_step is not None:
        err_post = err[disturb_step + 1 :]
        robust_rmse = float(np.sqrt(np.mean(err_post**2)))
    else:
        robust_rmse = rmse
    out = {"rmse": rmse, "energy": energy, "robust": robust_rmse}
    if return_traj:
        out["traj"] = X_hist
    return out


def run_pic_controller(
    dyn: Callable[[np.ndarray, float], np.ndarray],
    A_phi: np.ndarray,
    B_phi: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    dims: int,
    dt: float,
    ref_traj: np.ndarray,
    noise_seq: np.ndarray,
    horizon: int = 15,
    n_samples: int = 200,
    lamb: float = 1.0,
    u_max: float = 3.0,
    disturb_step: int = None,
    disturb: np.ndarray = None,
    return_traj: bool = False,
    x0: np.ndarray = None,
    sigma: float = 0.5,
) -> Dict[str, float]:
    """Run a path-integral MPC controller on the Koopman model."""
    n_steps = ref_traj.shape[0] - 1
    x = ref_traj[0].copy() if x0 is None else x0.copy()
    X_hist = [x.copy()]
    U_hist = []
    nominal = np.zeros(horizon)
    for k in range(n_steps):
        phi0 = feature_fn(x)
        noise = sigma * np.random.randn(n_samples, horizon)
        U_samples = np.clip(nominal + noise, -u_max, u_max)
        costs = np.zeros(n_samples)
        for i in range(n_samples):
            phi = phi0.copy()
            cost = 0.0
            for j in range(horizon):
                u = U_samples[i, j]
                phi = A_phi @ phi + (B_phi.flatten() * u)
                x_pred = phi[:dims]
                r = ref_traj[min(k + j + 1, ref_traj.shape[0] - 1)]
                cost += np.sum((x_pred - r) ** 2) + 0.01 * u**2
            costs[i] = cost
        w = np.exp(-costs / lamb)
        w /= np.sum(w) + 1e-12
        nominal = w @ U_samples
        u = float(nominal[0])
        nominal = np.append(nominal[1:], nominal[-1])
        x = rk4_step(x, u, dt, dyn)
        x += noise_seq[k]
        if disturb_step is not None and k == disturb_step:
            x += disturb
        X_hist.append(x.copy())
        U_hist.append(u)
    X_hist = np.array(X_hist)
    ref_used = ref_traj[: X_hist.shape[0]]
    err = X_hist - ref_used
    rmse = float(np.sqrt(np.mean(err**2)))
    energy = float(np.sum(np.array(U_hist) ** 2))
    if disturb_step is not None:
        err_post = err[disturb_step + 1 :]
        robust_rmse = float(np.sqrt(np.mean(err_post**2)))
    else:
        robust_rmse = rmse
    out = {"rmse": rmse, "energy": energy, "robust": robust_rmse}
    if return_traj:
        out["traj"] = X_hist
    return out


# -----------------------------------------------------------------------------
# Tracking experiment
# -----------------------------------------------------------------------------

def run_tracking_experiment(
    system_name: str,
    dt: float = 0.01,
    L_seg: int = 20,
    N_max: int = 200,
    gamma: float = 1e-2,
    rho0: float = 6.0,
    cov_window: int = 2000,
    ratio_window: int = 2000,
) -> None:
    """Collect data, identify models, and run tracking control for a system."""
    if system_name == "duffing":
        params = DuffingParams()
        dyn = lambda x, u: duffing_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2
        grid_sizes = (12, 24, 36)
    elif system_name == "lorenz":
        params = LorenzParams()
        dyn = lambda x, u: lorenz_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_3d(x, degree=2)
        dims = 3
        grid_sizes = (8, 12, 16)
    elif system_name == "vdp":
        params = VanDerPolParams()
        dyn = lambda x, u: vdp_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2
        grid_sizes = (12, 24, 36)
    else:
        raise ValueError(f"Unknown system: {system_name}")

    # A-PE data
    X_ape, U_ape = collect_data_a_pe(dyn, params, dt, L_seg, N_max, dims)
    Phi_ape = build_phi_matrix(X_ape[:-1], feature_fn)
    Phi_next_ape = build_phi_matrix(X_ape[1:], feature_fn)
    U_row_ape = U_ape.reshape(1, -1)
    A_phi_ape, B_phi_ape = edmdc_fit(Phi_ape, Phi_next_ape, U_row_ape)

    # G-PE data
    X_gpe, U_gpe, log_gpe = collect_data_g_pe(
        dyn=dyn,
        p=params,
        dt=dt,
        L_seg=L_seg,
        max_segments=N_max,
        state_dim=dims,
        feature_fn=feature_fn,
        dims=dims,
        grid_sizes=grid_sizes,
        gamma=gamma,
        rho0=rho0,
        cov_window=cov_window,
        ratio_window=ratio_window,
    )
    Phi_gpe = build_phi_matrix(X_gpe[:-1], feature_fn)
    Phi_next_gpe = build_phi_matrix(X_gpe[1:], feature_fn)
    U_row_gpe = U_gpe.reshape(1, -1)
    A_phi_gpe, B_phi_gpe = edmdc_fit(Phi_gpe, Phi_next_gpe, U_row_gpe)

    # Metrics
    lam_x_ape = lambda_min_cov(X_ape)
    lam_x_gpe = lambda_min_cov(X_gpe)
    Phi_c_ape = Phi_ape - Phi_ape.mean(axis=1, keepdims=True)
    Sigma_phi_ape = (Phi_c_ape @ Phi_c_ape.T) / Phi_c_ape.shape[1]
    lam_phi_ape = float(np.linalg.eigvalsh(Sigma_phi_ape).min())
    Phi_c_gpe = Phi_gpe - Phi_gpe.mean(axis=1, keepdims=True)
    Sigma_phi_gpe = (Phi_c_gpe @ Phi_c_gpe.T) / Phi_c_gpe.shape[1]
    lam_phi_gpe = float(np.linalg.eigvalsh(Sigma_phi_gpe).min())

    # Tracking controller: drive from random x0 to origin
    n_steps_ctrl = 100
    ref_traj = np.zeros((n_steps_ctrl + 1, dims))
    x0 = np.random.randn(dims)
    noise_seq = 0.01 * np.random.randn(n_steps_ctrl, dims)
    disturb_step = n_steps_ctrl // 2
    disturb = 0.1 * np.random.randn(dims)
    metrics_ape = run_pic_controller(
        dyn,
        A_phi_ape,
        B_phi_ape,
        feature_fn,
        dims,
        dt,
        ref_traj,
        noise_seq,
        horizon=15,
        n_samples=200,
        u_max=params.u_max,
        disturb_step=disturb_step,
        disturb=disturb,
        return_traj=True,
        x0=x0,
    )
    metrics_gpe = run_pic_controller(
        dyn,
        A_phi_gpe,
        B_phi_gpe,
        feature_fn,
        dims,
        dt,
        ref_traj,
        noise_seq,
        horizon=15,
        n_samples=200,
        u_max=params.u_max,
        disturb_step=disturb_step,
        disturb=disturb,
        return_traj=True,
        x0=x0,
    )

    print(f"\n===== {system_name.upper()} SYSTEM =====")
    print("A-PE segments (random):", len(U_ape))
    print("G-PE segments:", len(U_gpe))
    print(f"λ_min(Σ_x) A-PE: {lam_x_ape:.3e}, G-PE: {lam_x_gpe:.3e}")
    print(f"λ_min(Σ_φ) A-PE: {lam_phi_ape:.3e}, G-PE: {lam_phi_gpe:.3e}")
    print("Tracking RMSE A-PE: {0:.3e}, G-PE: {1:.3e}".format(metrics_ape["rmse"], metrics_gpe["rmse"]))
    print("Control energy A-PE: {0:.3e}, G-PE: {1:.3e}".format(metrics_ape["energy"], metrics_gpe["energy"]))
    print(
        "Post-disturb RMSE A-PE: {0:.3e}, G-PE: {1:.3e}".format(
            metrics_ape["robust"], metrics_gpe["robust"]
        )
    )

    viz.plot_phase_portraits(system_name, X_ape, X_gpe)
    viz.plot_lambda_history(system_name, log_gpe["lam"], gamma)
    viz.plot_ratio_history(system_name, log_gpe["ratios"], rho0)
    viz.plot_tracking(system_name, ref_traj, metrics_ape["traj"], metrics_gpe["traj"], dt)


if __name__ == "__main__":
    for sys in ["duffing", "lorenz", "vdp"]:
        run_tracking_experiment(system_name=sys)
