"""
Geometric Persistent Excitation (G‑PE) experiments for Koopman operator
identification on three nonlinear systems: Duffing oscillator, Lorenz‑63
system, and Van der Pol oscillator.  This script implements both a
baseline Algebraic PE (A‑PE) data collection scheme and the proposed
geometric PE scheme.  For each system we generate a training dataset
under A‑PE and G‑PE, fit a Koopman model via EDMDc, and report basic
metrics such as the minimum eigenvalue of the state and lifted
covariance matrices as well as one‑step prediction error.

Key components:

1. **System definitions**: Continuous‑time dynamics for Duffing,
   Lorenz‑63 and Van der Pol systems are provided with support for
   external control input.  Integration is performed using a fourth
   order Runge–Kutta scheme.

2. **Data collection**:
   - *A‑PE:* random control inputs are applied without regard to the
     geometric distribution of the resulting trajectory.  This is
     representative of standard persistent excitation through random
     probing.
   - *G‑PE:* a greedy algorithm drives the system along the direction
     corresponding to the minimum eigenvalue of the current state
     covariance.  A multi‑scale non‑clustering monitor ensures that
     the trajectory does not overly concentrate in any region of
     state space.  Data collection stops when both the minimum
     eigenvalue and non‑clustering criteria are satisfied.

3. **Koopman identification**: For each dataset we build a matrix of
   lifted observables using polynomial features and perform a least
   squares fit to determine the approximate Koopman matrix and control
   influence.  A single‑step prediction test reports the RMS error.

This file can be executed directly.  It prints summary metrics and
does not require any external dependencies beyond NumPy and
Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Any


# -----------------------------------------------------------------------------
# 1. System definitions
# -----------------------------------------------------------------------------

@dataclass
class DuffingParams:
    alpha: float = -1.0  # linear stiffness
    beta: float = 1.0    # cubic stiffness
    delta: float = 0.2   # damping
    b: float = 1.0       # input gain
    u_max: float = 3.0   # control bound


@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    b: float = 1.0
    u_max: float = 6.0


@dataclass
class VanDerPolParams:
    mu: float = 1.0
    b: float = 1.0
    u_max: float = 3.0


def duffing_dynamics(x: np.ndarray, u: float, p: DuffingParams) -> np.ndarray:
    """Continuous Duffing oscillator dynamics with control."""
    x1, x2 = x
    dx1 = x2
    dx2 = -p.delta * x2 - p.alpha * x1 - p.beta * (x1**3) + p.b * u
    return np.array([dx1, dx2], dtype=float)


def lorenz_dynamics(x: np.ndarray, u: float, p: LorenzParams) -> np.ndarray:
    """Continuous Lorenz‑63 dynamics with control on the second equation."""
    x1, x2, x3 = x
    dx1 = p.sigma * (x2 - x1)
    dx2 = x1 * (p.rho - x3) - x2 + p.b * u
    dx3 = x1 * x2 - p.beta * x3
    return np.array([dx1, dx2, dx3], dtype=float)


def vdp_dynamics(x: np.ndarray, u: float, p: VanDerPolParams) -> np.ndarray:
    """Continuous Van der Pol oscillator dynamics with control."""
    x1, x2 = x
    dx1 = x2
    dx2 = p.mu * (1 - x1**2) * x2 - x1 + p.b * u
    return np.array([dx1, dx2], dtype=float)


def rk4_step(x: np.ndarray, u: float, dt: float,
             dyn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """Perform one Runge–Kutta integration step for a given system."""
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * dt * k1, u)
    k3 = dyn(x + 0.5 * dt * k2, u)
    k4 = dyn(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -----------------------------------------------------------------------------
# 2. Utility functions for covariance and greedy control
# -----------------------------------------------------------------------------

def lambda_min_cov(X: np.ndarray) -> float:
    """Return the minimum eigenvalue of the centered covariance of X."""
    X = np.asarray(X)
    if X.shape[0] < 3:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    w = np.linalg.eigvalsh(C)
    return float(np.clip(w.min(), 0.0, None))


def min_eig_direction(X: np.ndarray) -> np.ndarray:
    """Compute the unit eigenvector of the smallest eigenvalue of Sigma_x."""
    X = np.asarray(X)
    if X.shape[0] < 3:
        v = np.random.randn(X.shape[1])
        return v / np.linalg.norm(v)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    vmin = V[:, np.argmin(w)]
    return vmin / (np.linalg.norm(vmin) + 1e-12)


def simulate_segment(x0: np.ndarray, u: float, L: int, dt: float,
                     dyn: Callable[[np.ndarray, float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a short segment of L steps and return final state and trajectory."""
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(L):
        x = rk4_step(x, u, dt, dyn)
        xs.append(x.copy())
    return x, np.array(xs)


def choose_u_for_direction(xk: np.ndarray, omega: np.ndarray, dt: float, L: int,
                           u_grid: np.ndarray, dyn: Callable[[np.ndarray, float], np.ndarray],
                           penalty: float = 0.15) -> float:
    """
    Given a current state xk and a target direction omega, choose a constant control
    u over a short horizon L dt to maximise the projected movement along omega
    minus a penalty for orthogonal movement.  Evaluate over a discrete grid u_grid.
    """
    best_u, best_score = 0.0, -np.inf
    for u in u_grid:
        x_end, _ = simulate_segment(xk, u, L, dt, dyn)
        d = x_end - xk
        proj = float(np.dot(omega, d))
        ortho = float(np.linalg.norm(d - proj * omega))
        score = proj - penalty * ortho
        if score > best_score:
            best_score = score
            best_u = u
    return best_u


# -----------------------------------------------------------------------------
# 3. Multi‑scale non‑clustering monitor
# -----------------------------------------------------------------------------

def multiscale_ratio(X: np.ndarray, dims: int, grid_sizes: Tuple[int, ...],
                     rho0: float, window: int = None) -> Dict[str, Any]:
    """
    Compute multi‑scale non‑clustering ratios for 2D or 3D data.  For each grid
    size g, form a uniform grid and count the number of points per cell.  The
    ratio is max(count) divided by expected count N / (#cells).  A window can
    be used to consider only the most recent samples to avoid bias from long
    histories.
    Returns a dict with keys 'ratios' (list of ratios) and 'ok' (boolean
    whether all ratios <= rho0).
    """
    X = np.asarray(X)
    if window is not None and X.shape[0] > window:
        X = X[-window:]
    N = X.shape[0]
    if N < 10:
        return {"ratios": [np.inf] * len(grid_sizes), "ok": False}
    # Compute bounding box
    mins = X.min(axis=0) - 1e-9
    maxs = X.max(axis=0) + 1e-9
    ratios = []
    ok = True
    for g in grid_sizes:
        # Create equispaced bins per dimension
        bins = [np.linspace(mins[d], maxs[d], g + 1) for d in range(dims)]
        # Use histogramdd for 3D, histogram2d for 2D
        if dims == 2:
            H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
        else:
            H, _ = np.histogramdd(X, bins=bins)
        max_count = H.max()
        expected = max(N / (g ** dims), 1e-12)
        ratio = float(max_count / expected)
        ratios.append(ratio)
        if ratio > rho0:
            ok = False
    return {"ratios": ratios, "ok": ok}


# -----------------------------------------------------------------------------
# 4. Feature construction for EDMDc
# -----------------------------------------------------------------------------

def poly_features_2d(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial features for 2‑D input up to the specified degree."""
    x1, x2 = x
    features = [x1, x2]
    if degree >= 2:
        features += [x1**2, x1 * x2, x2**2]
    if degree >= 3:
        features += [x1**3, x1**2 * x2, x1 * x2**2, x2**3]
    features.append(1.0)
    return np.array(features)


def poly_features_3d(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial features for 3‑D input up to quadratic order."""
    x1, x2, x3 = x
    features = [x1, x2, x3]
    # quadratic terms
    features += [x1**2, x2**2, x3**2, x1 * x2, x1 * x3, x2 * x3]
    features.append(1.0)
    return np.array(features)


def build_phi_matrix(X: np.ndarray, feature_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Build a matrix of lifted observations from a sequence of states."""
    return np.stack([feature_fn(x) for x in X], axis=1)


def edmdc_fit(Phi: np.ndarray, Phi_next: np.ndarray, U_row: np.ndarray, reg: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Dynamic Mode Decomposition with control.  Solve for matrices A_phi
    and B_phi such that Phi_next ≈ A_phi * Phi + B_phi * U.
    """
    Z = np.vstack([Phi, U_row])  # shape (q+1, N)
    G = Z @ Z.T
    G_reg = G + reg * np.eye(G.shape[0])
    K = Phi_next @ Z.T @ np.linalg.inv(G_reg)
    A_phi = K[:, :Phi.shape[0]]
    B_phi = K[:, Phi.shape[0]:]
    return A_phi, B_phi


def edmdc_predict_one_step(A_phi: np.ndarray, B_phi: np.ndarray,
                           x: np.ndarray, u: float,
                           feature_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Predict next state via Koopman approximation given current state and input."""
    phi = feature_fn(x)
    phi_next = A_phi @ phi + (B_phi.flatten() * u)
    # decode back using the linear part (first dims equal to state dims)
    return phi_next[: x.shape[0]]


# -----------------------------------------------------------------------------
# 5. Simple predictive controller in lifted space
# -----------------------------------------------------------------------------

def _prediction_matrices(
    A_phi: np.ndarray, B_phi: np.ndarray, C: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
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
            G[dims * i : dims * (i + 1), j] = (
                C @ A_power_j @ B_phi
            ).flatten()
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
) -> Dict[str, float]:
    """Run a simple MPC-like controller on the Koopman model.

    The controller optimises a quadratic cost over a short horizon using
    the identified lifted-state model.  The first state component of the
    reference trajectory is a sinusoid while other components are zero.
    ``noise_seq`` is added to the true system dynamics to assess robustness.

    Returns a dictionary with overall tracking RMSE, control energy and the
    RMSE after a disturbance (robustness metric).
    """
    n_steps = ref_traj.shape[0] - 1
    n_phi = A_phi.shape[0]
    C = np.zeros((dims, n_phi))
    C[:, :dims] = np.eye(dims)
    F, G = _prediction_matrices(A_phi, B_phi, C, horizon)

    x = ref_traj[0].copy()
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
        # simulate true dynamics and add noise
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
    return {"rmse": rmse, "energy": energy, "robust": robust_rmse}


# -----------------------------------------------------------------------------
# 6. Experiment driver
# -----------------------------------------------------------------------------

def collect_data_a_pe(
    dyn: Callable[[np.ndarray, float], np.ndarray],
    p: Any,
    dt: float,
    L_seg: int,
    n_segments: int,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect data with random control (A‑PE).  Returns arrays X and U."""
    x = np.random.randn(state_dim)  # random initial state
    X = [x.copy()]
    U = []
    for _ in range(n_segments):
        u = np.random.uniform(-p.u_max, p.u_max)
        for _ in range(L_seg):
            x = rk4_step(x, u, dt, dyn)
            X.append(x.copy()); U.append(u)
    return np.array(X), np.array(U)


def collect_data_g_pe(
    dyn: Callable[[np.ndarray, float], np.ndarray],
    p: Any,
    dt: float,
    L_seg: int,
    max_segments: int,
    state_dim: int,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    dims: int,
    grid_sizes: Tuple[int, ...],
    gamma: float,
    rho0: float,
    cov_window: int = None,
    ratio_window: int = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Collect data using geometric PE: greedily excite along weakest state direction
    while enforcing multi‑scale non‑clustering.  Stop when lambda_min exceeds
    gamma and ratios <= rho0 or when max_segments reached.
    Returns X, U and a log dict (including history of lambdas and ratios).
    """
    # start from random state
    x = np.random.randn(state_dim)
    X = [x.copy()]
    U = []
    lam_hist = []
    ratio_hist: List[List[float]] = []
    u_grid = np.linspace(-p.u_max, p.u_max, 21)
    for seg in range(1, max_segments + 1):
        # compute cov window; if provided, use recent window
        X_arr = np.array(X)
        if cov_window is not None and X_arr.shape[0] > cov_window:
            X_window = X_arr[-cov_window:]
        else:
            X_window = X_arr
        # smallest eigen direction
        omega = min_eig_direction(X_window)
        # choose u
        best_u = choose_u_for_direction(x, omega, dt, L_seg, u_grid, dyn)
        # simulate segment
        seg_pts = []
        for _ in range(L_seg):
            x = rk4_step(x, best_u, dt, dyn)
            X.append(x.copy()); U.append(best_u)
            seg_pts.append(x.copy())
        # update lam and ratio
        lam_val = lambda_min_cov(np.array(X_window))
        lam_hist.append(lam_val)
        ratio_res = multiscale_ratio(np.array(X_window), dims, grid_sizes, rho0, ratio_window)
        ratio_hist.append(ratio_res["ratios"])
        # stopping check
        if lam_val >= gamma and ratio_res["ok"]:
            break
    log = {"lam": lam_hist, "ratios": ratio_hist}
    return np.array(X), np.array(U), log


def run_system_experiment(
    system_name: str,
    dt: float = 0.01,
    L_seg: int = 20,
    N_warm: int = 20,
    N_max: int = 200,
    gamma: float = 1e-2,
    rho0: float = 6.0,
    cov_window: int = 2000,
    ratio_window: int = 2000,
) -> None:
    """
    Run an experiment on the specified system comparing A‑PE vs G‑PE.  Prints
    metrics and returns nothing.  This function encapsulates one full
    round for a single system.
    """
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
    # ---------------- A‑PE data collection (random inputs) ----------------
    X_ape, U_ape = collect_data_a_pe(dyn, params, dt, L_seg, N_max, dims)
    # build Phi and fit
    Phi_ape = build_phi_matrix(X_ape[:-1], feature_fn)
    Phi_next_ape = build_phi_matrix(X_ape[1:], feature_fn)
    U_row_ape = U_ape.reshape(1, -1)
    A_phi_ape, B_phi_ape = edmdc_fit(Phi_ape, Phi_next_ape, U_row_ape)
    # one‑step test for A‑PE
    x0_test = np.random.randn(dims)
    # generate test input as random sequence
    T_test = 4.0
    N_test = int(T_test / dt)
    U_test = np.random.uniform(-params.u_max, params.u_max, size=N_test)
    X_true = [x0_test.copy()]
    X_pred_ape = [x0_test.copy()]
    x_true = x0_test.copy()
    x_pred = x0_test.copy()
    for k in range(N_test):
        # true step
        x_true = rk4_step(x_true, U_test[k], dt, dyn)
        X_true.append(x_true.copy())
        # predict with A‑PE model using teacher forcing
        phi_x = feature_fn(X_pred_ape[-1])
        phi_next = A_phi_ape @ phi_x + (B_phi_ape.flatten() * U_test[k])
        x_pred = phi_next[:dims]
        X_pred_ape.append(x_pred.copy())
    X_true = np.array(X_true)
    X_pred_ape = np.array(X_pred_ape)
    rmse_ape = np.sqrt(np.mean((X_pred_ape[1:] - X_true[1:])**2))
    lam_x_ape = lambda_min_cov(X_ape)
    # measure lifted lam_x for APE
    Phi_c_ape = Phi_ape - Phi_ape.mean(axis=1, keepdims=True)
    Sigma_phi_ape = (Phi_c_ape @ Phi_c_ape.T) / Phi_c_ape.shape[1]
    lam_phi_ape = float(np.linalg.eigvalsh(Sigma_phi_ape).min())
    # ---------------- G‑PE data collection (greedy) ----------------
    X_gpe, U_gpe, log_gpe = collect_data_g_pe(
        dyn=dyn, p=params, dt=dt, L_seg=L_seg, max_segments=N_max, state_dim=dims,
        feature_fn=feature_fn, dims=dims, grid_sizes=grid_sizes,
        gamma=gamma, rho0=rho0, cov_window=cov_window, ratio_window=ratio_window
    )
    Phi_gpe = build_phi_matrix(X_gpe[:-1], feature_fn)
    Phi_next_gpe = build_phi_matrix(X_gpe[1:], feature_fn)
    U_row_gpe = U_gpe.reshape(1, -1)
    A_phi_gpe, B_phi_gpe = edmdc_fit(Phi_gpe, Phi_next_gpe, U_row_gpe)
    # one‑step test for G‑PE model
    X_pred_gpe = [x0_test.copy()]
    x_pred_g = x0_test.copy()
    for k in range(N_test):
        phi_x = feature_fn(X_pred_gpe[-1])
        phi_next = A_phi_gpe @ phi_x + (B_phi_gpe.flatten() * U_test[k])
        x_pred_g = phi_next[:dims]
        X_pred_gpe.append(x_pred_g.copy())
    X_pred_gpe = np.array(X_pred_gpe)
    rmse_gpe = np.sqrt(np.mean((X_pred_gpe[1:] - X_true[1:])**2))
    lam_x_gpe = lambda_min_cov(X_gpe)
    Phi_c_gpe = Phi_gpe - Phi_gpe.mean(axis=1, keepdims=True)
    Sigma_phi_gpe = (Phi_c_gpe @ Phi_c_gpe.T) / Phi_c_gpe.shape[1]
    lam_phi_gpe = float(np.linalg.eigvalsh(Sigma_phi_gpe).min())

    # ---------------- Tracking control comparison ----------------
    n_steps_ctrl = 100
    t_ref = np.arange(n_steps_ctrl + 1) * dt
    ref_traj = np.zeros((n_steps_ctrl + 1, dims))
    ref_traj[:, 0] = np.sin(0.5 * t_ref)
    noise_seq = 0.01 * np.random.randn(n_steps_ctrl, dims)
    disturb_step = n_steps_ctrl // 2
    disturb = 0.1 * np.random.randn(dims)
    metrics_ape = run_tracking_controller(
        dyn, A_phi_ape, B_phi_ape, feature_fn, dims, dt,
        ref_traj, noise_seq, horizon=10, disturb_step=disturb_step,
        disturb=disturb,
    )
    metrics_gpe = run_tracking_controller(
        dyn, A_phi_gpe, B_phi_gpe, feature_fn, dims, dt,
        ref_traj, noise_seq, horizon=10, disturb_step=disturb_step,
        disturb=disturb,
    )

    # print summary
    print(f"\n===== {system_name.upper()} SYSTEM =====")
    print("A‑PE segments (random):", len(U_ape))
    print("G‑PE segments:", len(U_gpe))
    print(f"RMSE A‑PE: {rmse_ape:.3e}, RMSE G‑PE: {rmse_gpe:.3e}")
    print(f"λ_min(Σ_x) A‑PE: {lam_x_ape:.3e}, G‑PE: {lam_x_gpe:.3e}")
    print(f"λ_min(Σ_φ) A‑PE: {lam_phi_ape:.3e}, G‑PE: {lam_phi_gpe:.3e}")
    print(
        "Tracking RMSE A‑PE: {0:.3e}, G‑PE: {1:.3e}".format(
            metrics_ape["rmse"], metrics_gpe["rmse"]
        )
    )
    print(
        "Control energy A‑PE: {0:.3e}, G‑PE: {1:.3e}".format(
            metrics_ape["energy"], metrics_gpe["energy"]
        )
    )
    print(
        "Post‑disturb RMSE A‑PE: {0:.3e}, G‑PE: {1:.3e}".format(
            metrics_ape["robust"], metrics_gpe["robust"]
        )
    )
    # optionally plot trajectories or log histories


if __name__ == "__main__":
    # Run experiments for all three systems
    for sys in ["duffing", "lorenz", "vdp"]:
        run_system_experiment(
            system_name=sys,
            dt=0.01,
            L_seg=20,
            N_warm=20,
            N_max=200,
            gamma=1e-2,
            rho0=6.0,
            cov_window=2000,
            ratio_window=2000,
        )