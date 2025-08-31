import numpy as np
import json
from pathlib import Path

from gpe_three_systems_experiment import (
    DuffingParams,
    duffing_dynamics,
    rk4_step,
    collect_data_a_pe,
    collect_data_g_pe,
    build_phi_matrix,
    poly_features_2d,
    edmdc_fit,
)


def fit_edmd_model(X, U, feature_fn):
    """Fit EDMDc model and return A, B, and Phi matrix."""
    Phi = build_phi_matrix(X[:-1], feature_fn)
    Phi_next = build_phi_matrix(X[1:], feature_fn)
    U_row = U.reshape(1, -1)
    A_phi, B_phi = edmdc_fit(Phi, Phi_next, U_row)
    return A_phi, B_phi, Phi


def covariance_condition_numbers(X, Phi):
    """Compute min eigenvalues and condition numbers of state and feature covariance."""
    X_c = X - X.mean(axis=0, keepdims=True)
    Sigma_x = (X_c.T @ X_c) / max(X_c.shape[0], 1)
    w_x = np.linalg.eigvalsh(Sigma_x)
    lam_x = float(w_x.min())
    cond_x = float(w_x.max() / max(lam_x, 1e-12))

    Phi_c = Phi - Phi.mean(axis=1, keepdims=True)
    # drop nearly constant features to avoid singular covariance from bias term
    variances = np.var(Phi_c, axis=1)
    Phi_c = Phi_c[variances > 1e-12, :]
    Sigma_phi = (Phi_c @ Phi_c.T) / max(Phi_c.shape[1], 1)
    w_phi = np.linalg.eigvalsh(Sigma_phi)
    lam_phi = float(w_phi.min())
    cond_phi = float(w_phi.max() / max(lam_phi, 1e-12))
    return lam_x, cond_x, lam_phi, cond_phi


def simulate_true(x0, U_seq, dt, dyn):
    x = x0.copy()
    xs = [x.copy()]
    for u in U_seq:
        x = rk4_step(x, u, dt, dyn)
        xs.append(x.copy())
    return np.array(xs)


def rollout_rmse(A_phi, B_phi, feature_fn, X_true, U_seq):
    x_pred = X_true[0].copy()
    X_pred = [x_pred.copy()]
    for k, u in enumerate(U_seq):
        phi = feature_fn(X_pred[-1])
        phi_next = A_phi @ phi + (B_phi.flatten() * u)
        x_pred = phi_next[: X_true.shape[1]]
        X_pred.append(x_pred.copy())
    X_pred = np.array(X_pred)
    return float(np.sqrt(np.mean((X_pred[1:] - X_true[1:]) ** 2)))


def compare_methods():
    dt = 0.01
    L_seg = 50
    N_segs = 200
    dims = 2
    params = DuffingParams()
    dyn = lambda x, u: duffing_dynamics(x, u, params)
    feature_fn = lambda x: poly_features_2d(x, degree=3)
    grid_sizes = (12, 24, 36)

    # A-PE baseline (random control)
    X_ape, U_ape = collect_data_a_pe(dyn, params, dt, L_seg, N_segs, dims)
    A_ape, B_ape, Phi_ape = fit_edmd_model(X_ape, U_ape, feature_fn)

    # G-PE with Poisson targets; gamma set high to use full budget
    X_gpe, U_gpe, _ = collect_data_g_pe(
        dyn=dyn,
        p=params,
        dt=dt,
        L_seg=L_seg,
        max_segments=N_segs,
        state_dim=dims,
        feature_fn=feature_fn,
        dims=dims,
        grid_sizes=grid_sizes,
        gamma=np.inf,
        rho0=6.0,
    )
    A_gpe, B_gpe, Phi_gpe = fit_edmd_model(X_gpe, U_gpe, feature_fn)

    # Metrics for both methods
    metrics = {}
    test_T = 8.0
    N_test = int(test_T / dt)
    U_test = params.u_max * 0.8 * np.sin(np.arange(N_test) * dt * 2.5) \
        + params.u_max * 0.2 * np.sin(np.arange(N_test) * dt * 11)
    x0_tests = [
        np.array([-1.5, 0.5]),
        np.array([1.0, 1.0]),
        np.array([0.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, -1.5]),
    ]

    for label, (X, U, A_phi, B_phi, Phi) in {
        "A-PE": (X_ape, U_ape, A_ape, B_ape, Phi_ape),
        "G-PE": (X_gpe, U_gpe, A_gpe, B_gpe, Phi_gpe),
    }.items():
        lam_x, cond_x, lam_phi, cond_phi = covariance_condition_numbers(X, Phi)
        rmses = []
        for x0 in x0_tests:
            X_true = simulate_true(x0, U_test, dt, dyn)
            rmses.append(rollout_rmse(A_phi, B_phi, feature_fn, X_true, U_test))
        metrics[label] = {
            "samples": int(len(U)),
            "lambda_min_Sigma_x": lam_x,
            "cond_Sigma_x": cond_x,
            "lambda_min_Sigma_phi": lam_phi,
            "cond_Sigma_phi": cond_phi,
            "rmse": float(np.mean(rmses)),
        }

    out_dir = Path("outputs/compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for label, m in metrics.items():
        print(f"\n=== {label} ===")
        print(f"samples: {m['samples']}")
        print(f"lambda_min(Sigma_x) = {m['lambda_min_Sigma_x']:.4f}")
        print(f"cond(Sigma_x) = {m['cond_Sigma_x']:.2e}")
        print(f"lambda_min(Sigma_phi) = {m['lambda_min_Sigma_phi']:.4e}")
        print(f"cond(Sigma_phi) = {m['cond_Sigma_phi']:.2e}")
        print(f"prediction RMSE = {m['rmse']:.4f}")
    print(f"\nMetrics saved to {out_dir / 'comparison_metrics.json'}")


if __name__ == "__main__":
    compare_methods()
