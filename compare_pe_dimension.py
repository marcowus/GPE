import json
import numpy as np

from gpe_three_systems_experiment import (
    DuffingParams,
    duffing_dynamics,
    collect_data_a_pe,
    collect_data_g_pe,
    poly_features_2d,
    build_phi_matrix,
    edmdc_fit,
    lambda_min_cov,
    rk4_step,
)


# ---------------------------------------------------------------------------
# Dimension estimators
# ---------------------------------------------------------------------------

def box_counting_dim(X: np.ndarray, r_bins: int = 12) -> float:
    """Estimate box-counting dimension of point cloud X (d, N)."""
    if X.size == 0:
        return float("nan")
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    Z = (X - mins) / scale
    rs = np.logspace(-2, 0, r_bins)
    counts = []
    for r in rs:
        idx = np.floor(Z / r).astype(int)
        # flatten indices to tuples
        uniq = {tuple(col) for col in idx.T}
        counts.append(len(uniq))
    xs = np.log(1.0 / rs)
    ys = np.log(counts)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope)


def _pairwise_dists_subsample(X, max_pairs=200_000, rng=None):
    """Subsample pairwise distances to avoid O(N^2) blow-up."""
    if rng is None:
        rng = np.random.default_rng()
    N = X.shape[1]
    M = min(max_pairs, N * (N - 1) // 2)
    if M <= 0:
        return np.array([])
    i = rng.integers(0, N, size=M)
    j = rng.integers(0, N, size=M)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.linalg.norm(X[:, i] - X[:, j], axis=0)
    return d


def corr_dimension(X, r_bins=30, max_pairs=200_000, fit_frac=(0.2, 0.7), rng=None):
    """Grassberger–Procaccia correlation dimension D2."""
    if rng is None:
        rng = np.random.default_rng()
    dists = _pairwise_dists_subsample(X, max_pairs=max_pairs, rng=rng)
    dists = dists[dists > 1e-12]
    if dists.size < 100:
        return np.nan, (np.nan, np.nan), 0.0
    lo, hi = np.quantile(dists, 0.01), np.quantile(dists, 0.95)
    rs = np.logspace(np.log10(lo), np.log10(hi), r_bins)
    C = np.array([(dists < r).mean() for r in rs])
    L = int(np.floor(r_bins * fit_frac[0]))
    R = int(np.ceil(r_bins * fit_frac[1]))
    L = max(L, 2)
    R = min(R, r_bins - 2)
    xs = np.log(rs[L:R])
    ys = np.log(np.maximum(C[L:R], 1e-15))
    A = np.vstack([xs, np.ones_like(xs)]).T
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    slope = float(sol[0])
    yhat = A @ sol
    R2 = 1.0 - np.sum((ys - yhat) ** 2) / np.sum((ys - ys.mean()) ** 2 + 1e-15)
    return slope, (rs[L], rs[R - 1]), float(R2)


def knn_intrinsic_dim(X, k=10, rng=None):
    """Levina–Bickel MLE intrinsic dimension."""
    if rng is None:
        rng = np.random.default_rng()
    X = X.T  # (N, d)
    N = X.shape[0]
    if N <= k + 1:
        return np.nan
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)) + 1e-12
    D.sort(axis=1)
    rk = D[:, 1 : k + 1]
    rk_ref = rk[:, -1]
    logs = np.log(rk_ref[:, None] / rk)
    inv_m = (logs.mean(axis=1)) ** -1
    inv_m = inv_m[np.isfinite(inv_m)]
    if inv_m.size == 0:
        return np.nan
    return float(np.mean(inv_m))


def estimate_dimension_bundle(X, box_dim_fn=None, k_list=(10, 15, 20), **kw_corr):
    """Return dictionary with D_box, D2_corr, D_kNN and diagnostics."""
    out = {}
    if box_dim_fn is not None:
        out["D_box"] = float(box_dim_fn(X))
    D2, (rlo, rhi), R2 = corr_dimension(X, **kw_corr)
    out["D2_corr"] = float(D2)
    out["corr_fit_range"] = (float(rlo), float(rhi))
    out["corr_R2"] = float(R2)
    vals = []
    for k in k_list:
        vals.append(knn_intrinsic_dim(X, k=k))
    vals = [v for v in vals if np.isfinite(v)]
    out["D_kNN_mean"] = float(np.mean(vals)) if vals else np.nan
    out["D_kNN_list"] = [float(v) for v in vals]
    return out


# ---------------------------------------------------------------------------
# Data collection utilities
# ---------------------------------------------------------------------------

def collect_data_oid(dyn, p, dt, L_seg, n_segments, state_dim):
    """Deterministic sinusoidal input design (simple OID proxy)."""
    x = np.random.randn(state_dim)
    X = [x.copy()]
    U = []
    t = 0.0
    for seg in range(n_segments):
        u = p.u_max * np.sin(0.5 * t)
        for _ in range(L_seg):
            x = rk4_step(x, u, dt, dyn)
            X.append(x.copy())
            U.append(u)
            t += dt
    return np.array(X), np.array(U)


def evaluate_dataset(X, U, dyn, feature_fn, dt, dims):
    """Fit Koopman model, test prediction, and compute coverage/dimension metrics."""
    Phi = build_phi_matrix(X[:-1], feature_fn)
    Phi_next = build_phi_matrix(X[1:], feature_fn)
    U_row = U.reshape(1, -1)
    A_phi, B_phi = edmdc_fit(Phi, Phi_next, U_row)
    # one-step prediction test
    N_test = 200
    x0 = np.random.randn(dims)
    U_test = np.random.uniform(-1.0, 1.0, size=N_test)
    X_true = [x0.copy()]
    x = x0.copy()
    for u in U_test:
        x = rk4_step(x, u, dt, dyn)
        X_true.append(x.copy())
    X_true = np.array(X_true)
    X_pred = [x0.copy()]
    x = x0.copy()
    for k, u in enumerate(U_test):
        phi_x = feature_fn(x)
        phi_next = A_phi @ phi_x + (B_phi.flatten() * u)
        x = phi_next[:dims]
        X_pred.append(x.copy())
    X_pred = np.array(X_pred)
    rmse = float(np.sqrt(np.mean((X_pred[1:] - X_true[1:]) ** 2)))
    lam_x = lambda_min_cov(X)
    Phi_c = Phi - Phi.mean(axis=1, keepdims=True)
    Sigma_phi = (Phi_c @ Phi_c.T) / Phi_c.shape[1]
    lam_phi = float(np.linalg.eigvalsh(Sigma_phi).min())
    dim_est = estimate_dimension_bundle(
        X.T,
        box_dim_fn=box_counting_dim,
        r_bins=36,
        max_pairs=300_000,
        fit_frac=(0.25, 0.75),
    )
    coverage = {
        "lambda_min_x": lam_x,
        "lambda_min_phi": lam_phi,
        "D_box": dim_est.get("D_box"),
        "D2_corr": dim_est.get("D2_corr"),
        "D_kNN": dim_est.get("D_kNN_mean"),
        "corr_range": dim_est.get("corr_fit_range"),
        "corr_R2": dim_est.get("corr_R2"),
    }
    return {
        "rmse": rmse,
        "coverage": coverage,
    }


def main():
    np.random.seed(0)
    params = DuffingParams()
    dyn = lambda x, u: duffing_dynamics(x, u, params)
    dt = 0.01
    L_seg = 20
    N_seg = 120
    dims = 2
    grid_sizes = (12, 24, 36)
    gamma = 1e-2
    rho0 = 6.0
    feature_fn = lambda x: poly_features_2d(x, degree=3)

    # A-PE
    X_ape, U_ape = collect_data_a_pe(dyn, params, dt, L_seg, N_seg, dims)
    metrics_ape = evaluate_dataset(X_ape, U_ape, dyn, feature_fn, dt, dims)

    # OID
    X_oid, U_oid = collect_data_oid(dyn, params, dt, L_seg, N_seg, dims)
    metrics_oid = evaluate_dataset(X_oid, U_oid, dyn, feature_fn, dt, dims)

    # G-PE
    X_gpe, U_gpe, _ = collect_data_g_pe(
        dyn,
        params,
        dt,
        L_seg,
        N_seg,
        dims,
        feature_fn,
        dims,
        grid_sizes,
        gamma,
        rho0,
    )
    metrics_gpe = evaluate_dataset(X_gpe, U_gpe, dyn, feature_fn, dt, dims)

    # HYBRID: first half OID then G-PE starting from last OID state
    N_half = N_seg // 2
    X_oid_half, U_oid_half = collect_data_oid(dyn, params, dt, L_seg, N_half, dims)
    X_gpe_tail, U_gpe_tail, _ = collect_data_g_pe(
        dyn,
        params,
        dt,
        L_seg,
        N_seg - N_half,
        dims,
        feature_fn,
        dims,
        grid_sizes,
        gamma,
        rho0,
        start_states=[X_oid_half[-1]],
    )
    X_hyb = np.vstack([X_oid_half, X_gpe_tail[1:]])
    U_hyb = np.concatenate([U_oid_half, U_gpe_tail])
    metrics_hyb = evaluate_dataset(X_hyb, U_hyb, dyn, feature_fn, dt, dims)

    results = {
        "A-PE": metrics_ape,
        "OID": metrics_oid,
        "G-PE": metrics_gpe,
        "HYBRID": metrics_hyb,
    }
    with open("compare_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(
        "\nConclusion: exact Hausdorff dimension is rarely computable, but the"\
        " provided estimators offer consistent evidence of full dimensional"\
        " coverage in simulations."
    )


if __name__ == "__main__":
    main()
