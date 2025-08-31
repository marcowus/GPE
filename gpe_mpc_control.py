import numpy as np
from typing import Callable, Tuple
from dataclasses import dataclass

from gpe_three_systems_experiment import (
    DuffingParams, LorenzParams, VanDerPolParams,
    duffing_dynamics, lorenz_dynamics, vdp_dynamics,
    rk4_step, collect_data_g_pe,
    build_phi_matrix, edmdc_fit,
    poly_features_2d, poly_features_3d
)
from gpe_visualization import plot_mpc_results


class PathIntegralMPC:
    def __init__(self, A_phi: np.ndarray, B_phi: np.ndarray,
                 feature_fn: Callable[[np.ndarray], np.ndarray], dims: int,
                 dt: float, horizon: int = 15, samples: int = 200,
                 u_limit: float = 1.0, noise_sigma: float = 0.3,
                 lam: float = 0.1):
        self.A = A_phi
        self.B = B_phi
        self.feature_fn = feature_fn
        self.dims = dims
        self.dt = dt
        self.horizon = horizon
        self.samples = samples
        self.u_limit = u_limit
        self.noise_sigma = noise_sigma
        self.lam = lam
        self.prev = np.zeros(horizon)

    def _koopman_step(self, x: np.ndarray, u: float) -> np.ndarray:
        phi = self.feature_fn(x)
        phi_next = self.A @ phi + (self.B.flatten() * u)
        return phi_next[: self.dims]

    def _simulate(self, x0: np.ndarray, u_seq: np.ndarray,
                  target: np.ndarray) -> float:
        x = x0.copy()
        cost = 0.0
        for i in range(self.horizon):
            u = np.clip(u_seq[i], -self.u_limit, self.u_limit)
            x = self._koopman_step(x, u)
            cost += np.sum((x - target) ** 2) + 0.01 * u ** 2
        cost += 10.0 * np.sum((x - target) ** 2)
        return cost

    def control(self, x: np.ndarray, target: np.ndarray) -> float:
        nominal = np.roll(self.prev, -1)
        pert = self.noise_sigma * np.random.randn(self.samples, self.horizon)
        U = np.clip(nominal + pert, -self.u_limit, self.u_limit)
        costs = np.array([self._simulate(x, U[i], target) for i in range(self.samples)])
        min_c = costs.min()
        w = np.exp(-(costs - min_c) / self.lam)
        w /= w.sum() + 1e-9
        self.prev = w @ U
        return float(self.prev[0])


def run_path_integral_tracking(dyn: Callable[[np.ndarray, float], np.ndarray],
                                A_phi: np.ndarray, B_phi: np.ndarray,
                                feature_fn: Callable[[np.ndarray], np.ndarray],
                                dims: int, dt: float,
                                target: np.ndarray,
                                n_steps: int = 100,
                                disturbance: float = 0.1) -> Tuple[float, float, np.ndarray, np.ndarray]:
    ctrl = PathIntegralMPC(A_phi, B_phi, feature_fn, dims, dt)
    x = np.random.randn(dims)
    X = [x.copy()]
    U = []
    for k in range(n_steps):
        u = ctrl.control(x, target)
        U.append(u)
        x = rk4_step(x, u, dt, dyn)
        if k == n_steps // 2:
            x += disturbance * np.random.randn(dims)
        X.append(x.copy())
    X = np.array(X)
    U = np.array(U)
    err = X - target
    rmse = float(np.sqrt(np.mean(err**2)))
    energy = float(np.sum(U ** 2))
    return rmse, energy, X, U


def identify_model(system: str, dt: float, segments: int = 200):
    if system == "duffing":
        params = DuffingParams()
        dyn = lambda x, u: duffing_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2
    elif system == "lorenz":
        params = LorenzParams()
        dyn = lambda x, u: lorenz_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_3d(x, degree=2)
        dims = 3
    else:
        params = VanDerPolParams()
        dyn = lambda x, u: vdp_dynamics(x, u, params)
        feature_fn = lambda x: poly_features_2d(x, degree=3)
        dims = 2

    X, U, _ = collect_data_g_pe(dyn, params, dt, 20, segments, dims,
                                feature_fn, dims, (12, 24, 36), 1e-2, 6.0)
    Phi = build_phi_matrix(X[:-1], feature_fn)
    Phi_plus = build_phi_matrix(X[1:], feature_fn)
    U_row = U.reshape(1, -1)
    A_phi, B_phi = edmdc_fit(Phi, Phi_plus, U_row)
    return dyn, A_phi, B_phi, feature_fn, dims


def main():
    dt = 0.01
    target0 = np.zeros(3)
    for sys in ["duffing", "lorenz", "vdp"]:
        dyn, A_phi, B_phi, feature_fn, dims = identify_model(sys, dt)
        target = target0[:dims]
        rmse, energy, X_traj, U_seq = run_path_integral_tracking(
            dyn, A_phi, B_phi, feature_fn, dims, dt, target
        )
        plot_mpc_results(sys, X_traj, U_seq, target, dt)
        print(f"{sys} tracking RMSE: {rmse:.3e}, control energy: {energy:.3e}")


if __name__ == "__main__":
    main()
