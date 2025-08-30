
# Koopman G-PE on Duffing with control (standalone script)
# (Generated from notebook execution)
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DuffingParams:
    alpha: float = -1.0
    beta: float = 1.0
    delta: float = 0.2
    b: float = 1.0
    u_max: float = 3.0

def duffing_dynamics(x, u, p: DuffingParams):
    x1, x2 = x
    dx1 = x2
    dx2 = -p.delta * x2 - p.alpha * x1 - p.beta * x1**3 + p.b * u
    return np.array([dx1, dx2])

def rk4_step(x, u, dt, p: DuffingParams):
    k1 = duffing_dynamics(x, u, p)
    k2 = duffing_dynamics(x + 0.5*dt*k1, u, p)
    k3 = duffing_dynamics(x + 0.5*dt*k2, u, p)
    k4 = duffing_dynamics(x + dt*k3, u, p)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def lambda_min_cov(X):
    X = np.asarray(X)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    if Xc.shape[0] < 3:
        return 0.0, np.eye(X.shape[1])
    C = (Xc.T @ Xc) / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    return float(w.min()), C

def min_eig_vec_x(X_seq):
    X = np.array(X_seq)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    vmin = V[:, np.argmin(w)]
    return vmin / (np.linalg.norm(vmin) + 1e-12)

def simulate_segment(x0, u, L, dt, p: DuffingParams):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(L):
        x = rk4_step(x, u, dt, p)
        xs.append(x.copy())
    return x, np.array(xs)

def choose_u_for_direction(xk, omega, dt, L, p: DuffingParams, u_grid=None, penalty=0.15):
    if u_grid is None:
        u_grid = np.linspace(-p.u_max, p.u_max, 25)
    best_u, best_score, best_end = 0.0, -1e18, None
    for u in u_grid:
        x_end, traj = simulate_segment(xk, u, L, dt, p)
        d = x_end - xk
        proj = float(np.dot(omega, d))
        ortho = float(np.linalg.norm(d - proj * omega))
        score = proj - penalty * ortho
        if score > best_score:
            best_score, best_u, best_end = score, u, x_end.copy()
    return best_u, best_end

def phi_poly3(x):
    x1, x2 = x
    return np.array([x1, x2, x1**2, x1*x2, x2**2, x1**3, x1**2*x2, x1*x2**2, x2**3, 1.0])

def build_phi_matrix(X):
    return np.stack([phi_poly3(x) for x in X], axis=1)

def edmdc_fit(Phi, Phi_next, U, reg=1e-6):
    Z = np.vstack([Phi, U])
    Y = Phi_next
    G = Z @ Z.T
    G_reg = G + reg * np.eye(G.shape[0])
    K = Y @ Z.T @ np.linalg.inv(G_reg)
    A_phi = K[:, :Phi.shape[0]]
    B_phi = K[:, Phi.shape[0]:]
    return A_phi, B_phi

def edmdc_predict_one_step(A_phi, B_phi, x, u):
    phi = phi_poly3(x)
    phi_next = A_phi @ phi + (B_phi.flatten() * u)
    return phi_next[:2], phi_next

def run():
    np.random.seed(123)
    p = DuffingParams()
    dt = 0.01
    L_seg = 20
    N_warm = 20
    N_max = 350
    gamma_phi = 0.010
    penalty = 0.15

    X_seq = []
    U_seq = []
    x = np.array([0.2, 0.0])
    t = 0.0
    X_seq.append(x.copy())

    for i in range(N_warm):
        u = np.random.uniform(-p.u_max, p.u_max)
        for _ in range(L_seg):
            x = rk4_step(x, u, dt, p)
            X_seq.append(x.copy())
            U_seq.append(u)

    def lambda_min_phi(X_seq):
        X_arr = np.array(X_seq)
        Phi = build_phi_matrix(X_arr)
        Phi_c = Phi - Phi.mean(axis=1, keepdims=True)
        Sigma_phi = (Phi_c @ Phi_c.T) / Phi_c.shape[1]
        w = np.linalg.eigvalsh(Sigma_phi)
        return float(w.min()), Sigma_phi

    reached = False
    phi_lambdas, x_lambdas = [], []

    for seg in range(N_max):
        omega = min_eig_vec_x(X_seq)
        omega = omega / (np.linalg.norm(omega) + 1e-12)

        u_opt, x_end = choose_u_for_direction(X_seq[-1], omega, dt, L_seg, p, penalty=penalty)
        u = u_opt
        for _ in range(L_seg):
            x = rk4_step(x, u, dt, p)
            X_seq.append(x.copy())
            U_seq.append(u)

        lam_phi, _ = lambda_min_phi(X_seq)
        lam_x, _ = lambda_min_cov(np.array(X_seq))
        phi_lambdas.append(lam_phi)
        x_lambdas.append(lam_x)
        if lam_phi >= gamma_phi:
            reached = True
            break

    X_arr = np.array(X_seq)
    U_arr = np.array(U_seq)
    Phi = build_phi_matrix(X_arr[:-1])
    Phi_next = build_phi_matrix(X_arr[1:])
    U_row = U_arr.reshape(1, -1)
    A_phi, B_phi = edmdc_fit(Phi, Phi_next, U_row, reg=1e-6)

    # test
    T_test = 6.0
    N_test = int(T_test / dt)
    U_test = np.random.uniform(-p.u_max, p.u_max, size=N_test)
    def simulate(x0, U_seq):
        xs = [x0.copy()]
        x = x0.copy()
        for u in U_seq:
            x = rk4_step(x, u, dt, p)
            xs.append(x.copy())
        return np.array(xs)

    x0_test = np.array([0.15, -0.05])
    X_true = simulate(x0_test, U_test)
    X_pred = np.zeros_like(X_true)
    X_pred[0] = x0_test.copy()
    for k in range(N_test):
        x_k = X_true[k]
        u_k = U_test[k]
        x_pred, _ = edmdc_predict_one_step(A_phi, B_phi, x_k, u_k)
        X_pred[k+1] = x_pred

    residuals = X_pred[1:] - X_true[1:]
    rmse = np.sqrt(np.mean(residuals**2, axis=0))

    print("RMSE x1, x2:", rmse)

if __name__ == "__main__":
    run()
