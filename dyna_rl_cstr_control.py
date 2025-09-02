# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import pinv
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 0. 辅助工具 (与原版相同)
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self, original_dim):
        self.mean = None
        self.std = None
        self.original_dim = original_dim

    def fit(self, data):
        original_data = data[:self.original_dim, :]
        self.mean = np.mean(original_data, axis=1, keepdims=True)
        self.std = np.std(original_data, axis=1, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x):
        return (x[:self.original_dim].reshape(-1, 1) - self.mean) / self.std

    def inverse_transform(self, x_norm):
        return x_norm.reshape(-1, 1) * self.std + self.mean


def lift_state(x_normalized, poly_order=2):
    x1, x2 = x_normalized.flatten()
    lifted = [1.0]
    for i in range(1, poly_order + 1):
        for j in range(i + 1):
            lifted.append((x1**(i - j)) * (x2**j))
    return np.array(lifted)


def lift_input(u):
    return np.array([u])


# ---------------------------------------------------------------------------
# 1. 环境 (Environment): CSTR 反应器
#    - 遵循标准的 RL 环境接口 (step, reset)
# ---------------------------------------------------------------------------
class CSTREnvironment:
    """代表真实世界的CSTR系统，作为RL环境。"""

    def __init__(self, sim_params, dt, initial_state, target_state, state_weights, integral_weights):
        self.params = sim_params
        self.dt = dt
        self.initial_state = np.copy(initial_state)
        self.target_state = np.copy(target_state)
        self.state = np.copy(self.initial_state)
        self.state_weights = state_weights
        self.integral_weights = integral_weights
        self.last_action = 300  # 用于计算控制变化成本

    def cstr_system_dynamics(self, t, x, u):
        Ca, T = x[0], x[1]
        Tc = u
        Ca = max(0, Ca)
        T = max(1.0, T)
        rA = self.params.k0 * np.exp(-self.params.EA_over_R / T) * Ca
        dCa_dt = self.params.q / self.params.V * (self.params.Caf - Ca) - rA
        dT_dt = (
            self.params.q / self.params.V * (self.params.Ti - T)
            + ((-self.params.deltaHr * rA) / (self.params.rho * self.params.C))
            + (self.params.UA * (Tc - T) / (self.params.rho * self.params.C * self.params.V))
        )
        return [dCa_dt, dT_dt]

    def reset(self):
        self.state = np.copy(self.initial_state)
        self.last_action = 300
        return self.state

    def step(self, action, integral_error):
        """执行一个动作，返回 (next_state, reward, done, info)"""
        sol = solve_ivp(
            lambda t, x: self.cstr_system_dynamics(t, x, action),
            [0, self.dt],
            self.state,
            method="RK45",
        )
        next_state = sol.y[:, -1]

        # 奖励是成本的负数
        reward = -self._calculate_cost(self.state, integral_error, action, self.last_action)

        self.state = next_state
        self.last_action = action

        # 对于连续控制，我们通常不设置 done=True
        done = False
        info = {}

        return next_state, reward, done, info

    def _calculate_cost(self, x, I, u, prev_u):
        state_cost = np.sum(self.state_weights * (x - self.target_state) ** 2)
        integral_cost = np.sum(self.integral_weights * I ** 2)
        control_cost = 0.2 * (u - 300) ** 2 + 5.0 * (u - prev_u) ** 2
        return state_cost + integral_cost + control_cost


# ---------------------------------------------------------------------------
# 2. 世界模型 (World Model): 智能体对环境的内部表示
# ---------------------------------------------------------------------------
class KoopmanWorldModel:
    """使用在线更新的Koopman算子作为智能体的世界模型。"""

    def __init__(self, normalizer, poly_order, control_dim, integral_dim, learning_rate=1e-6):
        self.normalizer = normalizer
        self.poly_order = poly_order
        self.learning_rate = learning_rate
        self.p_phi = 2  # 原始状态维度
        self.p_psi = lift_state(np.zeros((2, 1)), poly_order).shape[0] + integral_dim + control_dim

        # A, B 代表模型对世界动态的理解: w_next = A @ w_current + B @ z_current
        self.A = np.zeros((self.p_phi, self.p_phi))
        self.B = np.zeros((self.p_phi, self.p_psi))
        self.w_prev = np.zeros(self.p_phi)
        self.is_pre_trained = False

    def pretrain(self, x_trajs, u_trajs, w_trajs, dt, target_state):
        """使用离线数据对模型进行预训练（热启动）。"""
        print("正在对Koopman世界模型进行离线预训练...")

        # 1. 构建快照矩阵
        Theta_w, Upsilon_z, Theta_w_plus = [], [], []
        for x_traj, u_traj, w_traj in tqdm(zip(x_trajs, u_trajs, w_trajs), total=len(x_trajs)):
            integral_error = np.zeros(target_state.shape)
            for k in range(w_traj.shape[1] - 1):
                w_k, w_k_plus_1 = w_traj[:, k], w_traj[:, k + 1]
                x_k, u_k = x_traj[:, k], u_traj[k]

                z_k = self._get_augmented_lifted_state(x_k, u_k, integral_error)

                Theta_w.append(w_k)
                Upsilon_z.append(z_k)
                Theta_w_plus.append(w_k_plus_1)

                integral_error += (x_k - target_state) * dt

        Theta_w, Upsilon_z, Theta_w_plus = (
            np.array(Theta_w).T,
            np.array(Upsilon_z).T,
            np.array(Theta_w_plus).T,
        )

        # 2. 通过TEDMD计算初始A, B矩阵
        if Theta_w.shape[1] > 0:
            Psi = np.vstack([Theta_w, Upsilon_z])
            try:
                AB = Theta_w_plus @ pinv(Psi, rcond=1e-6)
                A_init, B_init = AB[:, : self.p_phi], AB[:, self.p_phi :]
                self.A, self.B = A_init, B_init
                self.is_pre_trained = True
                print("模型预训练完成，已设置初始 A 和 B 矩阵。")
            except np.linalg.LinAlgError:
                print("预训练期间发生线性代数错误，模型将从零初始化。")
        else:
            print("没有有效的离线数据进行预训练，模型将从零初始化。")

    def _get_augmented_lifted_state(self, x_current, u_current, I_current):
        x_norm = self.normalizer.transform(x_current)
        lifted_state = lift_state(x_norm, self.poly_order)
        lifted_input = lift_input(u_current)
        return np.concatenate([lifted_state, I_current, lifted_input])

    def predict(self, x_current, u_current, I_current, w_current):
        """(用于规划) 预测下一个状态残差 w_next = x_next - x_current。"""
        z_t = self._get_augmented_lifted_state(x_current, u_current, I_current)
        w_next_hat = self.A @ w_current + self.B @ z_t
        return w_next_hat

    def update(self, x_current, u_current, I_current, w_observed):
        """(模型学习) 使用真实的经验 (s,a,s') 更新模型参数 A 和 B。"""
        z_t = self._get_augmented_lifted_state(x_current, u_current, I_current)
        w_pred = self.A @ self.w_prev + self.B @ z_t
        error = w_pred - w_observed

        # 梯度下降更新
        grad_A = np.outer(error, self.w_prev)
        grad_B = np.outer(error, z_t)
        self.A -= self.learning_rate * grad_A
        self.B -= self.learning_rate * grad_B

        self.w_prev = w_observed


# ---------------------------------------------------------------------------
# 3. 智能体 (Agent): 决策者
# ---------------------------------------------------------------------------
class DynaAgent:
    """一个Dyna风格的智能体，它拥有一个世界模型，并使用它进行规划以选择行动。"""

    def __init__(self, world_model, target_state, dt, **kwargs):
        self.world_model = world_model
        self.target_state = target_state
        self.dt = dt

        # 规划器 (PI²) 的参数
        self.n_samples = kwargs.get("n_samples", 300)
        self.horizon = kwargs.get("horizon", 15)
        self.control_limits = kwargs.get("control_limits", (280, 320))
        self.noise_sigma = kwargs.get("noise_sigma", 0.75)
        self.lambda_ = kwargs.get("lambda_", 0.05)
        self.state_bounds = [(0.0, 2.0), (250.0, 500.0)]

        # 成本函数权重
        self.state_weights = kwargs.get("state_weights", np.array([5.0, 5.0]))
        self.integral_weights = kwargs.get("integral_weights", np.array([2.0, 0.5]))

        # 智能体内部状态
        self.integral_error = np.zeros(len(target_state))
        self.previous_controls = np.full(self.horizon, np.mean(self.control_limits))

    def reset(self):
        """重置智能体的内部状态。"""
        self.integral_error.fill(0)
        self.previous_controls.fill(np.mean(self.control_limits))
        self.world_model.w_prev.fill(0)  # 同时重置模型的记忆

    def act(self, current_state):
        """(规划+行动) 使用其世界模型进行规划，以选择最佳行动。"""
        nominal_controls = np.append(self.previous_controls[1:], self.previous_controls[-1])
        perturbed_controls = np.random.randn(self.n_samples, self.horizon) * self.noise_sigma
        control_samples = np.clip(nominal_controls + perturbed_controls, *self.control_limits)

        # 使用世界模型进行前向模拟并计算成本
        w_current_for_planning = self.world_model.w_prev
        costs = np.array([
            self._simulate_trajectory_cost(current_state, self.integral_error, w_current_for_planning, cs)
            for cs in control_samples
        ])

        valid = np.isfinite(costs)
        if not np.any(valid):
            best_action = nominal_controls[0]
        else:
            min_cost = np.min(costs[valid])
            exp_costs = np.exp(-(costs[valid] - min_cost) / self.lambda_)
            weights = exp_costs / (np.sum(exp_costs) + 1e-9)
            self.previous_controls = np.einsum("i,ij->j", weights, control_samples[valid])
            best_action = self.previous_controls[0]

        return best_action

    def observe_transition(self, next_state):
        """在与环境交互后，更新智能体的内部状态（如积分误差）。"""
        self.integral_error += (next_state - self.target_state) * self.dt

    def _planning_cost_function(self, x, I, u, prev_u):
        """智能体在规划时使用的内部成本函数。"""
        state_cost = np.sum(self.state_weights * (x - self.target_state) ** 2)
        integral_cost = np.sum(self.integral_weights * I ** 2)
        control_cost = 0.2 * (u - 300) ** 2 + 5.0 * (u - prev_u) ** 2
        return state_cost + integral_cost + control_cost

    def _planning_terminal_cost(self, x, I):
        return 100 * np.sum((x - self.target_state) ** 2) + 20 * np.sum(I ** 2)

    def _simulate_trajectory_cost(self, x_init, I_init, w_init, control_sequence):
        """在世界模型中模拟一条轨迹的成本。"""
        x, I, w = np.copy(x_init), np.copy(I_init), np.copy(w_init)
        total_cost = 0.0
        for i in range(self.horizon):
            w = self.world_model.predict(x, control_sequence[i], I, w)
            x += w  # 预测的下一个状态
            I += (x - self.target_state) * self.dt

            if not (
                self.state_bounds[0][0] <= x[0] <= self.state_bounds[0][1]
                and self.state_bounds[1][0] <= x[1] <= self.state_bounds[1][1]
            ):
                return np.inf

            prev_u = self.previous_controls[0] if i == 0 else control_sequence[i - 1]
            total_cost += self._planning_cost_function(x, I, control_sequence[i], prev_u)

        total_cost += self._planning_terminal_cost(x, I)
        return total_cost


# ---------------------------------------------------------------------------
# 4. 离线数据生成与可视化 (辅助函数)
# ---------------------------------------------------------------------------
# (核心修正) 使用平滑有界的随机输入生成数据
def generate_random_trajectories(n_trajectories, traj_length, params, x_bounds, dt):
    all_x, all_u = [], []
    t_span_per_traj = (0, (traj_length - 1) * dt)
    print("正在为离线训练生成数据（使用平滑随机输入）...")

    for _ in tqdm(range(n_trajectories), desc="生成轨迹"):
        x0 = [np.random.uniform(b[0], b[1]) for b in x_bounds]

        num_sin = np.random.randint(2, 5)
        amps = np.random.uniform(1, 5, num_sin)
        freqs = np.random.uniform(0.05, 0.2, num_sin)
        phases = np.random.uniform(0, 2 * np.pi, num_sin)
        bias = np.random.uniform(295, 305)

        def smooth_random_input(t_array):
            sine_waves = [amp * np.sin(freq * t_array + phase) for amp, freq, phase in zip(amps, freqs, phases)]
            signal = np.sum(np.array(sine_waves), axis=0)
            return bias + signal

        env_temp = CSTREnvironment(params, dt, x0, np.zeros(2), np.zeros(2), np.zeros(2))
        x_traj, u_traj = [env_temp.state], []
        t_eval = np.linspace(t_span_per_traj[0], t_span_per_traj[1], traj_length)
        u_values = smooth_random_input(t_eval)

        for k in range(traj_length - 1):
            next_state, _, _, _ = env_temp.step(u_values[k], np.zeros(2))
            if not np.all(np.isfinite(next_state)):
                x_traj = []
                break
            x_traj.append(next_state)
            u_traj.append(u_values[k])

        if x_traj:
            u_traj.append(u_values[-1])
            all_x.append(np.array(x_traj).T)
            all_u.append(np.array(u_traj))

    if not all_x:
        raise RuntimeError("未能生成任何有效轨迹，请检查仿真参数。")
    return all_x, all_u


def visualize_control_results(t_sim, states, controls, target_state, title_prefix=""):
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{title_prefix} 控制结果", fontsize=16)
    axs[0].plot(t_sim, states[0, :], "b-", label=r"实际 $C_a$")
    axs[0].axhline(target_state[0], color="b", linestyle="--", label=r"目标 $C_a$")
    axs[0].set_ylabel("浓度 (mol/L)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("状态轨迹", fontsize=14)
    axs[1].plot(t_sim, states[1, :], "r-", label="实际 T")
    axs[1].axhline(target_state[1], color="r", linestyle="--", label="目标 T")
    axs[1].set_ylabel("温度 (K)")
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(t_sim[:-1], controls, "g-", label="控制输入 $T_c$")
    axs[2].set_xlabel("时间 (s)")
    axs[2].set_ylabel("冷却剂温度 Tc (K)")
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_results.png", dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# 5. 主执行模块 (Dyna-RL 循环)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 1. 初始化 ---
    print("--- 1. 系统初始化 ---")

    class CSTRParams:
        q, V, rho, C, deltaHr, EA_over_R, k0, UA, Ti, Caf = (
            100,
            100,
            1000,
            0.239,
            -5e4,
            8750,
            7.2e10,
            5e4,
            350,
            1,
        )

    params = CSTRParams()
    poly_order, controller_dt, original_dim, integral_dim = 2, 0.5, 2, 2

    # 定义目标状态
    T_target = 320.0
    k_target = params.k0 * np.exp(-params.EA_over_R / T_target)
    Ca_target = (params.q / params.V * params.Caf) / (params.q / params.V + k_target)
    target_state = np.array([Ca_target, T_target])
    initial_state_test = np.array([0.5, 350.0])

    # --- 2. 离线数据生成与模型预训练 ---
    print("\n--- 2. 模型预训练阶段 ---")
    offline_x, offline_u = generate_random_trajectories(50, 100, params, [(0.7, 1.1), (300, 360)], controller_dt)

    normalizer = Normalizer(original_dim)
    normalizer.fit(np.hstack(offline_x))

    world_model = KoopmanWorldModel(normalizer, poly_order, 1, integral_dim, learning_rate=1e-6)

    offline_w = [np.diff(x_traj, axis=1) for x_traj in offline_x]
    world_model.pretrain(offline_x, offline_u, offline_w, controller_dt, target_state)

    # --- 3. 初始化环境和智能体 ---
    print("\n--- 3. 初始化RL环境和Dyna智能体 ---")
    agent_params = {
        "state_weights": np.array([5.0, 5.0]),
        "integral_weights": np.array([0.50, 2.5]),
    }

    env = CSTREnvironment(params, controller_dt, initial_state_test, target_state, **agent_params)
    agent = DynaAgent(world_model, target_state, controller_dt, **agent_params)

    # --- 4. Dyna-RL 主循环 ---
    print("\n--- 4. 开始Dyna-RL在线控制循环 ---")
    n_steps = 400
    current_state = env.reset()
    agent.reset()

    state_history = [current_state]
    control_history = []

    for i in tqdm(range(n_steps), desc="Dyna-RL 步骤"):
        action = agent.act(current_state)
        next_state, reward, done, info = env.step(action, agent.integral_error)

        w_observed = next_state - current_state
        agent.world_model.update(current_state, action, agent.integral_error, w_observed)

        agent.observe_transition(next_state)

        current_state = next_state
        state_history.append(current_state)
        control_history.append(action)

    # --- 5. 结果可视化 ---
    print("\n--- 5. 可视化控制结果 ---")
    state_history = np.array(state_history).T
    control_history = np.array(control_history)
    t_sim = np.arange(state_history.shape[1]) * controller_dt
    visualize_control_results(t_sim, state_history, control_history, target_state, "Dyna-RL风格的在线MPC (CSTR)")

