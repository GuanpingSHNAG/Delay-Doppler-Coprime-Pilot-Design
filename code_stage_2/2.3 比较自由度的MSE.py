import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from functools import reduce
from scipy.linalg import inv
import time
import pandas as pd

# ==========================================
# 1. 核心检测与估计函数
# ==========================================
def get_system_gcd(pilot_locs, Nc, Ns):
    diff_vectors = set()
    n_p = len(pilot_locs)
    int_locs = [(int(round(p[0])), int(round(p[1]))) for p in pilot_locs]
    
    for i in range(n_p):
        for j in range(i + 1, n_p):
            u = int_locs[i][0] - int_locs[j][0]
            v = int_locs[i][1] - int_locs[j][1]
            diff_vectors.add((u, v))
            
    if not diff_vectors: return Nc * Ns
    
    terms = {Nc * Ns}
    for u, v in diff_vectors:
        terms.add(abs(v * Nc))
        terms.add(abs(u * Ns))
            
    for i, (u1, v1) in enumerate(diff_vectors):
        for u2, v2 in list(diff_vectors)[i+1:]:
            terms.add(abs(u1 * v2 - u2 * v1))
            
    return reduce(math.gcd, list(terms))


def toroidal_manhattan_dist(p1, p2, M, N):
    """环面曼哈顿距离（考虑周期性边界）"""
    m1, n1 = p1
    m2, n2 = p2
    dm = min(abs(m1 - m2), M - abs(m1 - m2))
    dn = min(abs(n1 - n2), N - abs(n1 - n2))
    return dm + dn

def is_non_adjacent(pilot_locs, M, N):
    """所有导频对的环面曼哈顿距离均 ≠ 1"""
    n = len(pilot_locs)
    for i in range(n):
        for j in range(i+1, n):
            if toroidal_manhattan_dist(pilot_locs[i], pilot_locs[j], M, N) == 1:
                return False
    return True



def find_optimal_configurations(Sigma_theta, M=8, N=8, n_pilots=4):
    """搜索 FIM-MAP 最佳排布和 2D Coprime 排布"""
    all_indices = np.arange(M * N)
    combos = np.array(list(itertools.combinations(all_indices, n_pilots)))
    m = combos // N
    n = combos % N
    
    mean_m = np.mean(m, axis=1, keepdims=True)
    mean_n = np.mean(n, axis=1, keepdims=True)
    var_m = np.mean(m**2, axis=1) - mean_m.squeeze()**2
    var_n = np.mean(n**2, axis=1) - mean_n.squeeze()**2
    cov_mn = np.mean(m * n, axis=1) - (mean_m.squeeze() * mean_n.squeeze())
    
    coeff = (4 * np.pi**2)
    df = 1.0 / M
    Ts = 1.0 / N
    
    J11 = coeff * n_pilots * var_m * (df**2)
    J22 = coeff * n_pilots * var_n * (Ts**2)
    J12 = -coeff * n_pilots * cov_mn * (df * Ts)
    
    S11, S12, S22 = Sigma_theta[0, 0], Sigma_theta[0, 1], Sigma_theta[1, 1]
    
    A11 = S11 * J11 + S12 * J12
    A12 = S11 * J12 + S12 * J22
    A21 = S12 * J11 + S22 * J12
    A22 = S12 * J12 + S22 * J22
    
    dets = (1.0 + A11) * (1.0 + A22) - A12 * A21
    sorted_indices = np.argsort(dets)[::-1]
    
    locs_2d_coprime = None
    locs_opt = None
    
    for idx in sorted_indices:
        curr_locs = list(zip(m[idx], n[idx]))
        if is_non_adjacent(curr_locs, M, N): 
            if locs_opt is None:
                    locs_opt = curr_locs
            if locs_2d_coprime is None and locs_opt:
                if get_system_gcd(curr_locs, M, N) == 1: 
                    locs_2d_coprime = curr_locs
            if (locs_2d_coprime is not None) and (locs_opt is not None):
                break
            
    return locs_opt, locs_2d_coprime

def generate_received_signal(tau, nu, h, X, sigma2, N1=8, N2=8):
    D = N1 * N2
    s = np.zeros(D, dtype=complex)
    TD = np.diag(np.exp(-1j * 2 * np.pi * np.arange(N1) * tau / N1))
    
    for l in range(N2):
        gp = np.exp(1j * 2 * np.pi * nu * l / N2)
        s[l * N1 : (l + 1) * N1] = gp * (TD @ X[:, l])
        
    noise = np.sqrt(sigma2 / 2.0) * (np.random.randn(D) + 1j * np.random.randn(D))
    return h * s + noise

def sample_discrete_channel(mu_theta, Sigma_theta, q=19):
    while True:
        theta_raw = np.random.multivariate_normal(mu_theta, Sigma_theta)
        if 0.0 <= theta_raw[0] <= 8.0 and 0.0 <= theta_raw[1] <= 8.0:
            idx_tau = int(np.round(theta_raw[0] * q / 8.0))
            idx_nu  = int(np.round(theta_raw[1] * q / 8.0))
            tau_true = np.clip(idx_tau, 0, q - 1) * (8.0 / q)
            nu_true  = np.clip(idx_nu, 0, q - 1) * (8.0 / q)
            return tau_true, nu_true

def perform_exhaustive_map_estimation(y, X, mu_theta, Sigma_inv, beta, sigma2, q=19, N1=8, N2=8):
    tau_grid = np.arange(q) * (8.0 / q)
    nu_grid  = np.arange(q) * (8.0 / q)
    
    Y = y.reshape((N2, N1)).T * np.conj(X)
    W_tau = np.exp(1j * 2 * np.pi * np.outer(np.arange(N1), tau_grid) / N1)
    W_nu  = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N2), nu_grid) / N2)
    
    s_H_y_grid = (Y.T @ W_tau).T @ W_nu
    likelihood = (beta / (sigma2 * (sigma2 + beta))) * (np.abs(s_H_y_grid)**2)
    
    d_tau = tau_grid - mu_theta[0]
    d_nu  = nu_grid - mu_theta[1]
    P11, P12, P22 = Sigma_inv[0, 0], Sigma_inv[0, 1], Sigma_inv[1, 1]
    
    prior = -0.5 * (P11 * (d_tau**2)[:, np.newaxis] + P22 * (d_nu**2)[np.newaxis, :] + 2 * P12 * np.outer(d_tau, d_nu))
    posterior = likelihood + prior
    
    idx_row, idx_col = np.unravel_index(np.argmax(posterior), posterior.shape)
    return tau_grid[idx_row], nu_grid[idx_col]


# ==========================================
# 2. 仿真参数配置与执行
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    N1, N2 = 8, 8
    n_pilots = 4
    SNRdB = 10
    sigma2 = 10 ** (-SNRdB / 10.0)
    beta = 1
    mu_theta = np.array([0, 0])
    
    # 请根据计算资源调整蒙特卡洛次数 (跑论文图建议调高至 1000+)
    N_realizations = 500  
    
    configs_name = ['D-optimal', '2D coprime', 'Orthogonal coprime']
    colors = ["#D50B0B", '#1D73B0', '#669C03']
    styles = ['-', '--', '-.']
    
    tau_scale, doppler_scale = 1 / N2, 1 / N1

    # ==========================================
    # 3. 实验 A: 锁定方向，扫畸变程度 (Kappa)
    # ==========================================
    print("开始实验 A: SNR=10dB, 扫 Kappa 计算 MSE...")
    theta_fixed = np.pi / 3
    kappa_vals = np.logspace(0.2, 1.2, 10)
    
    mse_tau_A = {name: [] for name in configs_name}
    mse_nu_A = {name: [] for name in configs_name}
    
    t0 = time.time()
    for kappa in kappa_vals:
        lambda_max = 4.0
        lambda_min = lambda_max / kappa
        
        R = np.array([[np.cos(theta_fixed), -np.sin(theta_fixed)], 
                      [np.sin(theta_fixed),  np.cos(theta_fixed)]])
        Sigma_theta = R @ np.diag([lambda_max, lambda_min]) @ R.T
        Sigma_inv = inv(Sigma_theta)
        
        # 获取搜索出来的 Optimal 和 2D Coprime
        locs_opt, locs_2d_cop = find_optimal_configurations(Sigma_theta, N1, N2, n_pilots)
        
        # Separable Coprime 为固定的四个点
        locs_sep_cop = [(0, 0), (5, 5), (0, 5), (5, 0)]
        
        locs_list = [locs_opt, locs_2d_cop, locs_sep_cop]
        
        for idx, locs in enumerate(locs_list):
            X_pilot = np.zeros((N1, N2), dtype=complex)
            for r, c in locs: X_pilot[r, c] = 1.0
            X_pilot /= np.sqrt(np.sum(np.abs(X_pilot)**2))
            
            err_tau, err_nu = [], []
            for _ in range(N_realizations):
                tau_true, nu_true = sample_discrete_channel(mu_theta, Sigma_theta)
                h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
                y = generate_received_signal(tau_true, nu_true, h, X_pilot, sigma2, N1, N2)
                tau_est, nu_est = perform_exhaustive_map_estimation(y, X_pilot, mu_theta, Sigma_inv, beta, sigma2)
                
                err_tau.append((tau_est - tau_true)**2)
                err_nu.append((nu_est - nu_true)**2)
                
            mse_tau_A[configs_name[idx]].append(np.mean(err_tau))
            mse_nu_A[configs_name[idx]].append(np.mean(err_nu))
            
    print(f"实验 A 结束，耗时: {time.time() - t0:.2f} s")

    # ==========================================
    # 4. 实验 B: 锁定极度畸变，扫方向 (Theta)
    # ==========================================
    print("开始实验 B: SNR=10dB, 扫 Theta 计算 MSE...")
    kappa_fixed = 2.0
    theta_vals = np.linspace(0, np.pi, 10)
    theta_degrees = np.degrees(theta_vals)
    
    mse_tau_B = {name: [] for name in configs_name}
    mse_nu_B = {name: [] for name in configs_name}
    
    t0 = time.time()
    for theta in theta_vals:
        lambda_max = 4.0
        lambda_min = lambda_max / kappa_fixed
        
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta),  np.cos(theta)]])
        Sigma_theta = R @ np.diag([lambda_max, lambda_min]) @ R.T
        Sigma_inv = inv(Sigma_theta)
        
        locs_opt, locs_2d_cop = find_optimal_configurations(Sigma_theta, N1, N2, n_pilots)
        
        # 强制指定 Separable Coprime 为固定的四个点
        locs_sep_cop = [(0, 0), (5, 5), (0, 5), (5, 0)]
        
        locs_list = [locs_opt, locs_2d_cop, locs_sep_cop]
        
        for idx, locs in enumerate(locs_list):
            X_pilot = np.zeros((N1, N2), dtype=complex)
            for r, c in locs: X_pilot[r, c] = 1.0
            X_pilot /= np.sqrt(np.sum(np.abs(X_pilot)**2))
            
            err_tau, err_nu = [], []
            for _ in range(N_realizations):
                tau_true, nu_true = sample_discrete_channel(mu_theta, Sigma_theta)
                h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
                y = generate_received_signal(tau_true, nu_true, h, X_pilot, sigma2, N1, N2)
                tau_est, nu_est = perform_exhaustive_map_estimation(y, X_pilot, mu_theta, Sigma_inv, beta, sigma2)
                
                err_tau.append((tau_est - tau_true)**2)
                err_nu.append((nu_est - nu_true)**2)
                
            mse_tau_B[configs_name[idx]].append(np.mean(err_tau))
            mse_nu_B[configs_name[idx]].append(np.mean(err_nu))

    print(f"实验 B 结束，耗时: {time.time() - t0:.2f} s")

    # ==========================================
    # 5. 画图与保存
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ---- Exp A Plot ----
    ax1 = axes[0, 0]
    for idx, name in enumerate(configs_name):
        ax1.plot(kappa_vals, tau_scale * np.array(mse_tau_A[name]), marker='o', markersize=6,
                 color=colors[idx], linestyle=styles[idx], linewidth=2, label=name)
    ax1.set_xlabel(r'Prior Covariance Condition Number $\kappa$', fontsize=14)
    ax1.set_ylabel('Delay Estimation MSE', fontsize=14)
    ax1.set_title(r"Exp A: MSE vs $\kappa$ (Fixed $\theta = 60^\circ$)", fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(fontsize=12)

    ax2 = axes[0, 1]
    for idx, name in enumerate(configs_name):
        ax2.plot(kappa_vals, doppler_scale * np.array(mse_nu_A[name]), marker='^', markersize=6,
                 color=colors[idx], linestyle=styles[idx], linewidth=2, label=name)
    ax2.set_xlabel(r'Prior Covariance Condition Number $\kappa$', fontsize=14)
    ax2.set_ylabel('Doppler Estimation MSE', fontsize=14)
    ax2.set_title(r"Exp A: MSE vs $\kappa$ (Fixed $\theta = 60^\circ$)", fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend(fontsize=12)

    # ---- Exp B Plot ----
    ax3 = axes[1, 0]
    for idx, name in enumerate(configs_name):
        ax3.plot(theta_degrees, tau_scale * np.array(mse_tau_B[name]), marker='o', markersize=6,
                 color=colors[idx], linestyle=styles[idx], linewidth=2, label=name)
    ax3.set_xlim(0, 180)
    ax3.set_xticks(np.arange(0, 181, 30))
    ax3.set_xlabel(r'Prior Rotation Angle ($^\circ$)', fontsize=14)
    ax3.set_ylabel('Delay Estimation MSE', fontsize=14)
    ax3.set_title(r"Exp B: MSE vs $\theta$ (Fixed $\kappa = 10$)", fontsize=14)
    ax3.set_yscale('log')
    ax3.grid(True, which='both', linestyle='--')
    ax3.legend(fontsize=12)

    ax4 = axes[1, 1]
    for idx, name in enumerate(configs_name):
        ax4.plot(theta_degrees, doppler_scale * np.array(mse_nu_B[name]), marker='^', markersize=6,
                 color=colors[idx], linestyle=styles[idx], linewidth=2, label=name)
    ax4.set_xlim(0, 180)
    ax4.set_xticks(np.arange(0, 181, 30))
    ax4.set_xlabel(r'Prior Rotation Angle ($^\circ$)', fontsize=14)
    ax4.set_ylabel('Doppler Estimation MSE', fontsize=14)
    ax4.set_title(r"Exp B: MSE vs $\theta$ (Fixed $\kappa = 10$)", fontsize=14)
    ax4.set_yscale('log')
    ax4.grid(True, which='both', linestyle='--')
    ax4.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('Merged_Experiment_MSE_FixedCoprime.png', dpi=400)
    
    # 导出到 CSV
    data_A = pd.DataFrame({'Kappa': kappa_vals})
    data_B = pd.DataFrame({'Theta_deg': theta_degrees})
    for name in configs_name:
        data_A[f'{name}_Delay_MSE'] = tau_scale * np.array(mse_tau_A[name])
        data_A[f'{name}_Doppler_MSE'] = doppler_scale * np.array(mse_nu_A[name])
        data_B[f'{name}_Delay_MSE'] = tau_scale * np.array(mse_tau_B[name])
        data_B[f'{name}_Doppler_MSE'] = doppler_scale * np.array(mse_nu_B[name])

    data_A.to_csv('Experiment_A_MSE.csv', index=False)
    data_B.to_csv('Experiment_B_MSE.csv', index=False)
    
    print("\n所有实验已完毕，图像已保存为 Merged_Experiment_MSE_FixedCoprime.png，数据已导出至 CSV 文件。")
    plt.show()