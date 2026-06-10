import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from functools import reduce
from scipy.linalg import inv
import time

# ==========================================
# 1. 约束与拓扑检测器
# ==========================================
def get_system_gcd(pilot_locs, Nc, Ns):
    """计算二维导频排布的系统最大公约数 (System GCD)"""
    diff_vectors = set()
    n_p = len(pilot_locs)
    int_locs = [(int(round(p[0])), int(round(p[1]))) for p in pilot_locs]
    
    for i in range(n_p):
        for j in range(i + 1, n_p):
            u = int_locs[i][0] - int_locs[j][0]
            v = int_locs[i][1] - int_locs[j][1]
            diff_vectors.add((u, v))
            
    if not diff_vectors: 
        return Nc * Ns
    
    terms = {Nc * Ns}
    for u, v in diff_vectors:
        terms.add(abs(v * Nc))
        terms.add(abs(u * Ns))
            
    for i, (u1, v1) in enumerate(diff_vectors):
        for u2, v2 in list(diff_vectors)[i+1:]:
            terms.add(abs(u1 * v2 - u2 * v1))
            
    return reduce(math.gcd, list(terms))

def is_separable_coprime(pilot_locs, Nc, Ns):
    """检测是否为可分离的互质结构 (Separable Coprime)"""
    int_locs = [(int(round(p[0])), int(round(p[1]))) for p in pilot_locs]
    m_vals = list(set([p[0] for p in int_locs]))
    n_vals = list(set([p[1] for p in int_locs]))
    
    if len(m_vals) * len(n_vals) != len(pilot_locs): 
        return False
    if len(m_vals) != 2 or len(n_vals) != 2: 
        return False
        
    diff_m = abs(m_vals[0] - m_vals[1])
    diff_n = abs(n_vals[0] - n_vals[1])
    
    return (math.gcd(diff_m, Nc) == 1) and (math.gcd(diff_n, Ns) == 1)

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

# ==========================================
# 2. 快速矢量化导频模式搜索
# ==========================================
def find_optimal_configurations(Sigma_theta, M=8, N=8, n_pilots=4):
    """在 C(64, 4) 组合空间中搜索满足不同约束的最佳导频布局"""
    print("正在进行导频排布空间的矢量化搜索与筛选...")
    t_start = time.time()
    
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
    locs_sep_coprime = None
    locs_opt = None
    
    for idx in sorted_indices:
        curr_locs = list(zip(m[idx], n[idx]))
        if is_non_adjacent(curr_locs, M, N): 
            if locs_opt is None:
                    locs_opt = curr_locs            
            if locs_2d_coprime is None and locs_opt:
                if get_system_gcd(curr_locs, M, N) == 1:
                    locs_2d_coprime = curr_locs
            if locs_sep_coprime is None:
                if is_separable_coprime(curr_locs, M, N):
                    locs_sep_coprime = curr_locs

            if (locs_2d_coprime is not None) and (locs_sep_coprime is not None) and (locs_opt is not None):
                break
                
    print(f"搜索完成! 耗时: {time.time() - t_start:.2f} 秒.")
    return locs_opt, locs_2d_coprime, locs_sep_coprime

# ==========================================
# 3. 信号生成与确定性离散网格穷举
# ==========================================
def generate_received_signal(tau, nu, h, X, sigma2, N1=8, N2=8):
    """根据确定性的时延与多普勒真值生成接收观测信号"""
    D = N1 * N2
    s = np.zeros(D, dtype=complex)
    TD = np.diag(np.exp(-1j * 2 * np.pi * np.arange(N1) * tau / N1))
    
    for l in range(N2):
        gp = np.exp(1j * 2 * np.pi * nu * l / N2)
        s[l * N1 : (l + 1) * N1] = gp * (TD @ X[:, l])
        
    noise = np.sqrt(sigma2 / 2.0) * (np.random.randn(D) + 1j * np.random.randn(D))
    y = h * s + noise
    return y

def sample_discrete_channel(mu_theta, Sigma_theta, q=19):
    """
    从高斯先验中进行真值采样，并将其严格投影量化到 
    q x q 的确定性网格点上，确保无网格失配。
    """
    while True:
        theta_raw = np.random.multivariate_normal(mu_theta, Sigma_theta)
        # 3sigma 判定范围约束
        if 0.0 <= theta_raw[0] <= 8.0 and 0.0 <= theta_raw[1] <= 8.0:
            # 投影量化到离散网格 (步长 8.0 / q)
            idx_tau = int(np.round(theta_raw[0] * q / 8.0))
            idx_nu  = int(np.round(theta_raw[1] * q / 8.0))
            
            # 确保下标不越界
            idx_tau = np.clip(idx_tau, 0, q - 1)
            idx_nu  = np.clip(idx_nu, 0, q - 1)
            
            tau_true = idx_tau * (8.0 / q)
            nu_true  = idx_nu * (8.0 / q)
            return tau_true, nu_true

def perform_exhaustive_map_estimation(y, X, mu_theta, Sigma_inv, beta, sigma2, q=19, N1=8, N2=8):
    """
    全空间网格直接穷举法 (Exhaustive Search)：
    直接对 q x q 离散集合中的所有 4096 种可能的状态计算后验概率，
    通过全局 argmax 比对出最符合观测响应的时延和多普勒组合。
    """
    tau_grid = np.arange(q) * (8.0 / q)
    nu_grid  = np.arange(q) * (8.0 / q)
    
    # 1. 提取活跃信道响应并做复共轭处理
    y_matrix = y.reshape((N2, N1)).T
    Y = y_matrix * np.conj(X)
    
    # 2. 构造离散穷举矩阵的傅里叶核
    W_tau = np.exp(1j * 2 * np.pi * np.outer(np.arange(N1), tau_grid) / N1)
    W_nu  = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N2), nu_grid) / N2)
    
    # 3. 计算所有可能状态下的相似度匹配投影
    s_H_y_grid = (Y.T @ W_tau).T @ W_nu
    
    # 4. 似然矩阵评估
    likelihood = (beta / (sigma2 * (sigma2 + beta))) * (np.abs(s_H_y_grid)**2)
    
    # 5. 先验矩阵评估
    d_tau = tau_grid - mu_theta[0]
    d_nu  = nu_grid - mu_theta[1]
    P11, P12, P22 = Sigma_inv[0, 0], Sigma_inv[0, 1], Sigma_inv[1, 1]
    
    term_tau = P11 * (d_tau**2)[:, np.newaxis]
    term_nu  = P22 * (d_nu**2)[np.newaxis, :]
    term_cross = 2 * P12 * np.outer(d_tau, d_nu)
    prior = -0.5 * (term_tau + term_nu + term_cross)
    
    # 6. 直接穷举比较
    posterior = likelihood + prior
    
    idx_max = np.argmax(posterior)
    idx_row, idx_col = np.unravel_index(idx_max, posterior.shape)
    
    return tau_grid[idx_row], nu_grid[idx_col]

# ==========================================
# 4. 仿真主程序
# ==========================================
if __name__ == "__main__":
    N1, N2 = 8, 8
    beta = 1
    np.random.seed(42)
    
    
    # 构造联合高斯先验 (3-Sigma 范围不超出边界)
    mu_theta = np.array([0, 0])  
    angle = np.pi * 40.0 / 180.0     
    cond_num = 4                   
    lambda_max = 4                 # 标准差 1.14，3sigma 为 3.42，截断范围 [0.58, 7.42]
    lambda_min = lambda_max / cond_num
    
    R = np.array([[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle),  np.cos(angle)]])
    Sigma_theta = R @ np.diag([lambda_max, lambda_min]) @ R.T
    Sigma_inv = inv(Sigma_theta)
    
    # 搜索优化导频
    locs_opt, locs_2d_cop, locs_sep_cop = find_optimal_configurations(Sigma_theta, N1, N2, n_pilots=4)
    
    configs = []
    
    # 1) Full Sampling (基准)
    X_full = np.ones((N1, N2), dtype=complex)
    X_full /= np.sqrt(np.sum(np.abs(X_full)**2))
    configs.append({'name': 'Full Sampling ', 'matrix': X_full, 'color': 'black', 'style': '--'})
    
    # 2) FIM-MAP Optimal (4 Pilots)
    X_opt = np.zeros((N1, N2), dtype=complex)
    for r, c in locs_opt: X_opt[r, c] = 1.0
    X_opt /= np.sqrt(np.sum(np.abs(X_opt)**2))
    configs.append({'name': 'FIM-MAP Optimal', 'matrix': X_opt, 'color': "#D50B0B", 'style': '-'})
    
    # 3) 2D Coprime (4 Pilots)
    X_2d_cop = np.zeros((N1, N2), dtype=complex)
    for r, c in locs_2d_cop: X_2d_cop[r, c] = 1.0
    X_2d_cop /= np.sqrt(np.sum(np.abs(X_2d_cop)**2))
    configs.append({'name': '2D Coprime', 'matrix': X_2d_cop, 'color': "#1D73B0", 'style': '--'})
    
    # 4) Separable Coprime (4 Pilots)
    X_sep_cop = np.zeros((N1, N2), dtype=complex)
    for r, c in locs_sep_cop: X_sep_cop[r, c] = 1.0
    X_sep_cop /= np.sqrt(np.sum(np.abs(X_sep_cop)**2))
    configs.append({'name': 'Orthonormal Coprime', 'matrix': X_sep_cop, 'color': "#669C03", 'style': '-'})

    # 仿真参数
    SNRdB_list = np.arange(0, 61, 5)
    N_realizations = 5000      # 每次蒙特卡洛内的信道实现次数
    m2 = 1                 # 外层蒙特卡洛平均次数（可调）

    # 存储累积的MSE（用于多次蒙特卡洛平均）
    mse_tau_accum = {cfg['name']: np.zeros(len(SNRdB_list)) for cfg in configs}
    mse_nu_accum = {cfg['name']: np.zeros(len(SNRdB_list)) for cfg in configs}

    print(f"\n开始多重蒙特卡洛仿真: 外层循环 {m2} 次，每次内层 {N_realizations} 次信道实现")

    for mc_run in range(m2):
        print(f"\n========== 外层蒙特卡洛运行次数: {mc_run+1}/{m2} ==========")
        
        # 每次外层循环重新初始化单次MSE存储
        mse_tau_run = {cfg['name']: np.zeros(len(SNRdB_list)) for cfg in configs}
        mse_nu_run = {cfg['name']: np.zeros(len(SNRdB_list)) for cfg in configs}
        
        for snr_idx, SNRdB in enumerate(SNRdB_list):
            sigma2 = 10 ** (-SNRdB / 10.0)
            print(f"  当前 SNR = {SNRdB} dB...")
            
            # 预先生成本次SNR下的信道状态（严格投影在离散网格点上）
            channel_states = []
            for _ in range(N_realizations):
                tau_true, nu_true = sample_discrete_channel(mu_theta, Sigma_theta)
                h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
                channel_states.append((tau_true, nu_true, h))
            
            for cfg in configs:
                errors_tau = []
                errors_nu = []
                X_pilot = cfg['matrix']
                
                for realization in range(N_realizations):
                    tau_true, nu_true, h = channel_states[realization]
                    
                    # 生成观测信号
                    y = generate_received_signal(tau_true, nu_true, h, X_pilot, sigma2, N1, N2)
                    
                    # 离散全网格穷举直接比对
                    tau_est, nu_est = perform_exhaustive_map_estimation(
                        y, X_pilot, mu_theta, Sigma_inv, beta, sigma2, N1=N1, N2=N2
                    )
                    
                    errors_tau.append((tau_est - tau_true) ** 2)
                    errors_nu.append((nu_est - nu_true) ** 2)
                
                mse_tau_run[cfg['name']][snr_idx] = np.mean(errors_tau)
                mse_nu_run[cfg['name']][snr_idx] = np.mean(errors_nu)
        
        # 累加当前外层循环的结果
        for cfg in configs:
            mse_tau_accum[cfg['name']] += mse_tau_run[cfg['name']]
            mse_nu_accum[cfg['name']] += mse_nu_run[cfg['name']]

    # 最后取平均
    for cfg in configs:
        mse_tau_accum[cfg['name']] /= m2
        mse_nu_accum[cfg['name']] /= m2

    print("\n多重蒙特卡洛仿真完成，结果已平均。")

# 此时 mse_tau_accum 和 mse_nu_accum 即为最终平均后的MSE（维度 len(SNRdB_list)）
    # ----------------------------------------------------
    # 绘图输出
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tau_scale = 1 / N2
    doppler_scale = 1 / N1

    results = {
        "SNR_dB": SNRdB_list.tolist(),
        "configs": [],
        "tau_scale": tau_scale,
        "doppler_scale": doppler_scale,
    }

    for cfg in configs:
        name = cfg['name']
        mse_tau_scaled = (tau_scale * mse_tau_accum[name]).tolist()
        mse_nu_scaled = (doppler_scale * mse_nu_accum[name]).tolist()
        results["configs"].append({
            "name": name,
            "color": cfg['color'],
            "linestyle": cfg['style'],
            "mse_tau": mse_tau_scaled,     
            "mse_nu": mse_nu_scaled
        })

    # 保存为 JSON 文件
    import json
    with open("simulation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("结果已保存为 simulation_results.json")



    # 时延估计 MSE
    ax1 = axes[0]
    for cfg in configs:
        ax1.plot(SNRdB_list, tau_scale * mse_tau_accum[cfg['name']], marker='o', markersize=8,
                 color=cfg['color'], linestyle=cfg['style'], linewidth=2, label=cfg['name'])
    ax1.set_xlabel('SNR (dB)', fontsize=16)
    ax1.set_ylabel('Delay Estimation MSE', fontsize=16)
    #ax1.set_title('Delay Estimation MSE vs SNR', fontsize=16)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(fontsize=16, loc='lower left')
    
    # 多普勒估计 MSE
    ax2 = axes[1]
    for cfg in configs:
        ax2.plot(SNRdB_list, doppler_scale * mse_nu_accum[cfg['name']], marker='^',markersize=8, 
                 color=cfg['color'], linestyle=cfg['style'], linewidth=2, label=cfg['name'])
    ax2.set_xlabel('SNR (dB)', fontsize=16)
    ax2.set_ylabel('Doppler Estimation MSE', fontsize=16)
    #ax2.set_title('Doppler Estimation MSE vs SNR', fontsize=16)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend(fontsize=16, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('Exhaustive_Grid_MAP_Estimation_MSE.png', dpi=400)
    print("\n仿真运行成功！图像已保存为 Exhaustive_Grid_MAP_Estimation_MSE.png")
    plt.show()