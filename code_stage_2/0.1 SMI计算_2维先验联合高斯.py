"""
修改说明：
不再分别生成 tau 和 v 的列表并在内部进行笛卡尔积。
改为在主程序中定义具体的 (tau, v) 点对，并在函数中直接遍历这些特定的组合。
这实现了时延与多普勒的强耦合（Coupled Delay-Doppler pairs）。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, inv
import itertools
from time import time
from joblib import Parallel, delayed
from math import comb, factorial, log



# -------------------- 并行计算核心单元 --------------------

def _calculate_sigma_properties(tau, v, N1, N2, p, beta, sigma2, X, F, FH):
    """
    计算单个信道状态（由 p 条路径的 tau 和 v 组成）的协方差矩阵及其属性。
    
    参数:
    tau : array-like, shape (p,) -> 当前状态下 p 条路径的时延
    v   : array-like, shape (p,) -> 当前状态下 p 条路径的多普勒
    """
    D = N1 * N2
    A_big = np.zeros((D, p), dtype=complex)
    
    # 遍历 p 条路径
    for i in range(p):
        tau_i = tau[i]
        v_i = v[i]
        
        # 时域循环移位对应的频域相位矩阵 (Time Delay Matrix)
        # 对应原公式: diag(exp(-j*2*pi*k*tau/N))
        TD = np.diag(np.exp(-1j * 2 * np.pi * np.arange(N1) * tau_i / N1)) 

        for l in range(N2):
            # 符号间多普勒相位 (Inter-symbol Doppler phase)
            gp = np.exp(1j * 2 * np.pi * v_i * l / N2)
            
            x_col = X[:, l]
            
            # 信号模型：多普勒项 * (时延项 @ 发送信号)
            # 注意：此处未显式包含 F/FH，因为是在时域/频域混合处理，与原代码逻辑保持一致
            s = gp * (TD @ x_col)

            blk_slice = slice(l * N1, (l + 1) * N1)
            A_big[blk_slice, i] = s

    # 计算协方差矩阵 Sigma = beta * H * H' + sigma^2 * I
    Sigma = beta * (A_big @ A_big.conj().T) + sigma2 * np.eye(D)

    # 计算逆矩阵、对数行列式、Cholesky分解
    Sigma_inv = inv(Sigma)
    L_chol = cholesky(Sigma, lower=True)
    log_det_Sigma = 2 * np.sum(np.log(np.diag(L_chol))).real
    
    return Sigma_inv, log_det_Sigma, L_chol


def monte_carlo_gmm_entropy_discrete(N1, N2, p, beta, sigma2, X,
                                     tau_combos, v_combos, M, z_all):
    """
    计算给定离散信道状态集下的互信息。
    
    修改：
    不再接收独立的 tau_list 和 v_list 进行笛卡尔积。
    直接接收 tau_combos 和 v_combos，它们必须具有相同的长度 T。
    tau_combos[t] 和 v_combos[t] 共同定义了第 t 个信道状态。
    
    参数:
    tau_combos : np.array, shape (T, p) - 所有可能的时延组合
    v_combos   : np.array, shape (T, p) - 所有可能的多普勒组合 (与 tau_combos 一一对应)
    """
    D = N1 * N2
    
    # T 是信道状态的总数（即离散混合模型中的分量数）
    T = tau_combos.shape[0] 
    
    # 确保输入维度一致
    assert v_combos.shape[0] == T, "时延组合和多普勒组合的数量必须一致"

    F = F_matrix(N1)
    FH = F.conj().T

    # --- 1. 并行预计算所有信道状态的协方差矩阵 ---
    print(f'    正在并行预计算 {T} 种信道状态的协方差矩阵...')
    
    # 直接遍历 0 到 T-1，提取对应的 tau 和 v 组合
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_sigma_properties)(
            tau_combos[t], v_combos[t], N1, N2, p, beta, sigma2, X, F, FH
        ) for t in range(T)
    )

    # 解包结果
    Sigma_inv_cell = np.array([res[0] for res in results])      # Shape: (T, D, D)
    log_det_Sigma_cell = np.array([res[1] for res in results])  # Shape: (T,)
    Sigma_chol_cell = np.array([res[2] for res in results])     # Shape: (T, D, D)
    
    # 计算条件熵 H(Y|H)
    # H(Y|H) = log(det(pi * e * Sigma)) 的平均值
    H_cond = D * np.log(np.pi * np.e) + np.mean(log_det_Sigma_cell)

    # --- 2. 生成样本用于计算边缘熵 H(Y) ---
    # 这里的 M 是每个状态生成的样本数 (M_mc)
    # 总样本数 = T * M
    total_samples = T * M
    print(f'    正在生成 {total_samples} 个接收信号样本 Y 用于计算 H(Y)...')
    
    y_samples = np.zeros((D, total_samples), dtype=complex)
    
    # 为了利用预生成的 z_all (标准正态分布噪声)，我们需要确保 z_all 足够大
    # 如果外部传入的 z_all 大小不匹配，重新生成或截取
    if z_all.shape[1] < total_samples:
        z_use = (np.random.randn(D, total_samples) + 1j*np.random.randn(D, total_samples))/np.sqrt(2)
    else:
        z_use = z_all[:, :total_samples]

    for i in range(total_samples):
        # 确定该样本属于哪个信道状态 t_star
        t_star = i % T 
        L = Sigma_chol_cell[t_star, :, :]
        # y = Hx + n = L * z (因为 Sigma = LL^H, y ~ CN(0, Sigma))
        y_samples[:, i] = L @ z_use[:, i]
    
    # --- 3. 蒙特卡洛方法计算边缘熵 H(Y) ---
    print(f'    正在利用向量化和并行化方法计算 H(Y)...')
    
    # 定义内部函数用于并行计算 log probability density
    def calculate_log_density_for_state(t):
        S_inv = Sigma_inv_cell[t, :, :]
        log_det_S = log_det_Sigma_cell[t]
        # 马氏距离平方: y^H * Sigma^-1 * y
        # y_samples 形状 (D, total_samples)
        # S_inv 形状 (D, D)
        # 结果形状 (total_samples,)
        mahalanobis_dist_sq = np.sum(y_samples.conj() * (S_inv @ y_samples), axis=0).real
        return -D * np.log(np.pi) - log_det_S - mahalanobis_dist_sq

    # 计算所有样本在所有 T 个高斯分量下的对数概率密度
    # 结果形状: (T, total_samples)
    log_dens_matrix = np.array(Parallel(n_jobs=-1)(
        delayed(calculate_log_density_for_state)(t) for t in range(T)
    ))

    # Log-Sum-Exp 技巧计算混合分布的对数概率 log( sum(p(y|h)) / T )
    # log p(y) = log(1/T * sum_t exp(log_dens_matrix[t]))
    #          = -log(T) + log_sum_exp(log_dens_matrix)
    
    max_log_dens = np.max(log_dens_matrix, axis=0) # 按列求最大值 (针对每个样本)
    # log_pY_vector 形状: (total_samples,)
    log_pY_vector = max_log_dens + np.log(np.sum(np.exp(log_dens_matrix - max_log_dens), axis=0)) - np.log(T)
    
    # 求平均得到边缘熵 H(Y) = - E[log p(Y)]
    H_marg = -np.mean(log_pY_vector)

    # --- 4. 计算互信息 ---
    I_est = H_marg - H_cond
    return I_est


# -------------------- 主脚本 --------------------
if __name__ == "__main__":
    # 参数设置
    N1 = 8
    N2 = 8
    p = 1   # 路径数
    beta = 1.0
    M_mc = 1 # 每个状态的蒙特卡洛样本数 (根据内存调整)

    SNRdB = np.arange(-10,50, 5)
    sigma2_list = 10 ** (-SNRdB / 10.0) # 噪声功率 (归一化信号功率为1)



# 1. 设定联合高斯分布参数
    # mu_vec: [tau_mean, v_mean]
    # cov_mat: [[var_tau, cov], [cov, var_v]]
    # 这里的数值您可以根据需要随意修改
    mu_vec = np.array([0, 0]) 
    cov_mat = np.array([[ 5 ,2],
                    [ 2, 1]])

    N_samples = 1000  # 样本生成数量
    rng = np.random.default_rng(0)

    # 2. 生成联合高斯分布样本
    # samples shape: (N_samples, 2)
    samples = rng.multivariate_normal(mu_vec, cov_mat, N_samples)

    tau_raw = samples[:, 0]
    nu_raw  = samples[:, 1]

    # 3. 归一化处理 (映射到 [0, 1] 范围)
    # 保持原代码逻辑：先归一化，后续再乘以8映射到网格
    
    # --- 处理 Delay (tau) ---
    min_tau, max_tau = np.min(tau_raw), np.max(tau_raw)
    # print(max_tau, min_tau)
    if max_tau == min_tau:
        tau = np.zeros_like(tau_raw)
    else:
        tau = tau_raw

    # --- 处理 Doppler (nu) ---
    min_nu, max_nu = np.min(nu_raw), np.max(nu_raw)
    # print(max_nu, min_nu)
    if max_nu == min_nu:
        nu = np.zeros_like(nu_raw)
    else:
        nu = nu_raw

    # 4. 映射到 8x8 网格尺度

    g=1
    q=64

    tau_points=np.round(tau_raw * q) / q
    v_points  =np.round(nu_raw   * q) / q

    tau_points =tau_points*8
    v_points   =v_points*8

    print(f"生成的时延点 (tau): {max(tau_points)}, {min(tau_points)}")
    print(f"生成的多普勒点 (v): {max(v_points)}, {min(v_points)}")

    base_pairs = []

    if len(tau_points) == len(v_points):
        for t, v in zip(tau_points, v_points):
            base_pairs.append((t, v))

    pair_combinations = list(itertools.product(base_pairs, repeat=p))
    
    # 将组合转换为 numpy 数组以便索引
    # tau_combos shape: (Total_States, p)
    # v_combos shape:   (Total_States, p)
    tau_combos = np.array([[pair[0] for pair in state] for state in pair_combinations])
    v_combos   = np.array([[pair[1] for pair in state] for state in pair_combinations])

    print(f"生成的信道状态总数 T = {len(pair_combinations)}")
    print(f"tau_combos shape: {tau_combos.shape}")

    # --- 导频配置 (保持不变) ---

    configs = []
    #1. 2D Full pilots
    X = np.zeros((N1, N2), dtype=complex)
    X[:, :] = 1
    E = np.sum(np.abs(X)**2) 
    X = X / np.sqrt(E) 
    configs.append({'X': X, 'group': 1, 'name': 'full sampling'})




    X = np.zeros((N1, N2), dtype=complex)
    X[0, 0] = 1
    X[0, 4] = 1
    X[4, 0] = 1
    X[4, 4] = 1
    E = np.sum(np.abs(X)**2) 
    X = X/ np.sqrt(E) 
    configs.append({'X': X, 'group': 1, 'name': 'uniform sampling'})
    X= np.zeros((N1, N2), dtype=complex)



    # X = np.zeros((N1, N2), dtype=complex)
    # X[0, 0] = 1
    # X[0, 3] = 1
    # X[3, 0] = 1
    # X[3, 3] = 1
    # E = np.sum(np.abs(X)**2) 
    # X = X/ np.sqrt(E) 
    # configs.append({'X': X, 'group': 1, 'name': 'separable coprime'})
    # X= np.zeros((N1, N2), dtype=complex)


    # X = np.zeros((N1, N2), dtype=complex)
    # X[7, 7] = 1
    # X[0, 7] = 1
    # X[7, 6] = 1
    # X[7, 0] = 1
    # E = np.sum(np.abs(X)**2) 
    # X = X/ np.sqrt(E) 
    # configs.append({'X': X, 'group': 1, 'name': 'joint coprime'})
    # X= np.zeros((N1, N2), dtype=complex)

    # X = np.zeros((N1, N2), dtype=complex)
    # X[7, 5] = 1
    # X[7, 0] = 1
    # X[7, 7] = 1
    # X[7, 6] = 1
    # E = np.sum(np.abs(X)**2) 
    # X = X/ np.sqrt(E) 
    # configs.append({'X': X, 'group': 1, 'name': 'D-optimal'})
    # X= np.zeros((N1, N2), dtype=complex)




    # --- 仿真循环 ---
    plt.figure(figsize=(10,7))
    plt.grid(True, which='both', linestyle='--')
    plt.xlabel('SNR (dB)', fontsize=15)
    plt.ylabel('Mutual Infomation (nats)', fontsize=15)
    plt.title(f'MI vs SNR with Coupled Delay-Doppler Pairs', fontsize=15)

    # 字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

    # 预生成足够的随机噪声样本 (最大需求)
    D = N1 * N2
    max_samples = len(pair_combinations) * M_mc
    print(f"预生成噪声库大小: {D} x {max_samples}")
    z_all = (np.random.randn(D, max_samples) + 1j * np.random.randn(D, max_samples)) / np.sqrt(2)

    for cfg in configs:
        X_pilot = cfg['X']
        legend_name = cfg['name']
        
        I_vs_SNR = np.zeros_like(SNRdB, dtype=float)
        
        for k_idx, sigma2 in enumerate(sigma2_list):
            print(f"\n--- 计算配置: '{legend_name}', SNR: {SNRdB[k_idx]} dB ---")
            t_start = time()
            
            # 调用修改后的函数，传入配对好的 combos
            I_vs_SNR[k_idx] = monte_carlo_gmm_entropy_discrete(
                N1, N2, p, beta, sigma2, X_pilot, 
                tau_combos, v_combos, 
                M_mc, z_all
            )
            
            t_end = time()
            print(f"完成! I = {I_vs_SNR[k_idx]:.4f}, 耗时: {t_end - t_start:.2f} 秒")

        plt.plot(SNRdB, I_vs_SNR, '-o', linewidth=1.2, label=legend_name)
        print(f"配置 '{legend_name}', 互信息结果: {I_vs_SNR}")
    
    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('Coupled_Delay_Doppler_MI.png', dpi=1000)
    plt.show()