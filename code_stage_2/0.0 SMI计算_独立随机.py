
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, inv
from numpy.linalg import slogdet
import itertools
from time import time
from joblib import Parallel, delayed 
from math import comb, factorial, log

# -------------------- 工具函数 --------------------
def F_matrix(N):
    """
    生成 N x N 的 DFT 矩阵，归一化为 1/sqrt(N)
    与 Matlab 代码的 F_matrix 等价。
    """
    n = np.arange(N).reshape((N,1))
    k = np.arange(N).reshape((1,N))
    omega = np.exp(-1j*2*np.pi/N)
    F = omega ** (n * k) / np.sqrt(N)
    return F

# --- 【并行化】为并行计算提取出的辅助函数 ---
def _calculate_sigma_properties(tau, v, N1, N2, p, beta, sigma2, X, F, FH):
    """
    计算单个 (tau, v) 组合的协方差矩阵及其相关属性。
    这个函数是为并行化而设计的。
    """
    D = N1 * N2
    A_big = np.zeros((D, p), dtype=complex)
    for i in range(p):
        tau_i = tau[i]
        v_i = v[i]
        E = np.roll(np.eye(N1), -int(tau_i), axis=0) # 延迟对应时域的循环下移
        
        TD = np.diag(np.exp(-1j * 2*np.pi *np.arange(N1)*tau_i/ N1)) 

        for l in range(N2):
            gp = np.exp(1j * 2*np.pi * v_i * l / N2) 
            
            x_col = X[:, l]
            
            s = gp*(TD @ x_col)

            blk_slice = slice(l * N1, (l + 1) * N1)
            A_big[blk_slice, i] = s

    Sigma = beta* (A_big @ A_big.conj().T) + sigma2 * np.eye(D)

    # 计算并返回该组合所需的三个矩阵/值
    Sigma_inv = inv(Sigma)
    L_chol = cholesky(Sigma, lower=True)
    log_det_Sigma = 2 * np.sum(np.log(np.diag(L_chol))).real
    
    return Sigma_inv, log_det_Sigma, L_chol

def monte_carlo_gmm_entropy_discrete(N1, N2, p, beta, sigma2, X,
                                     tau_list, v_list, M, z_all):
    """
    对应 Matlab 中的 monte_carlo_gmm_entropy_discrete 函数
    包含向量化和并行化优化以提升性能。
    """
    D = N1 * N2
    T_tau = tau_list.shape[0]

    T_v = v_list.shape[0]
 
    T=T_tau*T_v

    F = F_matrix(N1)
    FH = F.conj().T

    print(f'    正在并行预计算 {T}种信道状态的协方差矩阵...')
    
    # 创建所有 (tau, v) 组合的列表
    channel_conditions = list(itertools.product(tau_list, v_list))
    print('     所有的时延多普勒多径组合张量是：',np.array(channel_conditions).shape)

    # 使用 joblib 并行执行计算
    # n_jobs=-1 表示使用所有可用的CPU核心
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_sigma_properties)(
            tau, v, N1, N2, p, beta, sigma2, X, F, FH
        ) for tau, v in channel_conditions
    )


    # 【并行化】将并行计算的结果解包回对应的数组中
    Sigma_inv_cell = np.array([res[0] for res in results])
    log_det_Sigma_cell = np.array([res[1] for res in results])
    Sigma_chol_cell = np.array([res[2] for res in results])
    
    # 使用log行列式直接计算 H(Y|H)
    H_cond = D * np.log(np.pi * np.e) + np.mean(log_det_Sigma_cell)

    # --- 3. 生成样本，用于计算边缘熵 H(Y) ---
    print(f'    正在生成 {T}*{M}={T*M} 个接收信号样本 Y 用于计算 H(Y)...')
    y_samples = np.zeros((D, T*M), dtype=complex)

    
    for tm in range(T*M):
        z = (np.random.randn(D, ) + 1j*np.random.randn(D, ))/np.sqrt(2)
        t_star = (tm) % T  # 选择对应的信道状态索引
        L = Sigma_chol_cell[t_star, :, :]
        y_samples[:, tm] = L @ z
    
    # --- 4. 【向量化优化】蒙特卡洛方法计算边缘熵 H(Y) ---
    print(f'    正在利用向量化和并行化方法计算 H(Y)...')
    
    # 【并行化修改】这个循环同样可以并行化
    def calculate_log_density_for_state(t):
        S_inv = Sigma_inv_cell[t, :, :]
        log_det_S = log_det_Sigma_cell[t]
        mahalanobis_dist_sq = np.sum(y_samples.conj() * (S_inv @ y_samples), axis=0).real
        return -D*np.log(np.pi) - log_det_S - mahalanobis_dist_sq

    log_dens_matrix = np.array(Parallel(n_jobs=-1)(
        delayed(calculate_log_density_for_state)(t) for t in range(T)
    ))

    # 对整个 T x M 矩阵进行 Log-Sum-Exp 操作
    max_log_dens = np.max(log_dens_matrix, axis=0)
    log_pY_vector = max_log_dens + np.log(np.sum(np.exp(log_dens_matrix - max_log_dens), axis=0)) - np.log(T)

    
    # 求平均得到边缘熵 H(Y)
    H_marg = -np.mean(log_pY_vector)

    # --- 5. 计算互信息 ---
    I_est = H_marg - H_cond
    return I_est


# -------------------- 主脚本 --------------------
if __name__ == "__main__":
    # 参数
    N1 = 8
    N2 = 8
    p = 1
    beta = 1.0
    M_mc = 1000# 减少样本数以加快测试

    SNRdB = np.arange(-10,41,5)
    SNR = 10 ** (SNRdB / 10.0)

   
    noise_std = 1 * np.sqrt(1.0 / SNR)
    sigma2_list = noise_std ** 2

    
    tau_vals = np.arange(0,8)
    v_vals = np.arange(0,8)
    print("tau_vals =", tau_vals)
    print("v_vals =", v_vals)


    # 展开成 tau_vec, v_vec 形式
    tau_list = np.array(list(itertools.product(tau_vals, repeat=p)), dtype=int)  # (N_tau^p, p)
    v_list = np.array(list(itertools.product(v_vals, repeat=p)), dtype=int)      # (N_v^p, p)

    # --- 配置 ---



    # --- 导频配置 (保持不变) ---
    configs = []


    # 1. 2D Full pilots
    X = np.zeros((N1, N2), dtype=complex)
    X[:, :] = 1
    E = np.sum(np.abs(X)**2) 
    X = X / np.sqrt(E) 
    configs.append({'X': X, 'group': 1, 'name': 'Full Sampling'})

    X = np.zeros((N1, N2), dtype=complex)
    X[0, 0] = 1
    X[4, 0] = 1
    X[4, 4] = 1
    X[0, 4] = 1
    E = np.sum(np.abs(X)**2) 
    X = X/ np.sqrt(E) 
    configs.append({'X': X, 'group': 1, 'name': 'Uniform Sparse'})
    X= np.zeros((N1, N2), dtype=complex)


    # X = np.zeros((N1, N2), dtype=complex)
    # X[1, 1] = 1
    # X[0, 0] = 1
    # X[0, 1] = 1
    # X[1, 0] = 1
    # E = np.sum(np.abs(X)**2) 
    # X = X/ np.sqrt(E) 
    # configs.append({'X': X, 'group': 1, 'name': 'Separable Coprime'})
    # X= np.zeros((N1, N2), dtype=complex)





    X = np.zeros((N1, N2), dtype=complex)
    X[7, 5] = 1
    X[7, 0] = 1
    X[7, 7] = 1
    X[7, 6] = 1
    E = np.sum(np.abs(X)**2) 
    X = X/ np.sqrt(E) 
    configs.append({'X': X, 'group': 1, 'name': 'D-optimal '})
    X= np.zeros((N1, N2), dtype=complex)







    # --- 仿真循环 ---
    plt.figure(figsize=(10,7))
    plt.grid(True, which='both', linestyle='--')
    plt.xlabel('SNR (dB)', fontsize=15)
    plt.ylabel('Mutual Infomation MI=H(Y)-H(Y|h) (nats)', fontsize=15)
    plt.title(f'Different pilot pattern\'s MI v SNR', fontsize=15)

    # 解决matplotlib显示中文问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

    for cfg in configs:
        X = cfg['X']
        legend_name = cfg['name']
        
        taus_all = tau_list
        vs_all = v_list

        I_vs_SNR = np.zeros_like(SNRdB, dtype=float)
        
        D = N1 * N2
        z_all = (np.random.randn(D, M_mc) + 1j * np.random.randn(D, M_mc)) / np.sqrt(2)

        for k_idx, sigma2 in enumerate(sigma2_list):
            print(f"\n--- 计算配置: '{legend_name}', SNR: {SNRdB[k_idx]} dB ---")
            t_start = time()
            I_vs_SNR[k_idx] = monte_carlo_gmm_entropy_discrete(
                N1, N2, p, beta, sigma2, X, taus_all, vs_all, M_mc, z_all
            )
            t_end = time()
            print(f"完成! I = {I_vs_SNR[k_idx]:.4f}, 耗时: {t_end - t_start:.2f} 秒")

        plt.plot(SNRdB, I_vs_SNR, '-o', linewidth=1.2, label=legend_name)
    



    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('不同导频模式下互信息与SNR的关系.png', dpi=1000)
    plt.show()

