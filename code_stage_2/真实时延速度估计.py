import numpy as np
#from dit.algorithms import mutual_information_kraskov
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


def XtoY(tau, v, N1, N2, p, sigma2, X, F):

    FH = F.conj().T

    D = N1 * N2
    A_big = np.zeros((D, p), dtype=complex)
    Y=np.zeros((D, p), dtype=complex)

    for i in range(p):
        tau_i = tau[i]
        v_i = v[i]
        
        TD = np.diag(np.exp(-1j * 2*np.pi *np.arange(N1)*tau_i/ N1)) 

        for l in range(N2):
            gp = np.exp(1j * 2*np.pi * v_i * l / N2) 
            Qil = gp * np.eye(N1)
            x_col = X[:, l]
            
            s = F @ (Qil@ (FH @(TD @ x_col)))

            blk_slice = slice(l * N1, (l + 1) * N1)
            A_big[blk_slice, i] = s

        Y[:,i] = (A_big[:, i]) #+ np.sqrt(sigma2/2) * (np.random.randn(D) + 1j*np.random.randn(D))
        
        T=Y.sum(axis=1).reshape(N2,N1).T
        

    return T

def estimate_delay_doppler_2D(Y, X):
    """
    二维时延-多普勒估计（匹配滤波法）
    Y: 接收信号矩阵 (M x N)
    X: 发射信号矩阵 (M x N)
    返回：
      delay_idx: 时延峰值索引
      doppler_idx: 多普勒峰值索引
      R_dd: 二维时延-多普勒匹配结果（幅度谱）
    """

    M, N = Y.shape

    # Step 1: 对每个符号做频域乘法（延时匹配滤波）
    R_tau = np.zeros((M, N), dtype=complex)
    for n in range(N):
        R_tau[:, n] = np.fft.ifft(Y[:, n] * np.conj(X[:, n]))

    # Step 2: 对时间维（即符号维）做FFT（多普勒匹配滤波）
    R_dd = np.fft.fft(R_tau, axis=1)

    # Step 3: 取幅度谱（功率谱）
    R_abs = np.abs(R_dd)

    # Step 4: 找到峰值位置
    delay_idx, doppler_idx = np.unravel_index(np.argmax(R_abs), R_abs.shape)

    return delay_idx, doppler_idx#, R_abs

def greedy_match(true_points, est_points):
    """
    纯 numpy 实现的贪心近似匹配（不完全等价匈牙利算法，但快且不依赖 scipy）
    """
    true_points = np.array(true_points)
    est_points = np.array(est_points)

    matched_true, matched_est = [], []
    est_used = set()

    for t in true_points:
        dists = np.linalg.norm(est_points - t, axis=1)
        for j in np.argsort(dists):  # 选择最近的未匹配估计目标
            if j not in est_used:
                est_used.add(j)
                matched_true.append(t)
                matched_est.append(est_points[j])
                break

    return np.array(matched_true), np.array(matched_est)


def estimate_delay_doppler_multi(Y, X, num_targets=3, threshold_ratio=0.9):
    """
    多目标二维时延-多普勒估计（匹配滤波 + 多峰检测）
    无需 skimage，仅用 numpy。
    """
    M, N = Y.shape

    # Step 1: 时延匹配滤波
    R_tau = np.zeros((M, N), dtype=complex)
    for n in range(N):
        R_tau[:, n] = np.fft.ifft(Y[:, n] * np.conj(X[:, n]))

    # Step 2: 多普勒匹配滤波
    R_dd = np.fft.fft(R_tau, axis=1)
    R_abs = np.abs(R_dd)
    R_norm = R_abs / np.max(R_abs)

    # Step 3: 手动检测局部最大值
    R_flat = R_norm.flatten()
    indices = np.argpartition(R_flat, -num_targets)[-num_targets:]  # 取前 num_targets 大值
    indices = indices[np.argsort(-R_flat[indices])]  # 从大到小排序

    tau_hat_list, v_hat_list = [], []
    for idx in indices:
        tau_idx = idx // N
        v_idx = idx % N
        if R_norm[tau_idx, v_idx] >= threshold_ratio:
            tau_hat_list.append(tau_idx)
            v_hat_list.append(v_idx)

    return np.array(tau_hat_list), np.array(v_hat_list), R_norm

import numpy as np

def estimate_delay_doppler_dictOMP(Y, X, num_targets=3, threshold_ratio=0.9):
    """
    多目标时延-多普勒估计（字典法 + OMP）
    完全利用稀疏导频 X，严格按照 XtoY 顺序：
      TD -> FH -> Qil -> F
    输入输出接口与原函数一致。
    """
    N1, N2 = Y.shape

    # 1. 找 pilot 位置
    pilot_pos = np.argwhere(np.abs(X) > 1e-12)
    P = pilot_pos.shape[0]
    if P == 0:
        raise ValueError("X has no non-zero pilots.")

    m_idx = pilot_pos[:, 0]  # time indices
    n_idx = pilot_pos[:, 1]  # pulse indices

    # 2. 构造单位化 DFT F 和 FH
    n = np.arange(N1)
    k = n.reshape((N1,1))
    F = (1/np.sqrt(N1)) * np.exp(-1j*2*np.pi*k*n/N1)
    FH = F.conj().T

    # 3. 生成完整字典 Φ
    num_tau = N1
    num_dop = N2
    num_atoms = num_tau * num_dop
    Phi = np.zeros((P, num_atoms), dtype=complex)

    X_cols = [X[:, l] for l in range(N2)]

    col = 0
    for tau_i in range(num_tau):
        TD_diag = np.exp(-1j*2*np.pi*np.arange(N1)*tau_i/N1)
        for v_j in range(num_dop):
            atom_vec = np.zeros(P, dtype=complex)
            for idx_p, (m, l) in enumerate(pilot_pos):
                x_col = X_cols[l]
                td_x = TD_diag * x_col
                temp = FH @ td_x
                gp = np.exp(1j * 2*np.pi * v_j * l / N2)  # 注意+号，和XtoY一致
                temp2 = gp * temp
                s = F @ temp2
                atom_vec[idx_p] = s[m]
            Phi[:, col] = atom_vec
            col += 1

    # 4. 测量向量
    y = Y[m_idx, n_idx]

    # 5. 简单 OMP 稀疏恢复
    residual = y.copy()
    support = []
    coef_s = None
    max_atoms = min(num_targets, num_atoms)
    for it in range(max_atoms):
        correlations = np.abs(Phi.conj().T @ residual)
        idx = int(np.argmax(correlations))
        if idx in support:
            break
        support.append(idx)
        Phi_s = Phi[:, support]
        coef_s, *_ = np.linalg.lstsq(Phi_s, y, rcond=None)
        residual = y - Phi_s @ coef_s
        if np.linalg.norm(residual) < 1e-8:
            break

    # 6. 构建稀疏系数 map
    coef = np.zeros(num_atoms, dtype=complex)
    if coef_s is not None and len(support) > 0:
        coef[np.array(support, dtype=int)] = coef_s

    R_map = np.abs(coef.reshape((num_tau, num_dop)))
    maxval = np.max(R_map)
    R_norm = R_map if maxval==0 else R_map/maxval

    # 7. 选峰
    tau_hat_list = []
    v_hat_list = []
    flat_idx = np.argsort(-R_norm.flatten())[:num_targets]
    for idx in flat_idx:
        tau_idx = int(idx // num_dop)
        v_idx = int(idx % num_dop)
        if R_norm[tau_idx, v_idx] >= threshold_ratio:
            tau_hat_list.append(tau_idx)
            v_hat_list.append(v_idx)

    return np.array(tau_hat_list), np.array(v_hat_list), R_norm




# 参数设置
p=1
M = 16*8
N=16*8
I_max = np.log(M*N)
print(f"理论上界(single path): {I_max:.2f} nats")



# 模拟：真实 τ, v 是连续值（不在网格上！）
n_samples = 20000
tau_true = [np.random.uniform(0, M)] # 连续！
v_true   = [np.random.uniform(0,N)] # 连续！

tau_true = [13,48,19,12] # 连续！
v_true   = [16,48,33,33]# 连续！


X = np.zeros((16,16), dtype=complex)
X[0, 0] = 1
X[1, 1] = 1
X[2, 1] = 1

X = np.tile(X, (8, 8))
E = np.sum(np.abs(X)**2) 
X = X/ np.sqrt(E)  # 归一化总能量为 MN

# OFDM 简化模型：接收信号幅度（忽略相位噪声）
SNR_dB = 10
sigma2 = 10**(-SNR_dB/10)
Y=XtoY(tau_true, v_true, M, N, p, sigma2, X, F_matrix(M))



# 示例：用你模拟的 Y
tau_hat, v_hat,_ = estimate_delay_doppler_multi(Y, X,p)


#做一个匈牙利匹配便于观看
true_points = np.column_stack([tau_true, v_true])
est_points  = np.column_stack([tau_hat,  v_hat])

t_aligned, e_aligned = greedy_match(true_points, est_points)

print("真实目标(匹配后):\n", t_aligned)
print("估计目标(匹配后):\n", e_aligned)



# # 构造联合样本：(τ, v, Re(Y), Im(Y))
# joint = np.column_stack([
#     tau_true, v_true,
#     Y.real, Y.imag
# ])  # shape: (n_samples, 4)

# # KSG 估计 I(τ,v; Y)
# I_est = mutual_information_kraskov(joint, k=3)
# print(f"KSG 估计: {I_est:.2f} bits")