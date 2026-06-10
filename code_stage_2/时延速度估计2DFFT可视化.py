import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors # 导入 colors 以便使用对数坐标

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
# --- 您提供的所有函数定义 (F_matrix, XtoY, greedy_match) ---
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

        # 注意：在您的原始代码中，噪声被注释掉了。
        # 如果需要加噪声，请取消下一行的注释
        # noise = np.sqrt(sigma2/2) * (np.random.randn(D) + 1j*np.random.randn(D))
        # Y[:,i] = A_big[:, i] + noise
        
        Y[:,i] = A_big[:, i] # 当前使用无噪声版本

    # 原始代码中 Y.sum(axis=1) 是对所有 p 个目标求和
    # 如果 p > 1，这将混合所有目标
    T = Y.sum(axis=1).reshape(N2,N1).T
    
    # 如果 p > 1 并且您希望在接收端模拟噪声，噪声应该加在这里
    if sigma2 > 0:
        D_flat = T.shape[0] * T.shape[1]
        noise_vec = np.sqrt(sigma2/2) * (np.random.randn(D_flat) + 1j*np.random.randn(D_flat))
        T = T + noise_vec.reshape(T.shape)

    return T

def greedy_match(true_points, est_points):
    """
    纯 numpy 实现的贪心近似匹配
    """
    true_points = np.array(true_points)
    est_points = np.array(est_points)
    
    if est_points.shape[0] == 0: # 如果没有估计到目标
        return true_points, np.array([])

    matched_true, matched_est = [], []
    est_used = set()

    for t in true_points:
        dists = np.linalg.norm(est_points - t, axis=1)
        best_dist = np.inf
        best_idx = -1
        
        # 寻找最近且未被使用的估计点
        for j in np.argsort(dists):
            if j not in est_used:
                best_dist = dists[j]
                best_idx = j
                break
        
        if best_idx != -1:
            est_used.add(best_idx)
            matched_true.append(t)
            matched_est.append(est_points[best_idx])

    return np.array(matched_true), np.array(matched_est)

def estimate_delay_doppler_multi(Y, X, num_targets=3, threshold_ratio=0.9):
    """
    多目标二维时延-多普勒估计（匹配滤波 + 多峰检测）
    """
    M, N = Y.shape

    # Step 1: 时延匹配滤波
    R_tau = np.zeros((M, N), dtype=complex)
    for n in range(N):
        R_tau[:, n] = np.fft.ifft(Y[:, n] * np.conj(X[:, n]))

    # Step 2: 多普勒匹配滤波
    # 我们使用 fftshift 来将 0 频（和 0 时延）移到中心
    R_dd_shifted = np.fft.fftshift(np.fft.fft(R_tau, axis=1), axes=1)
    
    R_abs = np.abs(R_dd_shifted)
    
    # 防止除以零
    max_val = np.max(R_abs)
    if max_val == 0:
        max_val = 1.0 
        
    R_norm = R_abs / max_val

    # Step 3: 手动检测局部最大值 (在 shift 后的谱上)
    R_flat = R_norm.flatten()
    # 确保 num_targets 不超过总点数
    k_peaks = min(num_targets, M * N) 
    
    # 使用 argpartition 获取前 k_peaks 大的值的索引
    indices_flat = np.argpartition(R_flat, -k_peaks)[-k_peaks:]
    # 对这 k_peaks 个值进行排序
    indices_sorted = indices_flat[np.argsort(-R_flat[indices_flat])]

    tau_hat_list, v_hat_list = [], []
    for idx in indices_sorted:
        tau_idx = idx // N
        v_idx = idx % N
        
        # 检查阈值
        if R_norm[tau_idx, v_idx] >= threshold_ratio:
            tau_hat_list.append(tau_idx)
            v_hat_list.append(v_idx)
            
    # 注意：返回的索引是 fftshift 后的索引
    return np.array(tau_hat_list), np.array(v_hat_list), R_norm

# --- 您的主程序 ---

# 参数设置
p = 1 # 目标数量改为4，匹配您下面的 tau_true/v_true
M = 6*8
N = 6*8

# 模拟：真实 τ, v (注意索引从0开始)
tau_true = [22] 
v_true=[3]

# 导频 X
X = np.zeros((6,6), dtype=complex)
X[1, 0] = 1
X[0, 3] = 1
X[-2, 2] = 1
X[-3, 5] = 1

X = np.tile(X, (8, 8))
E = np.sum(np.abs(X)**2) 
X = X / np.sqrt(E) if E > 0 else X # 归一化

# 模拟接收信号 Y
SNR_dB = 20# 提高信噪比以便看清峰值
sigma2 = 10**(-SNR_dB/10)
Y = XtoY(tau_true, v_true, M, N, p, sigma2, X, F_matrix(M))

# 估计
# 我们让估计函数尝试找到 p 个目标
tau_hat, v_hat, R_dd_norm = estimate_delay_doppler_multi(Y, X, num_targets=p, threshold_ratio=0.5)

# 匹配结果
true_points = np.column_stack([tau_true, v_true])
# 将 fftshift 后的索引转换回 [0, M-1] 和 [0, N-1] 范围内的索引
tau_hat_shifted = tau_hat
v_hat_shifted = (v_hat-N//2)% N  # 将多普勒索引移回 [0, N-1] 范围

est_points = np.column_stack([tau_hat_shifted, v_hat_shifted])

t_aligned, e_aligned = greedy_match(true_points, est_points)

print("真实目标(匹配后):\n", t_aligned)
print("估计目标(匹配后):\n", e_aligned)

# --- 正确可视化代码（与 fftshift 后的 R_dd_norm 对齐） ---

print("\n正在生成时延-多普勒谱可视化图...")


# 横轴多普勒坐标（fftshift 后）
doppler_axis = np.arange(-N//2, N//2)

# 使用 extent = [-N/2, N/2-1] 让图与坐标对齐

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# ======================================================
#   1) 你原来的 Colormap（增强版 jet ）
# ======================================================
jet = plt.cm.get_cmap('jet', 256)
colors_mod = jet(np.linspace(0.3, 0.92 ,256))
jet_strong_red = LinearSegmentedColormap.from_list("jet_strong_red", colors_mod)

# ======================================================
#   2) 构建 3D 网格坐标
# ======================================================
D = np.linspace(doppler_axis[0], doppler_axis[-1], R_dd_norm.shape[1])
T = np.arange(R_dd_norm.shape[0])
DD, TT = np.meshgrid(D, T)

# ======================================================
#   3) 绘制 3D surface
# ======================================================
surf = ax.plot_surface(
    DD, TT, R_dd_norm,
    cmap=jet_strong_red,
    linewidth=0,
    antialiased=True,
    rstride=1,
    cstride=1
)

# ======================================================
#   4) ⭐ 原始公式：真实点 / 估计点（含 shift）
# ======================================================

# ---- 真实点 (v_true, tau_true) shift 后 ----
v_true_shifted = np.array(v_true)

# ---- 估计点 (v_hat, tau_hat) shift 后 ----
v_hat_shifted = (v_hat) % N - N // 2

# ======================================================
#   5) 用箭头在 3D 图中指向峰值
# ======================================================

# --- 真值点（蓝色箭头） ---
for x0, y0 in zip(v_true_shifted, tau_true):
    # 找到对应格点的 z 值
    col = int(x0 - doppler_axis[0])     # 映射到矩阵列
    row = int(y0)                        # 时间轴映射到行
    z0 = R_dd_norm[row, col]

    ax.quiver(
        x0-0.2, y0-0.2, 1.01*z0,        # 起点稍微高于表面
        0, 0, z0*0.3,          # 朝 z 方向的箭头
        color='blue',
        arrow_length_ratio=0.15,
        linewidth=2
    )

# --- 估计点（红色箭头） ---
for x0, y0 in zip(v_hat_shifted, tau_hat):
    col = int(x0 - doppler_axis[0])
    row = int(y0)
    z0 = R_dd_norm[row, col]

    ax.quiver(
        x0+0.2, y0+0.2, 1.01*z0,
        0, 0, z0*0.3,
        color='red',
        arrow_length_ratio=0.15,
        linewidth=2
    )


from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='True Location',
           markerfacecolor='blue', markeredgecolor='blue', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Estimated Peak',
           markerfacecolor='red', markeredgecolor='red', markersize=12),
]

ax.legend(handles=legend_elements, loc='upper right')


# ======================================================
#   6) 标签与视觉调整
# ======================================================
ax.set_xlabel("Doppler index", fontsize=20)
ax.set_ylabel("Time index", fontsize=20)
ax.set_zlabel("Amplitude (normalized)", fontsize=20)

ax.set_title("3D Time-Delay Doppler Spectrum", fontsize=22)

ax.legend(handles=legend_elements, loc='upper right', fontsize=18)

cbar = fig.colorbar(surf, shrink=0.6, aspect=15)
cbar.ax.tick_params(labelsize=16)
cbar.set_label("normalized amplitude", fontsize=18)
plt.tight_layout()
plt.show()
