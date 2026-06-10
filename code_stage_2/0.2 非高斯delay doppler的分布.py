import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# 1. 参数区
# ===============================
c = 3e8
fc = 60e9
lam = c / fc

# 分布参数（硬编码）
d0 = 50# m
Delta_d = 2  # m

v0 = 15     # m/s
Delta_v = 1   # m/s

t0=0
T =6     # s

N = 40000      # Monte-Carlo 样本数（

# ===============================
# 2. Monte-Carlo 采样 (d, v, t)
# ===============================
rng = np.random.default_rng(0)
def trunc_normal(mean, std, low, high, size):
    """简单重采样实现截断高斯（无 scipy 也能跑）"""
    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        n = (size - filled) * 2  # 多采一些减少循环次数
        x = rng.normal(mean, std, n)
        x = x[(x >= low) & (x <= high)]
        m = min(len(x), size - filled)
        out[filled:filled+m] = x[:m]
        filled += m
    return out

# 用来做截断范围
d_low, d_high = d0 - Delta_d, d0 + Delta_d
v_low, v_high = max(0.1, v0 - Delta_v), v0 + Delta_v

# 把 Delta 当作 3σ
sigma_d = Delta_d 
sigma_v = Delta_v 

d = trunc_normal(d0, sigma_d, d_low, d_high, N)
v = trunc_normal(v0, sigma_v, v_low, v_high, N)

# t 还是均匀（按你要求 0~T）

t = rng.uniform(t0, T, N)

# ===============================
# 3. 映射到 Delay–Doppler
# ===============================
R = np.sqrt(d**2 + (v*t)**2)

tau = 2 * R / c                    # delay (s)
nu  = (2 / lam) * (v**2 * t / R)   # Doppler (Hz)

# ===============================
# 4. 在 (tau, nu) 上估计 PDF（2D 直方图）
# ===============================
tau_ns = tau * 1e9      # 转成 ns 
nu_khz = nu / 1e3       # 转成 kHz



import numpy as np

# Z: (N,2) 数据矩阵
Z = np.column_stack([tau_ns/tau_ns.max()*8, nu_khz/nu_khz.max()*8])

# 去均值
Zc = Z - Z.mean(axis=0, keepdims=True)

# 协方差
C = np.cov(Zc, rowvar=False)

# 特征分解``
eigvals, eigvecs = np.linalg.eigh(C)

print(eigvals)
# 排序（从大到小）
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

v_max_info = eigvecs[:, 0]   # 信息量最大的方向
v_min_info = eigvecs[:, -1]  # 信息量最小的方向

print("Max-info direction:", v_max_info)
print("Min-info direction:", v_min_info)




# 直方图网格
tau_bins = np.linspace(tau_ns.min(), tau_ns.max(), 120)
nu_bins  = np.linspace(nu_khz.min(), nu_khz.max(), 120)

H, tau_edges, nu_edges = np.histogram2d(
    tau_ns, nu_khz,
    bins=[tau_bins, nu_bins],
    density=True
)

# bin 中心
tau_c = 0.5 * (tau_edges[:-1] + tau_edges[1:])
nu_c  = 0.5 * (nu_edges[:-1] + nu_edges[1:])
TAU, NU = np.meshgrid(tau_c, nu_c, indexing="ij")

# ===============================
# 5. 3D 立体概率分布图
# ===============================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    TAU, NU, H,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)

ax.set_xlabel("Delay (ns)")
ax.set_ylabel("Doppler (kHz)")
ax.set_zlabel("Probability Density")

ax.set_title("Monte-Carlo Estimated Delay–Doppler PDF\n"
             "d,v,t ~ Uniform")

plt.tight_layout()
plt.show()


