import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from functools import reduce
import time

# ========== 全局绘图设置 ==========
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# ========== 1. 约束检测器 ==========
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

def get_system_gcd(pilot_locs, Nc, Ns):
    """计算二维导频排布的系统最大公约数"""
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

# ========== 2. Fisher信息矩阵计算 ==========
def fisher_matrix_from_pilots(locs_array, M, N, sigma2=1):
    """根据导频坐标列表计算 FIM J (2x2)"""
    locs_array = np.array(locs_array, dtype=float)
    coeff = (4 * np.pi**2) / (10**(sigma2))
    df = 1.0 / M
    Ts = 1.0 / N
    P = len(locs_array)
    if P == 0:
        return np.zeros((2, 2))

    m, n = locs_array[:, 0], locs_array[:, 1]
    var_m = np.mean(m**2) - np.mean(m)**2
    var_n = np.mean(n**2) - np.mean(n)**2
    cov_mn = np.mean(m * n) - np.mean(m) * np.mean(n)

    J11 = coeff * P * var_m * (df**2)
    J22 = coeff * P * var_n * (Ts**2)
    J12 = -coeff * P * cov_mn * (df * Ts)

    return np.array([[J11, J12], [J12, J22]])

def alignment_score(J, Sigma):
    """计算对齐评分 det(I + Sigma @ J)"""
    return np.linalg.det(np.eye(2) + Sigma @ J)

# ========== 3. 绘制椭圆的辅助函数（面积归一化） ==========
def plot_ellipse(ax, matrix, facecolor, edgecolor, linestyle, label=None, scale=1.0):
    """绘制归一化后的椭圆（行列式归一化为1，强调形状）"""
    det = np.linalg.det(matrix)
    if det > 1e-12:
        matrix_norm = matrix / np.sqrt(det)
    else:
        matrix_norm = matrix

    eigvals, eigvecs = np.linalg.eigh(matrix_norm)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width = 2 * np.sqrt(max(eigvals[1], 0)) * scale
    height = 2 * np.sqrt(max(eigvals[0], 0)) * scale

    ell = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                  facecolor=facecolor, edgecolor=edgecolor, linewidth=2,
                  linestyle=linestyle, alpha=0.5, label=label)
    ax.add_patch(ell)

# ========== 4. 生成有效组合（非相邻约束）及预计算 ==========
M, N, n_pilots = 8, 8, 4
sigma2 = 1
all_indices = list(range(M * N))

print("生成所有 C(64,4) 组合...")
combos_raw = list(itertools.combinations(all_indices, n_pilots))
print(f"总组合数: {len(combos_raw)}")

valid_combos = []          # 坐标列表形式
valid_indices_list = []    # 线性索引列表
print("过滤环面不相邻的组合...")
t0 = time.time()
for combo in combos_raw:
    locs = [(idx // N, idx % N) for idx in combo]
    if is_non_adjacent(locs, M, N):
        valid_combos.append(locs)
        valid_indices_list.append(list(combo))
print(f"满足不相邻条件的组合数: {len(valid_combos)}，耗时 {time.time()-t0:.2f} 秒")

n_combos = len(valid_combos)
if n_combos == 0:
    raise RuntimeError("没有满足非相邻约束的组合，无法继续。")

combos_arr = np.array(valid_indices_list, dtype=int)      # (n_combos, 4)
m_coords = combos_arr // N
n_coords = combos_arr % N

# 预计算 J 的统计量
print("预计算每个组合的 J 矩阵参数...")
mean_m = np.mean(m_coords, axis=1, keepdims=True)
mean_n = np.mean(n_coords, axis=1, keepdims=True)
var_m = np.mean(m_coords**2, axis=1) - mean_m.squeeze()**2
var_n = np.mean(n_coords**2, axis=1) - mean_n.squeeze()**2
cov_mn = np.mean(m_coords * n_coords, axis=1) - (mean_m.squeeze() * mean_n.squeeze())

coeff_fim = 4 * np.pi**2 / (10**sigma2)
df = 1.0 / M
Ts = 1.0 / N
J11_arr = coeff_fim * n_pilots * var_m * (df**2)
J22_arr = coeff_fim * n_pilots * var_n * (Ts**2)
J12_arr = -coeff_fim * n_pilots * cov_mn * (df * Ts)

# 计算 GCD 标志
print("计算系统 GCD 标志 (互质: gcd==1)...")
gcd_flag_arr = np.zeros(n_combos, dtype=bool)
for idx in range(n_combos):
    locs = valid_combos[idx]
    if get_system_gcd(locs, M, N) == 1:
        gcd_flag_arr[idx] = True
print(f"互质且非相邻的组合数: {np.sum(gcd_flag_arr)}")

# 固定正交 Coprime 模式 (00,55,05,50)
fixed_locs = [(0,0), (5,5), (0,5), (5,0)]
J_fixed = fisher_matrix_from_pilots(fixed_locs, M, N, sigma2)

# ========== 5. 不同先验协方差 Sigma 下的比较 ==========
theta = np.pi * 45 / 180          # 45度倾斜
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
a_vals = np.array([1, 2, 4, 8])   # 条件数 κ = a^2
num_points = len(a_vals)

# 顺序：正交(0), 全局最优(1), 2D互质(2)
method_names = ["Orthogonal", "D-optimal", "2D coprime"]
colors = ["#A8D357", "#E36363", "#75B1DC"]

fig, axes = plt.subplots(3, num_points, figsize=(14, 9), sharex=True, sharey=True)

print("\n========== 性能比较 (对齐评分 det(I+ΣJ)) ==========")
print(f"{'κ':<6} {'Orthogonal':>18} {'Global optimal':>18} {'2D coprime':>18}")
print("-" * 70)

for col, a in enumerate(a_vals):
    cond_num = a**2
    Lambda = np.array([[a, 0], [0, 1/a]])
    Sigma = R @ Lambda @ R.T

    # 计算所有组合的评分
    scores = np.zeros(n_combos)
    for i in range(n_combos):
        J = np.array([[J11_arr[i], J12_arr[i]],
                      [J12_arr[i], J22_arr[i]]])
        scores[i] = alignment_score(J, Sigma)

    # 全局最优（所有非相邻组合）
    global_best_idx = np.argmax(scores)
    global_best_J = np.array([[J11_arr[global_best_idx], J12_arr[global_best_idx]],
                              [J12_arr[global_best_idx], J22_arr[global_best_idx]]])
    global_score = scores[global_best_idx]

    # 互质最优（gcd==1的子集）
    coprime_mask = gcd_flag_arr
    if np.any(coprime_mask):
        coprime_scores = scores[coprime_mask]
        coprime_best_local_idx = np.argmax(coprime_scores)
        coprime_indices = np.where(coprime_mask)[0]
        coprime_best_idx = coprime_indices[coprime_best_local_idx]
        coprime_best_J = np.array([[J11_arr[coprime_best_idx], J12_arr[coprime_best_idx]],
                                   [J12_arr[coprime_best_idx], J22_arr[coprime_best_idx]]])
        coprime_score = coprime_scores[coprime_best_local_idx]
    else:
        coprime_best_J = None
        coprime_score = -np.inf

    # 固定正交模式的评分
    fixed_score = alignment_score(J_fixed, Sigma)

    print(f"{cond_num:<6.1f} {fixed_score:18.4e} {global_score:18.4e} {coprime_score:18.4e}")

    # 绘制三行（新顺序：0-正交，1-全局最优，2-2D互质）
    for row in range(3):
        ax = axes[row, col]

        # 画 Sigma 椭圆（红色虚线）
        plot_ellipse(ax, Sigma, facecolor='none', edgecolor='red',
                     linestyle='--', label=r'$\Sigma_\theta$', scale=10)

        # 根据行选择 J 和对应分数
        if row == 0:          # 正交
            best_J = J_fixed
            score_val = fixed_score
            color = colors[0]
        elif row == 1:        # 全局最优
            best_J = global_best_J
            score_val = global_score
            color = colors[1]
        else:                 # 2D 互质
            if coprime_best_J is not None:
                best_J = coprime_best_J
                score_val = coprime_score
            else:
                best_J = None
                score_val = None
            color = colors[2]

        if best_J is not None:
            plot_ellipse(ax, best_J, facecolor=color, edgecolor='black',
                         linestyle='-', label=r'$J_{\mathrm{eq}}$', scale=10)

        
        if score_val is not None:
            ax.text(0.05, 0.05, r'$\mathcal{{\bar{{U}}}}(\mathcal{{P}}) = {:.1f}$'.format(score_val),
                    transform=ax.transAxes, fontsize=18,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 装饰子图
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_aspect('equal')

        if row == 0:
            ax.set_title(rf"$\kappa = {cond_num:.1f}$", fontsize=18)
        if col == 0:
            
            ax.set_ylabel(method_names[row], fontsize=18, rotation=90, labelpad=10)
        if col == 0:
            ax.legend(loc='upper left', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Ellipse_Comparison_Nonadjacent.png', dpi=2000)
plt.savefig('Ellipse.eps', format='eps')
plt.show()

print("\n绘图完成。每个子图左下角显示了该模式在当前Σ下的对齐评分 P = det(I+ΣJ)。")