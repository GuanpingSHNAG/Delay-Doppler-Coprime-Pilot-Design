import numpy as np
import itertools
import math
from functools import reduce
import time
import csv

# ==========================================
# 1. 系统 GCD 检测器
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


# ==========================================
# 2. 环面曼哈顿距离与相邻性检查
# ==========================================
def toroidal_manhattan_dist(p1, p2, M, N):
    """环面曼哈顿距离（考虑周期性边界）"""
    m1, n1 = p1
    m2, n2 = p2
    dm = min(abs(m1 - m2), M - abs(m1 - m2))
    dn = min(abs(n1 - n2), N - abs(n1 - n2))
    return dm + dn

def is_non_adjacent(pilot_locs, M, N):
    """检查所有导频对：环面曼哈顿距离是否都不等于1"""
    n = len(pilot_locs)
    for i in range(n):
        for j in range(i+1, n):
            if toroidal_manhattan_dist(pilot_locs[i], pilot_locs[j], M, N) == 1:
                return False
    return True


# ==========================================
# 3. 生成所有满足“不相邻”的4导频组合（全网格）
# ==========================================
M, N = 8, 8
n_pilots = 4
coeff = 4 * np.pi**2
df = 1.0 / M
Ts = 1.0 / N

all_indices = np.arange(M * N)
print("生成所有 C(64,4) 组合...")
combos_raw = list(itertools.combinations(all_indices, n_pilots))
print(f"总组合数: {len(combos_raw)}")

valid_combos = []          # 存储坐标列表 [(m,n),...]
valid_combos_indices = []  # 存储线性索引列表
print("过滤环面不相邻的组合...")
t0 = time.time()
for combo in combos_raw:
    locs = [(idx // N, idx % N) for idx in combo]
    if is_non_adjacent(locs, M, N):
        valid_combos.append(locs)
        valid_combos_indices.append(list(combo))
print(f"满足不相邻条件的组合数: {len(valid_combos)}，耗时 {time.time()-t0:.2f} 秒")

n_combos = len(valid_combos)
if n_combos == 0:
    print("没有符合条件的组合，退出。")
    exit()

combos_arr = np.array(valid_combos_indices, dtype=int)  # (n_combos, 4)
m_coords = combos_arr // N
n_coords = combos_arr % N

# ==========================================
# 4. 预计算 J 参数和 GCD 标志
# ==========================================
print("计算每个组合的统计量...")
mean_m = np.mean(m_coords, axis=1, keepdims=True)
mean_n = np.mean(n_coords, axis=1, keepdims=True)
var_m = np.mean(m_coords**2, axis=1) - mean_m.squeeze()**2
var_n = np.mean(n_coords**2, axis=1) - mean_n.squeeze()**2
cov_mn = np.mean(m_coords * n_coords, axis=1) - (mean_m.squeeze() * mean_n.squeeze())

J11_arr = coeff * n_pilots * var_m * (df**2)
J22_arr = coeff * n_pilots * var_n * (Ts**2)
J12_arr = -coeff * n_pilots * cov_mn * (df * Ts)

print("计算系统 GCD 标志...")
t0 = time.time()
gcd_flag_arr = np.zeros(n_combos, dtype=bool)  # True 表示互质 (gcd==1)
for idx in range(n_combos):
    locs = valid_combos[idx]
    if get_system_gcd(locs, M, N) == 1:
        gcd_flag_arr[idx] = True
print(f"完成，耗时 {time.time()-t0:.2f} 秒")

# ==========================================
# 5. 遍历参数，寻找最优且非互质的模式
# ==========================================
lambda_max_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
cond_num_list = [1, 2, 3, 5, 8, 10]
angle_deg_list = np.arange(0, 91, 10)

results = []

print("\n开始遍历参数组合...")
total_params = len(lambda_max_list) * len(cond_num_list) * len(angle_deg_list)
param_idx = 0

for lam_max in lambda_max_list:
    for cond_num in cond_num_list:
        lam_min = lam_max / cond_num
        for angle_deg in angle_deg_list:
            param_idx += 1
            if param_idx % 50 == 0:
                print(f"进度: {param_idx}/{total_params}")

            angle_rad = np.deg2rad(angle_deg)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                          [np.sin(angle_rad),  np.cos(angle_rad)]])
            Sigma_theta = R @ np.diag([lam_max, lam_min]) @ R.T
            S11, S12, S22 = Sigma_theta[0,0], Sigma_theta[0,1], Sigma_theta[1,1]

            A11 = S11 * J11_arr + S12 * J12_arr
            A22 = S12 * J12_arr + S22 * J22_arr
            A12 = S11 * J12_arr + S12 * J22_arr
            A21 = S12 * J11_arr + S22 * J12_arr
            det_arr = (1 + A11) * (1 + A22) - A12 * A21

            best_idx = np.argmax(det_arr)
            best_det = det_arr[best_idx]

            if not gcd_flag_arr[best_idx]:   # 非互质
                best_locs = valid_combos[best_idx]
                results.append({
                    'lambda_max': lam_max,
                    'cond_num': cond_num,
                    'angle_deg': angle_deg,
                    'locs': best_locs,
                    'det': best_det,
                })

# ==========================================
# 6. 输出结果
# ==========================================
print(f"\n遍历完成。共找到 {len(results)} 个非互质最优模式 (环面不相邻约束)。")
if len(results) > 0:
    print("\n非互质最优导频模式汇总：")
    print("=" * 80)
    for r in results:
        print(f"λ_max={r['lambda_max']}, cond={r['cond_num']}, angle={r['angle_deg']}° -> 位置: {r['locs']} (det={r['det']:.4e})")

    with open('non_coprime_optimal_patterns_no_adjacent_toroidal.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda_max', 'cond_num', 'angle_deg', 'pilot_locations', 'det_value'])
        for r in results:
            locs_str = ';'.join([f"({m},{n})" for m, n in r['locs']])
            writer.writerow([r['lambda_max'], r['cond_num'], r['angle_deg'], locs_str, r['det']])
    print("\n结果已保存至 non_coprime_optimal_patterns_no_adjacent_toroidal.csv")
else:
    print("没有发现任何非互质的最优模式。")