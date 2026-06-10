import numpy as np
import itertools
import math
from functools import reduce
import time

# ==========================================
# 1. 系统 GCD 检测器（2D-coprime）
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
# 2. Separable coprime 检测器（仅适用于 4 个导频且构成 2×2 矩形）
# ==========================================
def is_separable_coprime(pilot_locs, Nc, Ns):
    int_locs = [(int(round(p[0])), int(round(p[1]))) for p in pilot_locs]
    m_vals = sorted(set([p[0] for p in int_locs]))
    n_vals = sorted(set([p[1] for p in int_locs]))
    # 检查是否为完整的矩形
    if len(int_locs) != len(m_vals) * len(n_vals):
        return False
    # 既然都是组合生成的，且所有点都在集合中，数量对就自动是矩形
    # 计算 M_set 中所有差值，取 gcd
    if len(m_vals) == 1:
        gcd_m = 0  # 单一值，差值为0，gcd(0, Nc) = Nc != 1 通常不是互质，所以直接返回 False？
        # 实际上只有一个行索引时，无法解析延迟模糊，除非 Nc=1，一般我们要求至少两个不同索引。
        # 所以如果 len(m_vals)==1，则 gcd_m = 0, math.gcd(0, Nc) = Nc，不等于1，除非 Nc=1。
        # 但也可认为不满足互质，所以返回 False。
    else:
        m_diffs = [abs(m_vals[i] - m_vals[j]) for i in range(len(m_vals)) for j in range(i+1, len(m_vals))]
        gcd_m = reduce(math.gcd, m_diffs)
    if len(n_vals) == 1:
        gcd_n = 0
    else:
        n_diffs = [abs(n_vals[i] - n_vals[j]) for i in range(len(n_vals)) for j in range(i+1, len(n_vals))]
        gcd_n = reduce(math.gcd, n_diffs)
    return math.gcd(gcd_m, Nc) == 1 and math.gcd(gcd_n, Ns) == 1
# ==========================================
# 3. 环面不相邻约束
# ==========================================
def toroidal_manhattan_dist(p1, p2, M, N):
    m1, n1 = p1
    m2, n2 = p2
    dm = min(abs(m1 - m2), M - abs(m1 - m2))
    dn = min(abs(n1 - n2), N - abs(n1 - n2))
    return dm + dn

def is_adjacent(p1, p2, M, N):
    return toroidal_manhattan_dist(p1, p2, M, N) == 1

def can_add(pilot_set, new_point, M, N):
    """检查 new_point 是否与 pilot_set 中所有点都不相邻（环面）"""
    for p in pilot_set:
        if is_adjacent(p, new_point, M, N):
            return False
    return True

# ==========================================
# 4. 递归回溯生成所有满足不相邻约束的组合
# ==========================================
def gen_non_adjacent_combinations(M, N, Np):
    """生成所有大小为 Np 的点集，任意两点环面曼哈顿距离 >= 2"""
    total_cells = M * N
    # 预先生成每个点的邻居列表（曼哈顿距离=1，环面）
    neighbors = [[] for _ in range(total_cells)]
    for idx in range(total_cells):
        m, n = divmod(idx, N)
        # 四个方向（环面）
        for dm, dn in [(-1,0),(1,0),(0,-1),(0,1)]:
            nm = (m + dm) % M
            nn = (n + dn) % N
            nidx = nm * N + nn
            neighbors[idx].append(nidx)
    
    # 用于快速排除已被禁止的点
    # 搜索顺序：按索引递增，保证组合无重复（组合而非排列）
    all_combos = []      # 存储每个组合的坐标列表
    
    def backtrack(start_idx, current_set_idx, current_coords):
        if len(current_coords) == Np:
            # 保存当前组合（深拷贝）
            all_combos.append(list(current_coords))
            return
        # 剩余需要选择的点数
        remaining = Np - len(current_coords)
        # 从 start_idx 开始尝试，避免重复
        for idx in range(start_idx, total_cells):
            # 剪枝：如果剩余格子不够，提前退出
            if total_cells - idx < remaining:
                break
            # 检查当前点是否与已选点冲突
            m, n = divmod(idx, N)
            conflict = False
            for (pm, pn) in current_coords:
                if is_adjacent((m,n), (pm,pn), M, N):
                    conflict = True
                    break
            if conflict:
                continue
            # 可选，加入
            current_coords.append((m,n))
            backtrack(idx + 1, current_set_idx + [idx], current_coords)
            current_coords.pop()
    
    backtrack(0, [], [])
    return all_combos

# ==========================================
# 5. 主程序：遍历所有尺寸和 Np，统计并打印
# ==========================================
def main():
    sizes = [4, 6, 8, 10, 12, 14]
    np_values = [4, 6, 8, 10]
    
    print("M,N | Np | 总组合数 | 2D-coprime 数 | separable coprime 数 (仅Np=4)")
    print("-" * 80)
    
    for M in sizes:
        N = M   # 方阵
        for Np in np_values:
            if Np > M*N:
                print(f"{M}x{M} | {Np:2} | 组合数=0 (Np > 网格大小) | 0 | 0")
                continue
            print(f"正在处理 {M}x{M}, Np={Np} ...", end='', flush=True)
            t0 = time.time()
            combos = gen_non_adjacent_combinations(M, N, Np)
            total = len(combos)
            count_2d = 0
            count_sep = 0
            for coords in combos:
                if get_system_gcd(coords, M, N) == 1:
                    count_2d += 1
                if Np == 4 and is_separable_coprime(coords, M, N):
                    count_sep += 1
            elapsed = time.time() - t0
            print(f" 完成 耗时 {elapsed:.2f}s")
           
            print(f"{M}x{M} | {Np:2} | {total:10} | {count_2d:12} | {count_sep:20}")

if __name__ == "__main__":
    main()