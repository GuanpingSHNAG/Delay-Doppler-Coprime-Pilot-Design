import numpy as np
import math
import itertools
from functools import reduce
import copy
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 物理计算内核 (Physics Kernel)
# ==========================================
class SMICalculator:
    def __init__(self, Nc, Ns, sigma2=0.01, prior_cov=None):
        """
        初始化 SMI 计算器
        Nc: 子载波数
        Ns: 符号数
        sigma2: 噪声方差
        prior_cov: 先验协方差矩阵 (描述目标参数的不确定性)
        """
        self.Nc = Nc
        self.Ns = Ns
        # 修正系数：根据包含初始未知相位的完整 Jacobian 矩阵推导，常数系数应为 4*pi^2
        self.coeff = (4 * np.pi**2) / (10**(sigma2/10))
        self.df = 1.0 / Nc
        self.Ts = 1.0 / Ns
        # 默认强相关先验，与论文设置一致
        self.Sigma_theta = prior_cov if prior_cov is not None else np.array([[1.25, 1], [1, 10]])

    def calc_fim(self, locs_array):
        """
        计算 Fisher Information Matrix (FIM)
        【已修正】：从绝对索引平方和，更正为消除参考点漂移的方差/协方差形式
        """
        P = len(locs_array)
        if P == 0: return np.zeros((2, 2))
        
        # 提取坐标 (支持连续浮点数或离散整数)
        m, n = locs_array[:, 0], locs_array[:, 1]
        
        # 计算基于中心矩的统计量 (总体方差和协方差)
        # 这等价于 Schur 补投影操作，消除了未知初始相位的干扰
        mean_m = np.mean(m)
        mean_n = np.mean(n)
        
        var_m = np.mean(m**2) - mean_m**2
        var_n = np.mean(n**2) - mean_n**2
        cov_mn = np.mean(m * n) - mean_m * mean_n
        
        # J 矩阵的各项，注意乘以导频总数 P
        J11 = self.coeff * P * var_m * (self.df**2)
        J22 = self.coeff * P * var_n * (self.Ts**2)
        J12 = -self.coeff * P * cov_mn * (self.df * self.Ts)
        
        return np.array([[J11, J12], [J12, J22]])

    def calc_smi(self, J):
        """计算 Sensing Mutual Information (SMI)"""
        try:
            mat = np.eye(2) + self.Sigma_theta @ J
            det_val = np.linalg.det(mat)
            # 加上数值保护，防止行列式非正
            return 0.5 * np.log(det_val) if det_val > 1e-12 else -10.0
        except:
            return -10.0

    def get_numerical_gradient(self, locs_continuous, eps=1e-3):
        """
        计算 SMI 关于导频位置的数值梯度 (Numerical Gradient)
        用于指导连续空间的探索方向
        """
        grad = np.zeros_like(locs_continuous)
        base_smi = self.calc_smi(self.calc_fim(locs_continuous))
        
        for i in range(len(locs_continuous)):
            for axis in [0, 1]: # 0: freq(m), 1: time(n)
                temp = locs_continuous.copy()
                temp[i, axis] += eps
                grad[i, axis] = (self.calc_smi(self.calc_fim(temp)) - base_smi) / eps
        return grad

# ==========================================
# 2. 互质约束检测 (Constraint Checker)
# ==========================================
def get_system_gcd(pilot_locs, Nc, Ns):
    """
    基于广义贝祖定理 (Theorem 1) 检测导频图案的全局互质性
    返回: 系统 GCD 值 (1 表示互质/无模糊)
    """
    diff_vectors = set()
    n_p = len(pilot_locs)
    
    # 鲁棒取整，确保输入是整数元组
    int_locs = []
    for p in pilot_locs:
        m = p[0] if isinstance(p, (list, tuple, np.ndarray)) else p
        n = p[1] if isinstance(p, (list, tuple, np.ndarray)) else p
        int_locs.append((int(round(m)), int(round(n))))
    
    # 1. 计算所有差分向量
    for i in range(n_p):
        for j in range(i + 1, n_p):
            u = int_locs[i][0] - int_locs[j][0]
            v = int_locs[i][1] - int_locs[j][1]
            diff_vectors.add((u, v))
            
    if not diff_vectors: return Nc * Ns
    
    # 2. 收集关键行列式项
    terms = {Nc * Ns}
    for u, v in diff_vectors:
        terms.add(abs(v * Nc))
        terms.add(abs(u * Ns))
        
    # 3. 差分向量之间的叉乘
    for i, (u1, v1) in enumerate(diff_vectors):
        for u2, v2 in list(diff_vectors)[i+1:]:
            terms.add(abs(u1 * v2 - u2 * v1))
            
    return reduce(math.gcd, list(terms))

# ==========================================
#环面曼哈顿距离与相邻性检查
# ==========================================
def toroidal_manhattan_dist(p1, p2, M, N):
    """环面曼哈顿距离（考虑周期性边界）"""
    m1, n1 = p1
    m2, n2 = p2
    dm = min(abs(m1 - m2), M - abs(m1 - m2))
    dn = min(abs(n1 - n2), N - abs(n1 - n2))
    return dm + dn

def is_non_adjacent(pilot_locs, M, N):
    """检查所有导频对：环面曼哈顿距离必须 > 1 (即不相邻且不重叠)"""
    n = len(pilot_locs)
    for i in range(n):
        for j in range(i+1, n):
            if toroidal_manhattan_dist(pilot_locs[i], pilot_locs[j], M, N) <= 1:
                return False
    return True


# ==========================================
# 3. 离散单步修正算子 (Refinement Operator)
# ==========================================
def run_discrete_single_swap(start_locs, calculator, Nc, Ns):
    """
    执行【单次】最佳离散交换 (Single Best Swap)。
    加入了不相邻 (环面曼哈顿距离 > 1) 约束。
    """
    # 1. 量化
    current_locs = [ (int(round(x[0]))%Nc, int(round(x[1]))%Ns) for x in start_locs ]
    
    # 2. 强制修复 (如果起点违反互质或不相邻约束)
    if get_system_gcd(current_locs, Nc, Ns) != 1 or not is_non_adjacent(current_locs, Nc, Ns):
        for _ in range(200): # 增加迭代次数以应对更严苛的约束
            idx = np.random.randint(0, len(current_locs))
            current_locs[idx] = (np.random.randint(Nc), np.random.randint(Ns))
            # 必须同时满足无重叠、互质、且不相邻
            if len(set(current_locs)) == len(current_locs) and \
               get_system_gcd(current_locs, Nc, Ns) == 1 and \
               is_non_adjacent(current_locs, Nc, Ns):
                break
            
    current_smi = calculator.calc_smi(calculator.calc_fim(np.array(current_locs)))
    all_points = list(itertools.product(range(Nc), range(Ns)))
    
    best_locs = list(current_locs)
    best_smi = current_smi
    found_improvement = False
    candidates = [] 
    
    # 3. 遍历所有可能的单步移动
    for i in range(len(current_locs)):
        others = current_locs[:i] + current_locs[i+1:]
        
        for cand in all_points:
            if cand in others: continue
            temp = others + [cand]
            
            # 【双重约束拦截】：必须互质且不相邻
            if get_system_gcd(temp, Nc, Ns) == 1 and is_non_adjacent(temp, Nc, Ns):
                val = calculator.calc_smi(calculator.calc_fim(np.array(temp)))
                if val > current_smi:
                    candidates.append((val, i, cand))

    # 4. 执行增益最大的移动
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_move = candidates[0]
        best_smi = best_move[0]
        best_locs[best_move[1]] = best_move[2]
        found_improvement = True
        
    return best_locs, best_smi, found_improvement

# ==========================================
# 4. 主程序：梯度引导的交替优化 (Zig-Zag)
# ==========================================
def run_zigzag_optimization():
    # --- 参数设置 ---
    Nc, Ns = 10, 10      # 网格大小
    N_pilots = 5      # 导频数量
    calc = SMICalculator(Nc, Ns)
    
    # --- 初始化 ---
# --- 初始化 ---
    print("Initializing...")
    while True:
        idxs = np.random.choice(Nc*Ns, N_pilots, replace=False)
        init_pat = [(i//Ns, i%Ns) for i in idxs]
        
        # 必须同时满足互质且不相邻
        if get_system_gcd(init_pat, Nc, Ns) == 1 and is_non_adjacent(init_pat, Nc, Ns): 
            smi = calc.calc_smi(calc.calc_fim(np.array(init_pat)))
            if -5.0 < smi < 4.0:
                break
            
    current_discrete_pat = init_pat
    current_discrete_smi = calc.calc_smi(calc.calc_fim(np.array(init_pat)))
    
    # 数据容器 (用于绘图)
    segments = []      # 存储连续轨迹段
    discrete_points = [] # 存储离散解点
    
    global_step = 1
    discrete_points.append((global_step, current_discrete_smi))
    
    max_epochs = 20 # 迭代轮数
    
    print(f"Start Pattern: {init_pat}")
    print(f"Start SMI: {current_discrete_smi:.4f}\n")
    print(f"{'Epoch':<6} | {'Continuous Max':<15} | {'Discrete Refined':<15}")
    print("-" * 45)
    
    # --- 主循环 ---
    for epoch in range(max_epochs):
        # -------------------------------------------------
        # Phase 1: 连续梯度探索 (Continuous Exploration)
        # -------------------------------------------------
        cont_pat = np.array(current_discrete_pat, dtype=float)
        
        explore_steps = 2 # 每轮探索步数
        lr =  (0.95 * epoch) # 学习率衰减
        
        traj_y = [calc.calc_smi(calc.calc_fim(cont_pat))]
        traj_x = [global_step]
        
        for _ in range(explore_steps):
            grad = calc.get_numerical_gradient(cont_pat)
            cont_pat += lr * grad
            
            # 边界限制 (Box Constraint)
            cont_pat[:, 0] = np.clip(cont_pat[:, 0], 0, Nc-0.01)
            cont_pat[:, 1] = np.clip(cont_pat[:, 1], 0, Ns-0.01)
            
            smi = calc.calc_smi(calc.calc_fim(cont_pat))
            global_step += 1
            traj_x.append(global_step)
            traj_y.append(smi)
            
        segments.append((traj_x, traj_y, '#e74c3c')) # 记录红色轨迹
        
        # -------------------------------------------------
        # Phase 2: 投影与离散精修 (Projection & Refinement)
        # -------------------------------------------------
        # 1. 投影/量化
        # 替换后代码
        quantized_pat = project_to_nearest_coprime(cont_pat, Nc, Ns)
        # 起点已经是合法的了，直接继续做你的单步搜索
        new_discrete_pat, new_discrete_smi, improved = run_discrete_single_swap(quantized_pat, calc, Nc, Ns)
        
        # 记录跌落虚线
        drop_x = [traj_x[-1], traj_x[-1]] 
        drop_y = [traj_y[-1], new_discrete_smi]
        segments.append((drop_x, drop_y, 'gray_dashed'))
        
        current_discrete_pat = new_discrete_pat
        discrete_points.append((traj_x[-1], new_discrete_smi))
        
        print(f"{epoch+1:<6} | {traj_y[-1]:<15.4f} | {new_discrete_smi:<15.4f}")
        
        # 如果连续 3 轮没有离散提升，且已经是后期，则停止
        if not improved and epoch > 20:
            break

    return segments, discrete_points, current_discrete_pat, new_discrete_smi



import itertools

def project_to_nearest_coprime(cont_locs, Nc, Ns):
    """
    寻找距离最近、满足【互质】且【不相邻】约束的离散整数坐标。
    """
    n_p = len(cont_locs)
    base_locs = [(int(round(p[0])) % Nc, int(round(p[1])) % Ns) for p in cont_locs]
    
    # 理想情况：直接满足所有条件
    if is_non_adjacent(base_locs, Nc, Ns) and get_system_gcd(base_locs, Nc, Ns) == 1:
        return base_locs

    # 2. 九宫格搜索
    offsets = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    best_valid = None
    min_dist = float('inf')
    
    for moves in itertools.product(offsets, repeat=n_p):
        cand = [((base_locs[i][0] + moves[i][0]) % Nc, 
                 (base_locs[i][1] + moves[i][1]) % Ns) for i in range(n_p)]
        
        # 【拦截 1 & 2】：不仅要不重叠，还必须不相邻（<=1 会处理两者）
        if not is_non_adjacent(cand, Nc, Ns):
            continue
            
        # 【拦截 3】：互质条件
        if get_system_gcd(cand, Nc, Ns) == 1:
            dist = sum(abs(cont_locs[i][0] - cand[i][0]) + abs(cont_locs[i][1] - cand[i][1]) for i in range(n_p))
            if dist < min_dist:
                min_dist = dist
                best_valid = cand
                
    if best_valid is not None:
        return best_valid
        
    # 3. 极端兜底逻辑 (Fallback)：确保即便挤成一团也能散开
    safe_locs = []
    all_grid_points = [(m, n) for m in range(Nc) for n in range(Ns)]
    
    for i in range(n_p):
        # 排除掉距离已有确定点 <=1 的所有点
        available_points = [pt for pt in all_grid_points if all(toroidal_manhattan_dist(pt, exist_pt, Nc, Ns) > 1 for exist_pt in safe_locs)]
        
        # 按连续点距离排序
        available_points.sort(key=lambda pt: abs(cont_locs[i][0] - pt[0]) + abs(cont_locs[i][1] - pt[1]))
        
        for pt in available_points:
            temp_locs = safe_locs + [pt]
            # 如果是最后一个导频，必须测试整体互质
            if len(temp_locs) == n_p:
                if get_system_gcd(temp_locs, Nc, Ns) == 1:
                    safe_locs.append(pt)
                    break
            else:
                # 不是最后一个，只要上一步保证了不相邻即可先加入
                safe_locs.append(pt)
                break
                
    return safe_locs

# ==========================================
# 5. 执行与绘图
# ==========================================

if __name__ == "__main__":
    start_time = time.time()
    segs, dots, best_pattern, best_smi = run_zigzag_optimization()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print("\n" + "="*50)
    print("FINAL OPTIMIZATION RESULTS (Variance-based FIM)")
    print("="*50)
    print(f"Optimal Pilot Indices (m, n): {best_pattern}")
    print(f"Max Sensing Mutual Information: {best_smi:.6f} nats")
    print("="*50)
    print(f"Total Execution Time: {elapsed_time:.4f} seconds")
    print("="*50)

    # 全局字体放大设置
    plt.rcParams.update({
        "font.size": 16,          
        "axes.titlesize": 18,     
        "axes.labelsize": 15,     
        "xtick.labelsize": 15,    
        "ytick.labelsize": 15,    
        "legend.fontsize": 15     
    })

    plt.figure(figsize=(10, 6))
    
    # 绘制线段
    for x, y, style in segs:
        if style == 'gray_dashed':
            plt.plot(x, y, linestyle='--', color='gray', alpha=0.6, linewidth=1.5)
        else:
            plt.plot(x, y, color=style, linewidth=2, alpha=0.9)
    
    # 绘制离散解点
    dot_x = [d[0] for d in dots]
    dot_y = [d[1] for d in dots]
    plt.scatter(dot_x, dot_y, color='#2980b9', s=70, zorder=10, label='Discrete Solution (Coprime)')
    plt.plot(dot_x, dot_y, color='#2980b9', linestyle=':', alpha=0.5)

    # 自定义图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#e74c3c', lw=2.5, label='Continuous Gradient Exploration'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Constraint Projection'),
        Line2D([0], [0], color='#2980b9', linestyle=':', lw=1.5, label='Discrete Refinement Trend'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=8, label='Discrete Solution')
    ]
    
    plt.legend(handles=custom_lines, loc='lower right', frameon=True)
    plt.xlabel("Cumulative Optimization Steps")
    plt.ylabel("Sensing Mutual Information (nats)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("zigzag_optimization_result_variance.pdf", bbox_inches='tight', dpi=800)
    plt.show()