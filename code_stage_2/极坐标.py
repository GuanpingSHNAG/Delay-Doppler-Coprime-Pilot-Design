import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 仿真参数定义 ---
N_SAMPLES = 5000000  # 蒙特卡洛样本数量，为了在合理时间内完成，适当调整
# 统一使用极坐标量化
DELTA_R = 0.01               # 极坐标的幅度 bin 宽度
DELTA_THETA = 0.02 * np.pi     # 极坐标的角度 bin 宽度 (相当于把圆周分为100份)
BINS_A = 1000                 # 用于计算条件熵时，对 a 进行分箱的数量

# 定义 A 的幅度范围 (dB)
A_db_list = np.linspace(-50, 20, 15)
# 将 dB 转换为线性幅度
A_mag_list = 10**(A_db_list / 20.0)
# 保持 A 的相位不变 (设为0)
phase_A = 0

# 初始化用于存储结果的列表
entropy_g_list = []
entropy_g_given_a_list = []
mutual_information_list = []

print("开始仿真互信息 I(g; a) 随 |A| 的变化 (使用极坐标量化)...")
print("-" * 30)

# --- 2. 主循环，遍历所有 |A| 的值 ---
for i, A_mag in enumerate(A_mag_list):
    # 根据当前幅度和固定相位构造复数 A
    A = A_mag * np.exp(1j * phase_A)
    
    print(f"正在计算 |A| = {A_db_list[i]:.1f} dB (进度: {i+1}/{len(A_mag_list)})...")
    
    # --- 2a. 生成数据 ---
    a = np.random.uniform(0, 2 * np.pi, N_SAMPLES)
    b = np.random.uniform(0, 2 * np.pi, N_SAMPLES)
    g = A * np.exp(1j * a) + np.exp(1j * b)
    g_mag = np.abs(g)
    g_angle = np.angle(g) # 值域为 [-π, π]

    # --- 2b. 计算 H(g) ---
    mag_min, mag_max = 0,100
    bins_mag = np.arange(mag_min, mag_max, DELTA_R)
    bins_angle = np.arange(-np.pi, np.pi + DELTA_THETA, DELTA_THETA)
    
    freqs_g, _, _ = np.histogram2d(g_mag, g_angle, bins=[bins_mag, bins_angle])
    probs_g = freqs_g / N_SAMPLES
    non_zero_probs_g = probs_g[probs_g > 0]
    entropy_g = -np.sum(non_zero_probs_g * np.log(non_zero_probs_g))
    entropy_g_list.append(entropy_g)
    
    # --- 2b. 计算 H(g|a) ---
    g = A * np.exp(0) + np.exp(1j * b)
    g_mag = np.abs(g)
    g_angle = np.angle(g) # 值域为 [-π, π]
    mag_min, mag_max = 0,20
    bins_mag = np.arange(mag_min, mag_max, DELTA_R)
    bins_angle = np.arange(-np.pi, np.pi + DELTA_THETA, DELTA_THETA)
    freqs_ga, _, _ = np.histogram2d(g_mag, g_angle, bins=[bins_mag, bins_angle])
    probs_ga = freqs_ga / N_SAMPLES
    non_zero_probs_ga = probs_ga[probs_ga > 0]
    entropy_ga = -np.sum(non_zero_probs_ga * np.log(non_zero_probs_ga))
    entropy_g_given_a_list.append(entropy_ga)








    # # --- 2c. 计算 H(g|a) ---
    # bins_a = np.linspace(0, 2 * np.pi, BINS_A + 1)
    # a_indices = np.digitize(a, bins_a)
    # df = pd.DataFrame({'a_idx': a_indices, 'g_mag': g_mag, 'g_angle': g_angle})
    
    # h_g_given_a = 0.0
    # # 遍历每个 'a' 的箱子
    # for _, group in df.groupby('a_idx'):
    #     n_group = len(group)
    #     if n_group < 2: continue
        
    #     p_a_k = n_group / N_SAMPLES
        
    #     # 对该组内的 g 进行极坐标量化并计算其熵
    #     g_mag_subset = group['g_mag'].values
    #     g_angle_subset = group['g_angle'].values

    #     mag_min_s, mag_max_s = g_mag_subset.min(), g_mag_subset.max()
    #     bins_mag_s = np.arange(mag_min_s, mag_max_s, DELTA_R)
    #     # 角度范围固定
    #     bins_angle_s = np.arange(-np.pi, np.pi + DELTA_THETA, DELTA_THETA)
        
    #     if len(bins_mag_s) < 2: continue

    #     freqs_subset, _, _ = np.histogram2d(g_mag_subset, g_angle_subset, bins=[bins_mag_s, bins_angle_s])
    #     probs_subset = freqs_subset / n_group
    #     non_zero_probs_subset = probs_subset[probs_subset > 0]
    #     h_subset = -np.sum(non_zero_probs_subset * np.log(non_zero_probs_subset))
        
    #     h_g_given_a += p_a_k * h_subset
    
    # entropy_g_given_a_list.append(h_g_given_a)

    # --- 2d. 计算互信息 I(g; a) ---
    mutual_information = entropy_g - entropy_ga
    mutual_information_list.append(mutual_information)

print("-" * 30)
print("仿真结束。")

# --- 3. 绘制结果图 ---
print("正在生成结果图...")
# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 8))

plt.plot(A_db_list, mutual_information_list, 'o-', label='互信息 I(g; a) (极坐标量化)')
plt.plot(A_db_list, entropy_g_list, 's--', label='H(g) (极坐标量化)', alpha=0.7)
plt.plot(A_db_list, entropy_g_given_a_list, '^-.', label='H(g|a) (极坐标量化)', alpha=0.7)

plt.title('互信息 I(g; a) 随 |A| 的变化趋势 (极坐标量化)', fontsize=16)
plt.xlabel('|A| 的幅度 (dB)', fontsize=14)
plt.ylabel('信息量 (nats)', fontsize=14)
plt.axvline(x=0, color='r', linestyle=':', label='|A| = 1 (0 dB)')
plt.legend(fontsize=12)
plt.grid(True, which='both')
plt.show()

