import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import matplotlib.colors as mcolors  # <--- 引入颜色管理模块


# ===================== 全局设置：Times New Roman =====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式也用 Times 风格

# ==== 1. 自定义配色区域 ====
# 这里的颜色会按顺序从低值(谷底)过渡到高值(山峰)
# 你可以在这里填入任何你喜欢的色号，数量不限
MY_COLORS = [
    "#5A7E95",  # 亮蓝
    "#7B9BAF",  # 深蓝 (谷底 - 极小值)
    "#9DB7C2",  # 亮蓝
    "#BECAC3",  # 深蓝 (谷底 - 极小值)
    "#DACDBE",  # 亮蓝
    "#E6C9B7",  # 深蓝 (谷底 - 极小值)
    "#E1B2A9",  # 亮蓝
    "#D99A9C",  # 青色
    "#CF848A",  # 黄色  # 红色 (山峰 - 极大值)
    "#E16E77",  # 红色 (山峰 - 极大值)
    "#C0565F"  # 红色 (山峰 - 极大值)
]

# 创建连续的 Colormap 对象
# N=256 表示将渐变切分为 256 个等级，保证视觉上的连续平滑
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("my_custom_theme", MY_COLORS, N=256)


def compute_psr_linear(F_shift, mainlobe_halfwidth=1):
    """ (保持不变) 计算 PSR """
    mag = np.abs(F_shift)
    power = mag ** 2
    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    i0, j0 = peak_idx
    P_peak = power[i0, j0]
    M, N = power.shape
    mask = np.ones_like(power, dtype=bool)
    i_min = max(i0 - mainlobe_halfwidth, 0)
    i_max = min(i0 + mainlobe_halfwidth, M - 1)
    j_min = max(j0 - mainlobe_halfwidth, 0)
    j_max = min(j0 + mainlobe_halfwidth, N - 1)
    mask[i_min:i_max+1, j_min:j_max+1] = False
    sidelobes_power = power[mask]
    P_sidelobe_max = sidelobes_power.max() if sidelobes_power.size > 0 else 0
    return P_peak / (P_sidelobe_max + 1e-20), 10 * np.log10(P_peak / (P_sidelobe_max + 1e-20) + 1e-20)

def fft2_and_plot_3d(x2d, use_db=False, cmap_obj=None, interp_factor=5.0):
    """
    参数 cmap_obj: 接收我们自定义的 colormap 对象
    功能：2D FFT + **能量归一化** + 3D 绘图
    """
    # 1. 计算 2D FFT
    F = np.fft.fft2(x2d)
    F_shift = np.fft.fftshift(F)  # 把零频移到中心 ✅ 修复原代码没 fftshift 的问题
    mag = np.abs(F_shift)

    # ===================== 核心：归一化能量 =====================

    # ==========================================================
    energy_total = np.sum(mag ** 2)  # 总能量
    mag_norm = mag / np.sqrt(energy_total)  # 能量归一化（保证总能量=1）

    if use_db:
        value_for_plot = 20 * np.log10(mag_norm)
        #z_label = 'Normalized |F|'
    else:
        value_for_plot = mag_norm
        #z_label = 'Normalized |F|'

    value_s = np.sum(value_for_plot ** 2)  # 总能量
    value_for_plot = value_for_plot / np.sqrt(value_s)  # 能量归一化（保证总能量=1）
    value_for_plot[:,:] = np.roll(np.roll(value_for_plot, shift=7, axis=0), shift=6, axis=1)

    # 2. 插值平滑
    plot_data_smooth = zoom(value_for_plot, interp_factor, order=3)

    # 构造坐标
    M, N = x2d.shape
    M_smooth, N_smooth = plot_data_smooth.shape
    fy = np.fft.fftfreq(M)
    fx = np.fft.fftfreq(N)
    fy_smooth = np.linspace(fy.min(), fy.max(), M_smooth)
    fx_smooth = np.linspace(0, 2*fx.max(), N_smooth)
    FX_smooth, FY_smooth = np.meshgrid(fx_smooth, fy_smooth)

    # 3. 绘图
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        FX_smooth, FY_smooth, plot_data_smooth,
        rstride=1, cstride=1,
        cmap=cmap_obj,
        linewidth=1,
        antialiased=True,
        shade=True,
    )
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Normalized delay',fontsize=14)
    ax.set_ylabel('Normalized Doppler',fontsize=14)
    plt.show()

    return F_shift

if __name__ == "__main__":

    

    X = np.zeros((8, 8), dtype=complex)
    #FIM最优
    X[1, 0] = 1
    X[0, 7] = 1
    X[7, 0] = 1
    X[6, 7] = 1
    
    # #coprime 最优
    # X[0, 1] = 1
    # X[1, 7] = 1
    # X[6, 7] = 1
    # X[7, 0] = 1

    #X = np.ones((8, 8), dtype=complex)
    

    # #正交互质
    # X[0, 0] = 1
    # X[3, 5] = 1
    # X[0, 5] = 1
    # X[3, 0] = 1


    # #均匀
    # X[0, 4] = 1
    # X[4, 4] = 1
    # X[4, 0] = 1
    # X[0, 0] = 1

    #X = np.ones((8, 8), dtype=complex)
    E = np.sum(np.abs(X)**2)
    X = X/ np.sqrt(E)
    #X = np.tile(X, (6,6))  # 8×8 → 128×128
    F_shift = fft2_and_plot_3d(X, cmap_obj=CUSTOM_CMAP, interp_factor=3)
    # ==== 计算并打印 PSR 值 ====
    psr_linear, psr_db = compute_psr_linear(F_shift, mainlobe_halfwidth=1)
    print(f"峰值旁瓣比 PSR (线性): {psr_linear:.4f}")
    print(f"峰值旁瓣比 PSR (dB): {psr_db:.2f} dB")