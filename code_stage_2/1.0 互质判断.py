import itertools
import numpy as np

def lattice_axes_bruteforce(vecs, R_init=4, R_max=20):
    """
    vecs: 形状 (N,2) 的 numpy 数组或 list，每行是一个差分向量 (vx, vy)
    R_init: 初始搜索整数系数范围 [-R, R]
    R_max:  最大搜索范围上限，防止死循环

    返回:
        (a, 0), (0, b)  —— 都是整型元组
    """

    vecs = np.asarray(vecs, dtype=int)
    N = vecs.shape[0]

    best_a = None
    best_b = None

    R = R_init
    while R <= R_max and (best_a is None or best_b is None):
        # 在 [-R, R]^N 里枚举所有整数系数组合
        for coeffs in itertools.product(range(-R, R + 1), repeat=N):
            if all(c == 0 for c in coeffs):
                continue

            coeffs_arr = np.array(coeffs, dtype=int)
            v = coeffs_arr @ vecs   # 线性组合，得到 (x, y)
            x, y = int(v[0]), int(v[1])

            # 找 (a, 0)
            if y == 0 and x != 0:
                ax = abs(x)
                if best_a is None or ax < best_a:
                    best_a = ax

            # 找 (0, b)
            if x == 0 and y != 0:
                by = abs(y)
                if best_b is None or by < best_b:
                    best_b = by

        # 如果还没同时找到 a 和 b，就扩大搜索范围
        if best_a is not None and best_b is not None:
            break
        R += 1

    if best_a is None or best_b is None:
        raise RuntimeError(
            f"在系数范围 [-{R_max}, {R_max}] 内没找到轴向向量，"
            f"可以尝试增大 R_max 或检查输入是否真是 2D 满秩 lattice。"
        )

    return (best_a, 0), (0, best_b)


if __name__ == "__main__":
    vecs = np.array([
        [0, 1],
        [1, 7],
        [6,7],
        [6,1]
    ])

    print("输入 vecs =")
    print(vecs)
    (a0, b0) = lattice_axes_bruteforce(vecs, R_init=3, R_max=8)
    print("最小 (a,0), (0,b) = ", (a0, b0))
