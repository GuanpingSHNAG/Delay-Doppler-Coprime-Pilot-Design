import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import matplotlib.pyplot as plt

# =====================================
# 1. 基础函数
# =====================================
def F_matrix(N):
    n = np.arange(N).reshape((N,1))
    k = np.arange(N).reshape((1,N))
    omega = np.exp(-1j * 2*np.pi / N)
    return omega ** (n * k) / np.sqrt(N)

def XtoY(tau, v, N1, N2, p, sigma2, X, F):
    FH = F.conj().T
    D = N1 * N2
    A_big = np.zeros((D, p), dtype=complex)
    Y = np.zeros((D, p), dtype=complex)

    for i in range(p):
        tau_i = tau
        v_i = v
        TD = np.diag(np.exp(-1j * 2*np.pi * np.arange(N1) * tau_i / N1))

        for l in range(N2):
            gp = np.exp(1j * 2*np.pi * v_i * l / N2)
            Qil = gp * np.eye(N1)
            x_col = X[:, l]
            s = F @ (Qil @ (FH @ (TD @ x_col)))
            blk = slice(l * N1, (l + 1) * N1)
            A_big[blk, i] = s

        Y[:, i] = np.sqrt(1/2) * (np.random.randn(1) + 1j*np.random.randn(1)) * (A_big[:, i])+np.sqrt(sigma2/2) * (np.random.randn(D) + 1j*np.random.randn(D))

    return Y.sum(axis=1).reshape(N2, N1).T


def estimate_delay_doppler_2D(Y, X):
    M, N = Y.shape
    R_tau = np.zeros((M, N), dtype=complex)
    for n in range(N):
        R_tau[:, n] = np.fft.ifft(Y[:, n] * np.conj(X[:, n]))
    R_dd = np.fft.fft(R_tau, axis=1)
    R_abs = np.abs(R_dd)
    delay_idx, doppler_idx = np.unravel_index(np.argmax(R_abs), R_abs.shape)
    return delay_idx, doppler_idx


# =====================================
# 2. Kraskov 互信息估计
# =====================================
def mutual_information_ksg(X, Y, k=5):
    n = X.shape[0]
    XY = np.hstack([X, Y])
    nbrs_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(XY)
    distances, _ = nbrs_xy.kneighbors(XY)
    eps = distances[:, k]
    tiny = 1e-10

    nbrs_x = NearestNeighbors(metric='chebyshev').fit(X)
    nbrs_y = NearestNeighbors(metric='chebyshev').fit(Y)
    cnt_x = np.array([len(neigh) for neigh in nbrs_x.radius_neighbors(X, radius=eps - tiny, return_distance=False)])
    cnt_y = np.array([len(neigh) for neigh in nbrs_y.radius_neighbors(Y, radius=eps - tiny, return_distance=False)])
    nx, ny = cnt_x - 1, cnt_y - 1

    I = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return I


# =====================================
# 3. 信号配置 & 噪声实验
# =====================================
def configure_signals(M, N):
    """
    定义几种典型的发射信号配置 X。
    """
    configs = {}

    X = np.zeros((M, N), dtype=complex)
    X[:, :] = 1
    configs['Full Pilot'] = X

    X = np.zeros((M, N), dtype=complex)
    X[0, 0] = 1
    X[1, 0] = 1
    X[0, 1] = 1
    configs['00 01 10 Coprime'] = X
    
    X = np.zeros((M, N), dtype=complex)
    X[0, 0] = 1
    X[1, 1] = 1
    X[2, 2] = 1
    configs['00 11 22 Noprime'] = X

    X = np.zeros((M//4, N//4), dtype=complex)
    X[0, 0] = 1
    X[0, 1] = 1
    X[1, 0] = 1
    X = np.tile(X, (4, 4))
    configs['4*4 00 10 01 Coprime'] = X

    X = np.zeros((M//4, N//4), dtype=complex)
    X[0, 0] = 1
    X[1, 1] = 1
    X[2, 2] = 1
    X = np.tile(X, (4, 4))
    configs['4*4 00 11 22 Noprime'] = X



    return configs


# =====================================
# 4. Monte Carlo 实验主循环
# =====================================
def run_mi_vs_snr():
    p, M, N = 1, 128, 128
    F = F_matrix(M)
    configs = configure_signals(M, N)
    snr_list = np.arange(-30, 51, 10)
    n_trials = 100000
    k = 5

    results = {name: [] for name in configs.keys()}

    for name, X in configs.items():
        E = np.sum(np.abs(X)**2)
        X = X / np.sqrt(E) * np.sqrt(M*N)
        print(f"\nRunning config: {name}")

        for snr_db in snr_list:
            sigma2 = 10 ** (-snr_db / 10)
            true_pairs = np.zeros((n_trials, 2))
            est_pairs  = np.zeros((n_trials, 2), dtype=float)

            for i in range(n_trials):
                tau = np.random.uniform(0, M//2)
                v   = np.random.uniform(0, N//2)
                Y = XtoY(tau, v, M, N, p, sigma2, X, F)
                tau_hat, v_hat = estimate_delay_doppler_2D(Y, X)
                true_pairs[i] = [tau, v]
                est_pairs[i]  = [tau_hat, v_hat]

            I_est = mutual_information_ksg(true_pairs, est_pairs, k=k)
            results[name].append(I_est)
            print(f"  SNR={snr_db:>3} dB → I={I_est:.3f} nats")

    # =====================================
    # 5. 绘图
    # =====================================
    plt.figure(figsize=(8,5))
    for name, I_list in results.items():
        plt.plot(snr_list, I_list,'-o', label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mutual Information  I((τ,v);(τ̂,v̂)) [nats]")
    plt.title("Mutual Information vs SNR for Different Transmit Configurations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================
# 运行实验
# =====================================
if __name__ == "__main__":
    run_mi_vs_snr()
