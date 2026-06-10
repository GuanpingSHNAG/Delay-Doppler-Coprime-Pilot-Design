import numpy as np
from math import comb, factorial, log

def H_unordered(p: int, N: int):
    """
    计算 p 个接口、N 种符号时，
    不区分顺序下接收到的符号组合的信息量 H(p,N)。

    参数
    ----
    p : int          接口数
    N : int          每个接口符号数
    logbase : str    'nat' 返回 nats，'bit' 返回 bits

    返回
    ----
    H : float        信息量
    """
    # 先算 E[ ln(K!) ] ，K~Binomial(p, 1/N)
    q = 1.0 / N
    E_lnKfact = 0.0
    for k in range(p + 1):
        prob = comb(p, k) * (q**k) * ((1 - q)**(p - k))
        E_lnKfact += prob * log(factorial(k))

    # 主公式： H = p ln N - ln p! + N * E[ln(K!)]
    H = p * log(N) - log(factorial(p)) + N * E_lnKfact
    
    return H

def H_my(p: int, N: int):
    """
    计算 p 个接口、N 种符号时，
    不区分顺序下接收到的符号组合的信息量 H(p,N)。

    参数
    ----
    p : int          接口数
    N : int          每个接口符号数
    logbase : str    'nat' 返回 nats，'bit' 返回 bits

    返回
    ----
    H : float        信息量
    """
    # 先算 E[ ln(K!) ] ，K~Binomial(p, 1/N)
    q = 1.0 / N
    E_lnKfact = 0.0
    for k in range(p + 1):
        prob = comb(p, k) * (q**k) * ((1 - q)**(p - k))
        factor=comb(p, k)*factorial(p-k)
        E_lnKfact += prob * log(factor)

    # 主公式： H = p ln N - ln p! + N * E[ln(K!)]
    H = p * log(N) - E_lnKfact
    
    return H

if __name__ == "__main__":
    # 示例
    path_num=2
    cases_num=8
    '''
    Time Delay cases * Doppler Cases
    '''
    test_cases = [(path_num, cases_num)]
    for p, N in test_cases:
        H_nat = H_unordered(p, N)
        H_m = H_my(p, N)
        print(f"p={p:2d}, N={N:2d} ->\nH = {H_nat:.6f} nats\n")
        print(2*log(cases_num)-log(2)+1/cases_num*log(2))
        print(f"H_my = {H_m:.6f} nats\n")
