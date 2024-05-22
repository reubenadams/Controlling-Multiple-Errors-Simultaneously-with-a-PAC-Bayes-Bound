import torch as t
from mpmath import *  # Note this includes infinity and log
# from pac_bayes.bisecting import bisect as pb_bisect
from scipy.optimize import bisect as sp_bisect

mp.dps = 1000


# def kl_divergence(mean1, var1, mean2, var2):
#     assert mean1.shape == var1.shape == mean2.shape == var2.shape
#     term1 = t.log(var2 / var1).sum()
#     term2 = ((var1 + (mean1 - mean2) ** 2) / var2).sum()
#     term3 = -mean1.numel()
#     return (term1 + term2 + term3) / 2


def kl_divergence_scale(mean1, scale1, mean2, scale2):
    # assert mean1.shape == scale1.shape == mean2.shape == scale2.shape
    term1 = (scale2 - scale1).sum()
    term2 = t.exp(2 * (scale1 - scale2)).sum() / 2
    # term3 = ((mean1 - mean2) ** 2 / (2 * t.exp(2 * scale2))).sum()
    # Hoped this version would more stable and avoid the KL being negative, but no cigar - same results
    term3 = (((mean1 - mean2) ** 2) * t.exp(-2 * scale2)).sum() / 2
    term4 = -mean1.numel() / 2
    return term1 + term2 + term3 + term4


def kl_component(q_i, p_i):
    assert 0 <= q_i <= 1 and 0 <= p_i <= 1
    if q_i == 0:
        return 0
    if p_i == 0:
        return t.inf  # TODO: Revert this to inf
    if isinstance(q_i, mpf) or isinstance(p_i, mpf):
        return q_i * log(q_i / p_i)
    return q_i * t.log(q_i / p_i)


def kl_scalars(q, p):
    return kl_component(q, p) + kl_component(1 - q, 1 - p)


def kl_vectors(q_vec, p_vec):
    return sum([kl_component(q_i, p_i) for q_i, p_i in zip(q_vec, p_vec)])


def kl_scalars_inverse(
    q,
    B,
    x_tol=mp.mpf("0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"),
    y_tol=mp.mpf("0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"),
):
    if B == 0:
        return q
    if q == 0:
        return 1 - exp(-B)  # Easy pen and paper check
    if q == 1:
        return 1
    p_max = 1 - x_tol / 2
    f = lambda p: kl_scalars(q, p) - B
    if f(p_max) < mp.mpf("0"):
        print("No bound on p")
        return 1
    interval = (q, p_max)
    root = pb_bisect(f, interval, x_tol=x_tol, y_tol=y_tol)
    return float(root)


def kl_scalars_inverse_scipy_version(q, B, x_tol):
    if B == 0:
        return q
    if q == 0:
        return 1 - exp(-B)  # Easy pen and paper check
    if q == 1:
        return 1
    p_max = 1 - x_tol / 2
    assert q < p_max < 1
    f = lambda p: kl_scalars(q, p) - B
    if f(p_max) < 0:
        print("No bound on p")
        return 1
    # interval = (q, p_max)
    root = sp_bisect(f=f, a=q, b=p_max, xtol=x_tol)
    return root
