import torch as t
from bisecting import bisect
from mpmath import *  # Note this includes infinity and log

mp.dps = 1000  # TODO: Increase precision


def f(loss_vec, v_vec):
    return sum([loss * v for loss, v in zip(loss_vec, v_vec)])
    # return (loss_vec * v_vec).sum()


def phi(loss_vec, u_vec, mu):
    term1 = log(sum([-u / (mu + loss) for u, loss in zip(u_vec, loss_vec)]))
    # term1 = log(
    #     - (u_vec / (mu + loss_vec)).sum()
    # )
    term2 = sum([u * log(-(mu + loss)) for u, loss in zip(u_vec, loss_vec)])
    # term2 = (u_vec * log(-(mu + loss_vec))).sum()
    return term1 + term2


def get_interval(max_loss, phi_func):
    a = -mpf(max_loss + 1)
    # print("a:", a)
    # print("phi(a):", phi_func(a))
    while phi_func(a) > 0:
        a *= 2
        # print("a:", a)
        # print("phi(a):", phi_func(a))
    eps = mpf("0.1")
    b = -(max_loss + eps)
    # b = - mpf(str((max_loss + eps)))
    # print("b:", b)
    # print("eps:", eps)
    # print("phi(b):", phi_func(b))
    while phi_func(b) < 0:
        eps /= 2
        b = -(max_loss + eps)
        # print("b:", b)
        # print("eps:", eps)
        # print("phi(b):", phi_func(b))
    return a, b


def get_mu_star(loss_vec, u_vec, c):
    """Solves c = phi(loss_vec, u_vec, mu) for mu"""

    def phi_func(mu):
        return phi(loss_vec, u_vec, mu) - c

    max_loss = max(loss_vec)
    interval = get_interval(max_loss, phi_func)
    # print("interval:", interval)
    root = bisect(phi_func, interval)
    return root


def get_lamb_star(loss_vec, u_vec, mu_star):
    return 1 / sum([u / (mu_star + loss) for u, loss in zip(u_vec, loss_vec)])
    # return 1 / sum(u_vec / (mu_star + loss_vec))


def get_v_star(loss_vec, u_vec, mu_star, lamb_star):
    return [lamb_star * u / (mu_star + loss) for u, loss in zip(u_vec, loss_vec)]
    # return lamb_star * u_vec / (mu_star + loss_vec)


def get_u_derivs(u_vec, lamb_star, v_star):
    return [lamb_star * (1 + log(u / v)) for u, v in zip(u_vec, v_star)]
    # return lamb_star * (1 + log(u_vec / v_star))


def get_c_deriv(lamb_star):
    return -lamb_star


# TODO: Convert the incoming arguments to mpfs
def get_data(loss_vec, u_vec, c):
    loss_vec_precise = [mpf(loss) for loss in loss_vec]
    u_vec_precise = [mpf(u.item()) for u in u_vec]
    c_precise = mpf(c.item())
    mu_star = get_mu_star(loss_vec_precise, u_vec_precise, c_precise)
    lamb_star = get_lamb_star(loss_vec_precise, u_vec_precise, mu_star)
    v_star = get_v_star(loss_vec_precise, u_vec_precise, mu_star, lamb_star)
    u_derivs = get_u_derivs(u_vec_precise, lamb_star, v_star)
    c_deriv = get_c_deriv(lamb_star)
    return [float(v) for v in v_star], [float(u_d) for u_d in u_derivs], float(c_deriv)


def get_proxy(loss_vec, u_vec, c):
    v_star, u_derivs, c_deriv = get_data(loss_vec, u_vec, c)


if __name__ == "__main__":
    loss_vec = [1, 2, 3, 4, 5, 6, 7]
    u_vec = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1]
    c = 100
    v_star, u_derivs, c_deriv = get_data(loss_vec, u_vec, c)
    print("v_star:", v_star)
    print("u_derivs:", u_derivs)
    print("c_deriv:", c_deriv)
