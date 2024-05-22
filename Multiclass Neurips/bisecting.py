from mpmath import *


def bisect(f, interval, x_tol=1., y_tol=1.):
    a, b = interval
    assert a < b
    # TODO: Is this redundant?
    if not f(a) < 0 < f(b):
        pass
    assert f(a) < 0 < f(b)
    x_margin = b - a  # TODO: Should these be mpfs?
    y_margin = f(b) - f(a)
    c = (a + b) / 2
    while x_margin > x_tol or y_margin > y_tol:
        # print("x_margin:", x_margin, "y_margin:", y_margin)
        c = (a + b) / 2
        if a == c or c == b:
            print("Not enough precision")
            return
        f_c = f(c)
        if f_c == 0:
            return c
        if f_c < 0:
            a = c
        else:
            b = c
        x_margin = b - a
        y_margin = f(b) - f(a)
    return c


# def bisect_x_tol_only_version(f, interval, x_tol):
# 
#     a, b = interval
# 
#     assert a < b
#     assert f(a) < 0 < f(b)
# 
#     x_margin = b - a
#     c = (a + b) / 2
# 
#     while x_margin > x_tol:
# 
#         c = (a + b) / 2
#         if a == c or c == b:
#             return c
# 
#         f_c = f(c)
#         if f_c == 0:
#             return c
#         if f_c < 0:
#             a = c
#         else:
#             b = c
# 
#         x_margin = b - a
# 
#     return c
