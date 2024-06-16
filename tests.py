from SourceCode import *
from math import pi, e
import numpy as np
from typing import Callable


def test1():
    def analyt_func(x, y):
        n: int = 100
        total_s: float = 0
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                total_s += (
                    (-1) ** ((i + j) // 2 - 1)
                    / (i * j * (i * i + j * j))
                    * np.cos(i * pi / 2 * x)
                    * np.cos(j * pi / 2 * y)
                )
        total_s = total_s * (8 / (pi * pi)) ** 2
        return total_s

    def left_part(main_f: Callable, test_f: Callable) -> Callable:
        return lambda x, y: (
            get_func(main_f, "deriv_x", 1)(x, y) * get_func(test_f, "deriv_x", 1)(x, y)
            + get_func(main_f, "deriv_y", 1)(x, y)
            * get_func(test_f, "deriv_y", 1)(x, y)
        )

    right_part = lambda x, y: 1
    dirichle_cond = lambda x, y: 0
    xl = -1
    xr = 1
    yl = -1
    yr = 1
    n_points = 10
    domain = Domain2DRectangle(n_points, n_points, xl, xr, yl, yr)
    fem_obj = EllipticDirichletFEM(domain, left_part, right_part, dirichle_cond,
                                   symmetric_problem=True)
    appr_sol = fem_obj.get_solution()
    x, y = domain.get_domain()
    exact_sol = analyt_func(x, y)
    error = inf_norm(appr_sol, exact_sol)
    print(error)


#
def test2():
    def left_part(main_f: Callable, test_f: Callable) -> Callable:
        return lambda x: (
            get_func(main_f, "deriv_x", 1)(x) * get_func(test_f, "deriv_x", 1)(x)
            + get_func(main_f)(x) * get_func(test_f)(x)
        )

    right_part = lambda x: 0
    dirichle_cond = lambda x: x
    xl = 0
    xr = 1
    n_points = 100
    domain = Domain1D(n_points, xl, xr)
    fem_obj = EllipticDirichletFEM(domain, left_part, right_part, dirichle_cond, symmetric_problem=True)
    appr_sol = fem_obj.get_solution()
    analyt_func = lambda x: e / (e * e - 1) * np.exp(x) + e / (1 - e * e) * np.exp(-x)
    x = domain.get_domain()
    exact_sol = analyt_func(x)
    error = inf_norm(appr_sol, exact_sol)
    print(error)