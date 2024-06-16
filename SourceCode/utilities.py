from typing import Callable
import numpy as np


def get_func(func: Callable, deriv_var: str = "", deriv_deg: int = 0):
    if deriv_var == "":
        return func["func_val"]
    else:
        return func[deriv_var][deriv_deg - 1]


def inf_norm(appr_sol, analyt_sol):
    return np.max(np.abs(appr_sol - analyt_sol))
