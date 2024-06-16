import numpy as np
from scipy import sparse
from typing import Callable
from SourceCode.Domains import AbstractDomain


class EllipticDirichletFEM:
    def __init__(
        self,
        domain: AbstractDomain,
        left_part_eq: Callable,
        right_part_eq: Callable,
        dirichlet_init_cond_func: Callable,
        symmetric_problem: bool = False
    ):
        self.domain = domain
        assert callable(left_part_eq), "left_part_eq must be callable function"
        assert (
            "test_f" in left_part_eq.__code__.co_varnames
        ), "left_part_eq must have test_f parameter"
        assert (
            "main_f" in left_part_eq.__code__.co_varnames
        ), "left_part_eq must have main_f parameter"
        self.left_part_eq = left_part_eq
        self.right_part_eq = right_part_eq
        self.dirichlet_init_cond_func = dirichlet_init_cond_func
        self.symmetric = symmetric_problem

    def get_final_left_part(self, *, main_f: Callable, test_f: Callable) -> Callable:
        return self.left_part_eq(main_f=main_f, test_f=test_f)

    def get_final_right_part(self, basis_func: Callable) -> Callable:
        return lambda *args: self.right_part_eq(*args) * basis_func["func_val"](*args)

    def get_solution(self) -> np.array:
        solution = self.calculate_solution()
        solution = solution.reshape(self.domain.get_shape())
        return solution

    def apply_bound_conds(self, a_matr: sparse.dok_matrix, b_vec: np.array) -> tuple:
        for bound_ind in self.domain.bound_inds:
            _, non_zero_cols = a_matr[bound_ind, :].nonzero()
            for j in non_zero_cols:
                a_matr[bound_ind, j] = 0.0
            a_matr[bound_ind, bound_ind] = 1.0
            b_vec[bound_ind] = self.dirichlet_init_cond_func(
                *self.domain.points[bound_ind].get_val()
            )
        return a_matr, b_vec

    def calculate_solution(self) -> np.array:
        a_matr, b_vec = self.assemble()
        a_matr, b_vec = self.apply_bound_conds(a_matr, b_vec)
        a_matr = a_matr.tocsr()
        solution = sparse.linalg.spsolve(a_matr, b_vec)
        return solution

    def assemble(self) -> tuple:
        dtype = np.float64
        total_n_nodes = self.domain.n_points
        a_matr = sparse.dok_matrix((total_n_nodes, total_n_nodes), dtype=dtype)
        b_vec = np.zeros(total_n_nodes, dtype=dtype)
        nodes_in_fin_el = len(self.domain.finite_elms[0])
        for fin_el in self.domain.finite_elms:
            for row_i in range(nodes_in_fin_el):
                test_f = fin_el.local_funcs[row_i]
                final_right_part = self.get_final_right_part(test_f)
                int_val = fin_el.calculate_integral(final_right_part)
                b_vec[fin_el.points[row_i].ind] += int_val
                col_i_start = 0 if not self.symmetric else row_i
                for col_i in range(col_i_start, nodes_in_fin_el):
                    main_basis_f = fin_el.local_funcs[col_i]
                    final_left_part = self.get_final_left_part(
                        main_f=main_basis_f, test_f=test_f
                    )
                    int_val = fin_el.calculate_integral(final_left_part)
                    a_matr[
                        fin_el.points[row_i].ind, fin_el.points[col_i].ind
                    ] += int_val
                    if self.symmetric and col_i != row_i:
                        a_matr[
                            fin_el.points[col_i].ind, fin_el.points[row_i].ind
                        ] += int_val
        return a_matr, b_vec
