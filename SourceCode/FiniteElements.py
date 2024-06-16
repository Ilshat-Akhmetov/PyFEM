from SourceCode.Points import Point2D, Point1D
from typing import Callable


class Lin_basis_func:
    left = {
        "func": lambda x, dx, end: (end - x) / dx,
        "func_deriv": lambda x, dx: -1 / dx,
    }
    right = {
        "func": lambda x, dx, start: (x - start) / dx,
        "func_deriv": lambda x, dx: 1 / dx,
    }


# here for calculating integrals midpoint integral rule was used


class Finite_el_1D_2point_chord:
    def __init__(self, left_node: Point1D, right_node: Point1D):
        self.dx = right_node.x - left_node.x
        self.start_coord = left_node
        self.points = [left_node, right_node]
        f_l = {
            "func_val": lambda x: Lin_basis_func.left["func"](
                x, self.dx, self.start_coord.x + self.dx
            ),
            "deriv_x": [lambda x: Lin_basis_func.left["func_deriv"](x, self.dx)],
        }
        f_r = {
            "func_val": lambda x: Lin_basis_func.right["func"](
                x, self.dx, self.start_coord.x
            ),
            "deriv_x": [lambda x: Lin_basis_func.right["func_deriv"](x, self.dx)],
        }
        self.local_funcs = [f_l, f_r]

    def __len__(self):
        return 2

    def calculate_integral(self, f: Callable):
        return f(self.start_coord.x + self.dx / 2) * self.dx
        # return self.dx / 3 * (f(self.start_coord.x)+4*f(self.start_coord.x+self.dx/2)+f(self.start_coord.x+self.dx)) #Simpson's rule


class Finite_el_2D_rectangle:
    def __init__(
        self,
        base_node: Point2D,
        right_node: Point2D,
        top_node: Point2D,
        top_right_node: Point2D,
    ):
        self.dx = right_node.x - base_node.x
        self.dy = top_node.y - base_node.y
        self.start_coord = base_node
        self.points = [base_node, right_node, top_node, top_right_node]
        f_base = {
            "func_val": lambda x, y: Lin_basis_func.left["func"](
                x, self.dx, self.start_coord.x + self.dx
            )
            * Lin_basis_func.left["func"](y, self.dy, self.start_coord.y + self.dy),
            "deriv_x": [
                lambda x, y: Lin_basis_func.left["func_deriv"](x, self.dx)
                * Lin_basis_func.left["func"](y, self.dy, self.start_coord.y + self.dy)
            ],
            "deriv_y": [
                lambda x, y: Lin_basis_func.left["func"](
                    x, self.dx, self.start_coord.x + self.dx
                )
                * Lin_basis_func.left["func_deriv"](y, self.dy)
            ],
        }

        f_right = {
            "func_val": lambda x, y: Lin_basis_func.right["func"](
                x, self.dx, self.start_coord.x
            )
            * Lin_basis_func.left["func"](y, self.dy, self.start_coord.y + self.dy),
            "deriv_x": [
                lambda x, y: Lin_basis_func.right["func_deriv"](x, self.dx)
                * Lin_basis_func.left["func"](y, self.dy, self.start_coord.y + self.dy)
            ],
            "deriv_y": [
                lambda x, y: Lin_basis_func.right["func"](
                    x, self.dx, self.start_coord.x
                )
                * Lin_basis_func.left["func_deriv"](y, self.dy)
            ],
        }

        f_top = {
            "func_val": lambda x, y: Lin_basis_func.left["func"](
                x, self.dx, self.start_coord.x + self.dx
            )
            * Lin_basis_func.right["func"](y, self.dy, self.start_coord.y),
            "deriv_x": [
                lambda x, y: Lin_basis_func.left["func_deriv"](x, self.dx)
                * Lin_basis_func.right["func"](y, self.dy, self.start_coord.y)
            ],
            "deriv_y": [
                lambda x, y: Lin_basis_func.left["func"](
                    x, self.dx, self.start_coord.x + self.dx
                )
                * Lin_basis_func.right["func_deriv"](y, self.dy)
            ],
        }

        f_top_right = {
            "func_val": lambda x, y: Lin_basis_func.right["func"](
                x, self.dx, self.start_coord.x
            )
            * Lin_basis_func.right["func"](y, self.dy, self.start_coord.y),
            "deriv_x": [
                lambda x, y: Lin_basis_func.right["func_deriv"](x, self.dx)
                * Lin_basis_func.right["func"](y, self.dy, self.start_coord.y)
            ],
            "deriv_y": [
                lambda x, y: Lin_basis_func.right["func"](
                    x, self.dx, self.start_coord.x
                )
                * Lin_basis_func.right["func_deriv"](y, self.dy)
            ],
        }

        self.local_funcs = [f_base, f_right, f_top, f_top_right]

    def __len__(self):
        return 4

    def calculate_integral(self, f: Callable):
        return (
            f(self.start_coord.x + self.dx / 2, self.start_coord.y + self.dy / 2)
            * self.dx
            * self.dy
        )
        # return f(self.start_coord.x + self.dx, self.start_coord.y + self.dy) * self.dx * self.dy
