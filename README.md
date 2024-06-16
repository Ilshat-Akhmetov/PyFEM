# PyFEM
Python implementation of the finite elements method based on 
Galerkin method (not Ritz!) for solving 
ordinary and partial differential equations. Currently, it 
supports solving arbitrary 1d and 2d elliptic equations 
on a simple rectangular domain. As finite elements,  
it also uses rectangular elements in 2d case and linear 
elements in 1d.

## References

More information about FEM you may find here

https://en.wikipedia.org/wiki/Finite_element_method#:~:text=The%20finite%20element%20method%20(FEM,mass%20transport%2C%20and%20electromagnetic%20potential.

https://encyclopediaofmath.org/wiki/Galerkin_method 

https://en.wikipedia.org/wiki/Galerkin_method.

https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4screen.pdf: introduction to numerical methods for solving variational problems

https://julianroth.org/documentation/fem/: nice and easy to understand python implementation of FEM



## How to use?

In **Presentations.ipynb** you may find various examples. 
For example, you want to solve the following equation:

$$ u_{xx}+u_{yy}=-1 $$

$$ u(-1,y)=u(1,y)=u(x,-1)=u(x,1)=0 $$


According to classic Galerkin FEM algorithm, first you need to multiply 
left and right parts of the equation by a test function $v$ 
so that

$ \int\int (u_{xx}+u_{yy})v dx dy=-\int\int v dx dy $

Test function should be chosen so that it should equals 
to zero at the domain boundaries.
Then you have to conduct integration by parts https://en.wikipedia.org/wiki/Integration_by_parts 
and thereby get the equation to solve

$$ \int\int u_xv_x+u_yv_y dx dy=\int\int v dx dy $$

$$ u(-1,y)=u(1,y)=u(x,-1)=u(x,1)=0 $$

To solve this equation with this program, you need to specify 
left part of the equation after it is integrated by parts. 
The right part should be specified as it is. You also should 
define domain and a function which satisfies boundary conditions. 

Here is an example of how to solve the aforementioned equation

```Python
from SourceCode import *
from math import pi
import numpy as np
from typing import Callable

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
```

The Left part must be defined as function necessary with main_f and test_f input arguments. 
Expression for left_part obtained after integrating scalar product of the initial 
left_part with a test function by parts.
It must return a callable function. The right part should be defined as it is, 
program will multiply it by a test function on its own. If you get the left part as a 
symmetric bilinear functional after the integration by parts, 
you may also specify argument symmetric_problem=True in EllipticDirichletFEM __init__. 
It will significantly speed up calculations, reducing computation time of the global 
stiffness matrix twice.

Optionally, you may also compare your obtained solution with an exact one. 







