######################################################################## imports

import numpy as np
import jax
import jax.numpy as jnp
import math
import time

#################################################################### subroutines

def jacobian(F, x: jnp.array):
    '''
    Compute jacobian of F at x
    Inputs:
        F: F = (f_1, f_2, ..., f_n)
        x: x = (x_1, x_2, ..., x_n)
    Outputs:
        J: n x n Jacobian matrix of f evaluated at x
    '''
    return jax.jacfwd(F)(x)


def nDNewton(x0, F, tol, nMax):
    '''
    Run Newton's Method on a vector valued function
    Inputs:
        x0: initial guess, x0 = (x_1, ..., x_n)
        F: F = (f_1, ..., f_n)
        tol: tolerance
        nMax: max iterations
    Outputs:
        xStar: approximate root
        err: error message
        its: number of iterations run
    '''

    # run until we reach nMax or tolerance
    for its in range(nMax):
        J = jacobian(F, x0)
        Jinv = jnp.linalg.inv(J)
        
        x1 = x0 - jnp.dot(Jinv, F(x0))


# F_1(x, y) = [f_1(x, y), f_2(x, y)]
# f_1(x, y) = x^2 + y^2 - 4 = 0
# f_2(x, y) = e^x + y - 1 = 0
def F_1(x: jnp.array):
    F = []
    F.append(x[0] ** 2 + x[1] ** 2 - 4)
    F.append(jnp.exp(x[0]) + x[1] - 1)
    
    return jnp.array([F[0], F[1]])

print(jacobian(F_1, jnp.array([1., 1.])))

