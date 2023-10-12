######################################################################## imports

import jax
import jax.numpy as jnp
import time

#################################################################### subroutines

def jacobian(F, x: jnp.array):
    '''
    Compute jacobian of F at x
    Inputs:
        F: F = (f_1(x), ..., f_n(x))
        x: x = (x_1, ..., x_n)
    Outputs:
        J: n x n Jacobian matrix of f evaluated at x
    '''
    return jnp.array(jax.jacfwd(F)(x))

def gradient(F, x: jnp.array):
    '''
    Computes gradient of scalar-valued F at x
    Inputs:
        F: F = (f_1(x), ..., f_n(x))
        x: x = (x_1, ..., x_n)
    Outputs:

    '''
    return jnp.array(jax.grad(F)(x))


def nDNewton(x0: jnp.array, F, tol, nMax):
    '''
    Run Newton's Method on a vector valued function F
    Inputs:
        x0: initial guess, x0 = (x_1, ..., x_n)
        F: F = (f_1(x), ..., f_n(x))
        tol: tolerance
        nMax: max iterations
    Outputs:
        [xStar, err, its] where:
            xStar: approximate root
            err: error message
            its: number of iterations run
    '''

    # run until we reach nMax or tolerance
    for its in range(nMax):
        # compute jacobian and it's inverse
        J = jacobian(F, x0)
        Jinv = jnp.linalg.inv(J)
        
        # iterate
        x1 = x0 - jnp.dot(Jinv, F(x0))

        # terminate if within tolerance
        if (jnp.linalg.norm(x1 - x0) < tol):
            xStar = x1
            err = 0
            return [xStar, err, its]

        x0 = x1
    
    # exceeded max iterations
    xStar = x1
    err = 1
    return [xStar, err, its]

def nDLazyNewton(x0: jnp.array, F, tol, nMax):
    '''
    Run Newton's Method on a vector valued function F
    Inputs:
        x0: initial guess, x0 = (x_1, ..., x_n)
        F: F = (f_1(x), ..., f_n(x))
        tol: tolerance
        nMax: max iterations
    Outputs:
        [xStar, err, its] where:
            xStar: approximate root
            err: error message
            its: number of iterations run
    '''

    # compute Jacobian and inverse only once
    J = jacobian(F, x0)
    Jinv = jnp.linalg.inv(J)
    
    # run until we reach nMax or tolerance
    for its in range(nMax):
        # iterate
        x1 = x0 - jnp.dot(Jinv, F(x0))

        # terminate if within tolerance
        if (jnp.linalg.norm(x1 - x0) < tol):
            xStar = x1
            err = 0
            return [xStar, err, its]

        x0 = x1
    
    # exceeded max iterations
    xStar = x1
    err = 1
    return [xStar, err, its]

def nDBroyden(x0: jnp.array, F, tol, nMax):
    '''
    Run Broyden's method on a vector-valued function F
    Inputs:
        x0: initial guess, x0 = (x_1, ..., x_n)
        F: F = (f_1(x), ..., f_n(x))
        tol: tolerance
        nMax: max iterations
    Outputs:
        [xStar, err, its] where:
            xStar: approximate root
            err: error message
            its: number of iterations run
    '''

    A0 = jacobian(F, x0)
    v = F(x0)
    A = jnp.linalg.inv(A0)
    s = -1 * jnp.dot(A, v)
    xk = x0 + s

    for its in range(nMax):
        w = v
        v = F(xk)
        y = v - w
        z = -1 * jnp.dot(A, y)
        p = -1 * jnp.dot(s, z)
        u = jnp.dot(s, A)
        tmp = s + z
        tmp2 = jnp.outer(tmp, u)
        A = (A + (1. / p)) * tmp2
        s = -1 * jnp.dot(A, v)
        xk = xk + s
        if (jnp.linalg.norm(s) < tol):
            xStar = xk
            err = 0
            return [xStar, err, its]
        
    xStar = xk
    err = 1
    return [xStar, err, its]

def steepestDescent(x0: jnp.array, g, tol, nMax):
    '''
    Approximate a solution that minimizes g(x)
    Inputs:
        x0: initial guess, x0 = [x_1, ..., x_n]
        g: scalar function to minimize
        tol: tolerance
        nMax: max iterations
    Outputs:
        [x, err, its, g1] where:
            x: approximate minimum
            err: error code
            its: number of iterations run
            g1: approximated minimum value of g
    '''

    x = x0
    
    for its in range(nMax):
        g1 = g(x)
        z = gradient(g, x)
        z0 = jnp.linalg.norm(z)

        if (z0 == 0):
            err = 1
            return [x, err, its, g1]
        z = z / z0
        alpha1 = 0
        alpha3 = 1
        g3 = g(x - alpha3 * z)

        while (g3 >= g1):
            alpha3 = alpha3 / 2
            g3 = g(x - alpha3 * z)

            if (alpha3 < tol / 2):
                err = 2
                return [x, err, its, g1]
        
        alpha2 = alpha3 / 2
        g2 = g(x - alpha2 * z)

        h1 = (g2 - g1) / alpha2
        h2 = (g3 - g2) / (alpha3 - alpha2)
        h3 = (h2 - h1) / alpha3

        alpha0 = 0.5 * ((alpha2 - h1) / h3)
        g0 = g(x - alpha0 * z)

        if (g0 <= g3):
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval = g3
        
        x = x - alpha * z

        if (abs(gval - g1) < tol):
            err = 0
            return [x, err, its, g1]
        
    err = 3
    return [x, err, its, g1]


def driver():
############################################################################# 1)
    print("")
    print("Problem 1)")

    # F_1(x, y) = [f_1(x, y), f_2(x, y)]
    # f_1(x, y) = x^2 + y^2 - 4 = 0
    # f_2(x, y) = e^x + y - 1 = 0
    def F_1(x: jnp.array):
        F = []
        F.append(x[0] ** 2 + x[1] ** 2 - 4)
        F.append(jnp.exp(x[0]) + x[1] - 1)
        return jnp.array([F[0], F[1]])
    
    tolerance = 1e-10

########################################## i)
    print("")
    print("i)")

    # time how long it takes to call nDNewton
    start = time.time()
    # x0 = [1, 1]
    iNewton = nDNewton(jnp.array([1., 1.]), F_1, tolerance, 100)
    end = time.time()
    iNewtonDuration = end - start

    # time how long it takes to call nDLazyNewton
    start = time.time()
    # x0 = [1, 1]
    iLazyNewton = nDLazyNewton(jnp.array([1., 1.]), F_1, tolerance, 8)
    end = time.time()
    iLazyNewtonDuration = end - start

    # time how long it takes to call nDBroyden
    start = time.time()
    # x0 = [1, 1]
    iBroyden = nDBroyden(jnp.array([1., 1.]), F_1, tolerance, 3)
    end = time.time()
    iBroydenDuration = end - start


    # print results
    print("Iterations needed for Newton's Method: " + str(iNewton[2]) +\
        ", duration ran: " + str(iNewtonDuration) + ", approximated root: " +\
            str(iNewton[0]) + ", error code: " + str(iNewton[1]))
    print("Iterations needed for Lazy Newton's Method: " +\
          str(iLazyNewton[2]) + ", duration ran: " + str(iLazyNewtonDuration) +\
            ", approximated root: " + str(iLazyNewton[0]) + ", error code: " +\
                  str(iLazyNewton[1]))
    print("Iterations needed for Broyden's Method: " + str(iBroyden[2]) +\
        ", duration ran: " + str(iBroydenDuration) + ", approximated root: " +\
            str(iBroyden[0]) + ", error code: " + str(iBroyden[1]))


######################################### ii)
    print("")
    print("ii)")

    # time how long it takes to call nDNewton
    start = time.time()
    # x0 = [1, -1]
    iiNewton = nDNewton(jnp.array([1., -1.]), F_1, tolerance, 100)
    end = time.time()
    iiNewtonDuration = end - start

    # time how long it takes to call nDLazyNewton
    start = time.time()
    # x0 = [1, -1]
    iiLazyNewton = nDLazyNewton(jnp.array([1., -1.]), F_1, tolerance, 100)
    end = time.time()
    iiLazyNewtonDuration = end - start

    # time how long it takes to call nDBroyden
    start = time.time()
    # x0 = [1, -1]
    iiBroyden = nDBroyden(jnp.array([1., -1.]), F_1, tolerance, 30)
    end = time.time()
    iiBroydenDuration = end - start


    # print results
    print("Iterations needed for Newton's Method: " + str(iiNewton[2]) +\
        ", duration ran: " + str(iiNewtonDuration) + ", approximated root: " +\
            str(iiNewton[0]) + ", error code: " + str(iiNewton[1]))
    print("Iterations needed for Lazy Newton's Method: " +\
          str(iiLazyNewton[2]) + ", duration ran: " + \
            str(iiLazyNewtonDuration) + ", approximated root: " +\
                  str(iiLazyNewton[0]) + ", error code: " +\
                    str(iiLazyNewton[1]))
    print("Iterations needed for Broyden's Method: " + str(iiBroyden[2]) +\
        ", duration ran: " + str(iiBroydenDuration) + ", approximated root: " +\
            str(iiBroyden[0]) + ", error code: " + str(iiBroyden[1]))
    
######################################### iii)
    print("")
    print("iii)")

    # time how long it takes to call nDNewton
    start = time.time()
    # x0 = [0, 0]
    iiiNewton = nDNewton(jnp.array([0., 0.]), F_1, tolerance, 2)
    end = time.time()
    iiiNewtonDuration = end - start

    # time how long it takes to call nDLazyNewton
    start = time.time()
    # x0 = [0, 0]
    iiiLazyNewton = nDLazyNewton(jnp.array([0., 0.]), F_1, tolerance, 2)
    end = time.time()
    iiiLazyNewtonDuration = end - start

    # time how long it takes to call nDBroyden
    start = time.time()
    # x0 = [0, 0]
    iiiBroyden = nDBroyden(jnp.array([0., 0.]), F_1, tolerance, 2)
    end = time.time()
    iiiBroydenDuration = end - start


    # print results
    print("Iterations needed for Newton's Method: " + str(iiiNewton[2]) +\
        ", duration ran: " + str(iiiNewtonDuration) + ", approximated root: " +\
            str(iiiNewton[0]) + ", error code: " + str(iiiNewton[1]))
    print("Iterations needed for Lazy Newton's Method: " +\
          str(iiiLazyNewton[2]) + ", duration ran: " + \
            str(iiiLazyNewtonDuration) + ", approximated root: " +\
                  str(iiiLazyNewton[0]) + ", error code: " +\
                    str(iiiLazyNewton[1]))
    print("Iterations needed for Broyden's Method: " + str(iiiBroyden[2]) +\
        ", duration ran: " + str(iiiBroydenDuration) +\
            ", approximated root: " + str(iiiBroyden[0]) + ", error code: " +\
                str(iiiBroyden[1]))
    
############################################################################# 2)


if __name__ == "__main__":
    driver()




