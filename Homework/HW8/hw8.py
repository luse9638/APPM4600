######################################################################## imports

import numpy as np
import matplotlib.pyplot as plt

#################################################################### subroutines'

def equiNodes(a, b, n):
    '''
    Create n interpolation nodes, equally spaced on inteval [a, b]
    '''

    x = np.linspace(a, b, n)
    return x

def p_lagrange(x, f, n, interpNode):
    '''
    Evaluates p, the lagrange interpolating polynomial for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        interpNode: function that creates interpolation nodes
    Outputs:
        p: p evaluated at x
    '''

    # interpolation node vectors
    xInt = interpNode(n)

    # vector of individual Lagrange functions
    # L = [L_n1, ..., L_nk, ..., Lnn]
    L = []
    
    # build the kth Lagrange function (n + 1) functions
    for k in range(0, n + 1):
        # L_nk(x) = 
        # ((x - x_0) * ... * (x - x_k-1) * (x - x_k+1) * ... * (x - x_n)) /
        # ((x_k - x_0) * ... * (x_k - x_k-1) * (x_k - x_k+1) * ... * (x_k - x_n)

        # go term by term through numerator and denominator (n + 1 terms)
        L_k_numerator = 1.0
        L_k_denominator = 1.0
        for i in range(0, n + 1):
            if (i != k):
                L_k_numerator *= (x - xInt[i])
                L_k_denominator *= (xInt[k] - xInt[i])

        L_k = L_k_numerator / L_k_denominator
        L.append(L_k)

    # p(x) = f(x_0)L_n0(x_0) + ... + f(x_n)L_nn(x_n)
    p = 0.0
    for i in range(0, len(L)):
        p += f(xInt[i]) * L[i]
    
    return p

##################################################################### Problem 1)
print("Problem 1)")

def f1(x):
    '''
    Calculate f(x) = 1 / (1 + x^2)
    '''
    return 1 / (1 + (x ** 2))

# x values to evaluate function and interpolations at
xEval = np.linspace(-5, 5, 1000)
# actual function evaluation
f1Eval = f1(xEval)

#f1LagrangeEval = p_lagrange(xEval, f1, 5, equiNodes)

# plot actual function
plt.plot(xEval, f1Eval)
plt.show()


