######################################################################## imports

import numpy as np
import jax
import matplotlib.pyplot as plt

#################################################################### subroutines

def equiNodes(a, b, n):
    '''
    Create n interpolation nodes, equally spaced on inteval [a, b]
    '''

    x = np.linspace(a, b, n)
    return x

def pLagrange(x, f, xInt):
    '''
    Evaluates p, the lagrange interpolating polynomial for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        xInt: vector of interpolation nodes
    Outputs:
        p: Lagrange interpolation evaluated at x
    '''

    # vector of individual Lagrange functions
    # L = [L_n1, ..., L_nk, ..., Lnn]
    L = []
    
    # build the kth Lagrange function (n + 1) functions
    for k in range(0, len(xInt)):
        # L_nk(x) = 
        # ((x - x_0) * ... * (x - x_k-1) * (x - x_k+1) * ... * (x - x_n)) /
        # ((x_k - x_0) * ... * (x_k - x_k-1) * (x_k - x_k+1) * ... * (x_k - x_n)

        # go term by term through numerator and denominator (n + 1 terms)
        L_k_numerator = 1.0
        L_k_denominator = 1.0
        for i in range(0, len(xInt)):
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

def pHermite(x, f, xInt):
    '''
    Evaluates p, the Hermite interpolating polynomial for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        xInt: vector of interpolation nodes
    Outputs:
        p: Hermite interpolation evaluated at x
    '''

    # f'(x) evaluator function
    fPrime = jax.grad(f)
    
    # Q is matrix of divided differences
    z = np.zeros(2 * len(xInt))
    Q = np.zeros([2 * len(xInt), 2 * len(xInt)])

    # initialize Q and z with inputted data
    for i in range(0, len(xInt)):
        z[2 * i] = xInt[i]
        z[2 * i + 1] = xInt[i]
        Q[2 * i][0] = f(xInt[i])
        Q[2 * i + 1][0] = f(xInt[i])
        Q[2 * i + 1][1] = fPrime(xInt[i])

        if (i != 0):
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) /\
                (z[2 * i] - z[2 * i - 1])

    # create divided difference coefficients
    for i in range(2, 2 * len(xInt)):
        for j in range(2, i): 
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])
    
    coefficientList = np.zeros(2 * len(xInt))
    for i in range(0, len(Q[0])):
        coefficientList[i] = Q[i][i]

    eval = f[z[0]]  

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
# Lagrange interpolation, n = 5
xEquiInterp5 = equiNodes(-5, 5, 5)
f1LagrangeEval5 = pLagrange(xEval, f1, xEquiInterp5)

# n = 5 plots
plt.figure("Problem 1) 5 nodes")
# plot actual function
plt.plot(xEval, f1Eval)
# plot Lagrange interpolation, n = 5
plt.plot(xEval, f1LagrangeEval5)
#plt.show()


def fTest(x):
    return 

print(pHermite(0, f1, xEquiInterp5))