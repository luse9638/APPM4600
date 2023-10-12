######################################################################## imports

import numpy as np
import matplotlib.pyplot as plt

#################################################################### subroutines
def monomial(x, f):
    '''
    Compute coefficients for monomial function p(x) interpolating f(x)
    Inputs:
        x: vector of x-values that p(x) should pass through
        f: function of x to be interpolated
    Outputs:
        a: vector of coefficients of size N + 1
    '''

    # y = [f(x_0), ..., f(x_n)]
    y = f(x)

    # V = [1, x_0, ..., x_0^n]
    #     [1, x_1, ..., x_1^n]
    #     [.   .    .     .  ]
    #     [1, x_n, ..., x_n^n]
    V = np.zeros((len(x), len(x)))
    for row in range(0, len(x)): 
        for col in range(0, len(x)): 
            V[row][col] = x[row] ** col

    # V^-1
    Vinv = np.linalg.inv(V)

    # a = V^-1 y
    a = np.matmul(Vinv, y)

    return a

def p_mon(x, f, n, interpNode):
    '''
    Evaluates p, the interpolating polynomial using monomials for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        interpNode: function that creates interpolation nodes
    Outputs:
        p: p evaluated at x
    '''
    coefficientList = monomial(interpNode(n), f)
    p = np.zeros((len(x)))
    for (j, coefficient) in zip(range(0, len(coefficientList)),\
                                coefficientList):
        p += coefficient * x ** j
    return p

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

def driver():
########################################################################### 3.1)
    def f(x):
        '''
        Evaluates f(x) = 1 / (1 + (10x)^2)
        '''
        return 1.0 / (1 + (10 * x) ** 2)
    
    def interpNode1(N):
        '''
        Gives interpolation nodes using formula x_i = -1 + (i - 1)h where
        h = 2 / (N - 1), j = 0, 1, ..., N
        Inputs:
            N: creates N + 1 interpolation nodes
        Outputs:
            x1: vector of x values to be used as interpolation nodes
        '''

        x1 = np.zeros((N + 1))
        h = 2.0 / (N - 1)
        
        for i in range(0, N + 1):
            x1[i] = -1 + (i - 1) * h

        return x1
    
    # 1000 x values
    x = np.linspace(-1, 1, 1000)

    N = 30
    
    # actual function values of f
    y1 = f(x)
    # monomial interpolated function
    y2 = p_mon(x, f, N, interpNode1)
    # lagrange polynomial interpolated function
    y3 = p_lagrange(x, f, N, interpNode1)

    # upload plots for N = 10

    # time to plot!
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(-3, 3)
    plt.legend(["Original", "Monomial", "Lagrange"])
    plt.show()


if (__name__ == "__main__"):
    driver()


