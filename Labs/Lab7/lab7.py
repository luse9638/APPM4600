######################################################################## imports

import numpy as np
import matplotlib.pyplot as plt

########################################################################### 3.1)

def f(x):
    return 1.0 / (1 + (10 * x) ** 2)

def intNode1(N):
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

def monomial(x, f):
    '''
    Compute coefficients for Nth order monomial function p(x) interpolating f(x)
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
    for i in range(0, len(x)): # rows
        for j in range(0, len(x)): # cols
            V[i][j] = x[i] ** j

    # V^-1
    Vinv = np.linalg.inv(V)

    # a = V^-1 y
    a = np.matmul(Vinv, y)

    return a


def driver():
    x = np.linspace(-1, 1, 1000)
    
    # actual function values of f
    y1 = f(x)
    
    # store interpolation function values here
    y2 = np.zeros((1000))
    # find coefficients for interpolation function
    coefficientList = monomial(intNode1(20), f)
    
    for i in range(0, 1000):
        for (j, coefficient) in zip(range(0, len(coefficientList)), coefficientList):
            y2[i] += coefficient * x[i] ** j



    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()


if (__name__ == "__main__"):
    driver()


