######################################################################## imports

import numpy as np
import matplotlib.pyplot as plt

#################################################################### subroutines

def monomial(x, f):
    '''
    Compute coefficients for monomial expansion function p interpolating f
    Inputs:
        x: vector of x-values that p(x) should pass through
        f: function of x to be interpolated
    Outputs:
        a: vector of coefficients of size
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
    Evaluates p, the interpolating monomial expansion for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        interpNode: function that creates interpolation nodes
    Outputs:
        p: p evaluated at x
    '''
    # get monomial coefficients
    coefficientList = monomial(interpNode(n), f)
    
    p = np.zeros((len(x)))
    for (j, coefficient) in zip(range(0, len(coefficientList)),\
                                coefficientList):
        p += coefficient * x ** j
    return p

############################################################################# 1)
print("Problem 1)")

##################################### b)
print("b)")

def f1(x):
    '''
    Evaluates f1(x) = 1 / (1 + (10x)^2)
    '''
    return 1.0 / (1 + (10 * x) ** 2)

def interpNode1(N):
    '''
    Gives interpolation nodes using formula x_i = -1 + (i - 1)h where
    h = 2 / (N - 1), i = 1, ..., N (N interpolation nodes)
    Inputs:
        N: creates N + 1 interpolation nodes
    Outputs:
        x1: vector of x values to be used as interpolation nodes
    '''

    x1 = np.zeros((N))
    h = 2.0 / (N - 1)
    
    # x1[0] = x_1, x1[1] = x_2, ..., x1[N - 1] = x_N
    for i in range(0, N):
        x1[i] = -1 + ((i + 1) - 1) * h
    return x1

# y0 = f1(x0)
x0 = np.linspace(-1, 1, 1001)
y0 = f1(x0)
plt.figure("Problem 1b): f(x), p_5(x), p_10(x), p_15(x)")
plt.plot(x0, y0)

# y1 = p_mon(x0), N = 5
N1 = 5
y1 = p_mon(x0, f1, N1, interpNode1)
plt.plot(x0, y1)

# y2 = p_mon(x0), N = 10
N2 = 10
y2 = p_mon(x0, f1, N2, interpNode1)
plt.plot(x0, y2)

# y3 = p_mon(x0), N = 15
N3 = 15
y3 = p_mon(x0, f1, N3, interpNode1)
plt.plot(x0, y3)
plt.legend(["f(x)", "p_5(x)", "p_10(x)", "p_15(x)"])

# y4 = p_mon(x0), N = 19
N3 = 19
y4 = p_mon(x0, f1, N3, interpNode1)
plt.figure("Problem 1b): f(x), p_19(x)")
plt.plot(x0, y0)
plt.plot(x0, y4)
plt.legend(["f(x)", "p_19(x)"])
plt.show()





