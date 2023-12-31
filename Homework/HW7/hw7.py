######################################################################## imports

import numpy as np
import matplotlib.pyplot as plt

#################################################################### subroutines

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
    for i in range(0, N): # [0, N - 1]
        x1[i] = -1 + ((i + 1) - 1) * h
    return x1

def interpNode2(N):
    '''
    Gives interpolation nodes using formula x_j = cos(((2j - 1)pi) / (2N))
    for j = 1, ..., N (N interpolation nodes)
    Inputs:
        N: creates N + 1 interpolation nodes
    Outputs:
        x1: vector of x values to be used as interpolation nodes
    '''

    x1 = np.zeros((N))
    i = np.arange(1, N + 1, 1)
    x1 = np.cos(((2 * i) - 1) * np.pi / (2 * N))
    
    # # x1[0] == x_1, ..., x[N - 1] == x_N
    # for i in range(0, N): # [0, N - 1], N iterations
    #     x1[i] = np.cos(((2 * i - 1) * np.pi) / (2 * N))
    return x1

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

def p_bary(x, f, n, interpNode):
    '''
    Evaluates p, the barycentric interpolating polynomial for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        interpNode: function that creates interpolation nodes
    Outputs:
        p: p evaluated at x
    '''

    # create n interpolation nodes
    xInt = interpNode(n)

    # compute phi_n
    phi_n = 1
    # x_i = xInt[i - 1]: xInt[0] = x_1, xInt[1] = x_2, ..., xInt[n - 1] = x_n
    for i in range(0, n): # [0, n - 1]
        print(i)
        phi_n *= (x - xInt[i])

    # compute main summation
    mainSum = 0
    # x_j = xInt[j - 1]: xInt[0] = x_1, xInt[1] = x_2, ..., xInt[n - 1] = x_n
    for j in range(0, n): # [0, n - 1]
        # compute w_j
        wjRecp = 1
        # x_i = xInt[i - 1]: xInt[0] = x_1, ..., xInt[n - 1] = x_n
        for i in range(0, n): # [0, n - 1]
            if (i != j):
                wjRecp *= (xInt[j] - xInt[i])
        wj = np.reciprocal(wjRecp)

        # calculate sum term if division by zero doesn't occur
        if (0 not in (x - xInt[j])):
            mainSum += (wj / (x - xInt[j])) * f(xInt[j])

    return phi_n * mainSum

############################################################################# 1)
print("Problem 1)")

##################################### b)
print("b)")

def f1(x):
    '''
    Evaluates f1(x) = 1 / (1 + (10x)^2)
    '''
    return 1.0 / (1 + (10 * x) ** 2)

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
N4 = 19
y4 = p_mon(x0, f1, N4, interpNode1)
plt.figure("Problem 1b): f(x), p_19(x)")
plt.plot(x0, y0)
plt.plot(x0, y4)
plt.legend(["f(x)", "p_19(x)"])

##################################################################### Problem 2)
print("Problem 2)")

# y0 = f1(x0)
plt.figure("Problem 2): f(x), p_19(x)")
plt.plot(x0, y0)

# y5 = p_bary(x0), N = 19
y5 = p_bary(x0, f1, N4, interpNode1)
plt.plot(x0, y5)
plt.legend(["f(x)", "p_19(x)"])

##################################################################### Problem 3)
print("Problem 3)")

# y0 = f1(x0)
plt.figure("Problem 3): f(x), p_10(x), p_30(x), p_50(x)")
plt.plot(x0, y0)

# y6 = p_bary(x0), N = 10
y6 = p_bary(x0, f1, 10, interpNode2)
plt.plot(x0, y6)

# y7 = p_bary(x0), N = 30
y7 = p_bary(x0, f1, 30, interpNode2)
plt.plot(x0, y7)

# y8 = p_bary(x0), N = 50
y8 = p_bary(x0, f1, 50, interpNode2)
plt.plot(x0, y8)
plt.legend(["f(x)", "p_10(x)", "p_30(x)", "p_50(x)"])
plt.show()













