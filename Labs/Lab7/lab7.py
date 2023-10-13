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

def dividedDifferences(x, f):
    '''
    Computes coefficients for the divided differences function p interpolating f
    Inputs:
        x: vector of x-values that p(x) should pass through
        f: function that p interpolates
    Outputs:
        a: vector of coefficients of size (n + 1)
    '''

    # matrix of dividedDifferences
    F = np.zeros((len(x), len(x)))
     
    # first column is f(x)
    for i in range(0, len(x)):
        F[i][0] = f(x[i]) 

    # build rest of matrix
    for i in range(1, len(x)):
        for j in range(1, i):
            F[i][j] = (F[i][j - 1] - F[i - 1][j - 1]) / (x[i] - x[i - j])
    
    # coefficients used are values along the diagonal
    a = []
    for i in range(0, len(x)):
        a.append(F[i][i])
    
    return a

def p_divided(x, f, n, interpNode):
    '''
    Evaluates p, the divided differences interpolating polynomial for f, at x
    Inputs:
        x: value to evaluate p at
        f: function that p interpolates
        n: number of interpolating nodes to use
        interpNode: function that creates interpolation nodes
    Outputs:
        p: p evaluated at x
    '''

    # get divided differences coefficients
    coefficientList = dividedDifferences(interpNode(n), f)

    p = coefficientList[0]
    for k in range(1, len(coefficientList)):
        pl = 1
        for l in range(0, k):
            return

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

    N = 20
    
    # actual function values of f
    y1 = f(x)
    # monomial interpolated function
    y2 = p_mon(x, f, N, interpNode1)
    y2err = np.abs(y2 - y1)
    # lagrange polynomial interpolated function
    y3 = p_lagrange(x, f, N, interpNode1)
    y3err = np.abs(y3 - y1)

    # upload plots for N = 10

    # time to plot!
    plt.figure()
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(-3, 3)
    plt.legend(["Original", "Monomial", "Lagrange"])
    plt.figure()
    plt.plot(x, y2err)
    plt.plot(x, y3err)
    plt.legend(["Monomial errror", "Lagrange error"])

########################################################################### 3.2)

    def interpNode2(N):
        '''
        Gives interpolation nodes using formula x_i = cos(((2j - 1)pi) / (2N))
        for j = 0, ..., N
        Inputs:
            N: creates N + 1 interpolation nodes
        Outputs:
            x1: vector of x values to be used as interpolation nodes
        '''

        x1 = np.zeros((N + 1))
        
        for i in range(0, N + 1):
            x1[i] = np.cos(((2 * i - 1) * np.pi) / (2 * N))
        return x1
    
    y4 = p_mon(x, f, N, interpNode2)
    y4err = np.abs(y4 - y1)
    plt.figure()
    plt.plot(x, y1)
    plt.plot(x, y4)
    plt.ylim(-3, 3)
    plt.legend(["Original", "Lagrange"])
    plt.figure()
    plt.plot(x, y4err)
    plt.legend(["Lagrange error"])
    


if (__name__ == "__main__"):
    driver()


