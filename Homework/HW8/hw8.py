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
        for j in range(2, i + 1): 
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])
    
    coefficientList = np.zeros(2 * len(xInt))
    for i in range(0, len(Q[0])):
        coefficientList[i] = Q[i][i]

    eval = f(z[0])
    for k in range(1, len(coefficientList)):
        term = coefficientList[k]
        for l in range(0, k):
            index = int(np.floor(l / 2))
            term *= (x - xInt[index])
        eval += term

    return eval
       
def natCubeSplineCoeff(a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)
    
    h = np.zeros(Nint)
    for i in range(0, Nint): # [0, Nint - 1], Nint iterations
        h[i] = xint[i + 1] - xint[i]
    
    alpha = np.zeros(Nint)
    for i in range(1, Nint): # [1, Nint - 1], (Nint - 1) iterations
        alpha[i] = (3 / h[i]) * (yint[i + 1] - yint[i]) -\
            (3 / h[i - 1]) * (yint[i] - yint[i - 1])
    
    l = np.zeros(Nint)
    l[0] = 0
    mu = np.zeros(Nint)
    mu[0] = 0
    z = np.zeros(Nint)
    z[0] = 0
    for i in range(1, Nint): # [1, Nint - 1], Nint - 1 values
        l[i] = 2 * (xint[i + 1] - xint[i - 1]) - (h[i - 1] * mu[i - 1])
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - (h[i - 1] * z[i - 1])) / l[i]

    c = np.zeros(Nint + 1)
    c[-1] = 0
    b = np.zeros(Nint)
    d = np.zeros(Nint)
    for j in range(Nint - 1, -1, -1): # [0, Nint - 1], backwards, Nint iterations
        c[j] = z[j] - (mu[j] * c[j + 1])
        b[j] = ((yint[j + 1] - yint[j]) / h[j]) - (h[j] * (c[j + 1] + 2 * c[j]) * (1/3))
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return [yint, b, c, d]

def clampCubeSplineCoeff(a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)
    fPrime = jax.grad(f)
    FPO = fPrime(xint[0])
    FPN = fPrime(xint[-1])
    
    h = np.zeros(Nint)
    for i in range(0, Nint): # [0, Nint - 1], Nint iterations
        h[i] = xint[i + 1] - xint[i]
    
    alpha = np.zeros(Nint)
    alpha[0] = ((3 * (yint[1] - yint[0])) / h[0]) - (3 * FPO)
    alpha[-1] = (3 * FPN) - ((3 * (yint[-1]) - yint[-2])) / (h[-2])
    for i in range(1, Nint): # [1, Nint - 1], (Nint - 1) iterations
        alpha[i] = (3 / h[i]) * (yint[i + 1] - yint[i]) -\
            (3 / h[i - 1]) * (yint[i] - yint[i - 1])
        
    l = np.zeros(Nint)
    l[0] = 2 * h[0]
    mu = np.zeros(Nint)
    mu[0] = 1 / 2.0
    z = np.zeros(Nint)
    z[0] = alpha[0] / l[0]
    for i in range(1, Nint): # [1, Nint - 1], Nint - 1 values
        l[i] = 2 * (xint[i + 1] - xint[i - 1]) - (h[i - 1] * mu[i - 1])
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - (h[i - 1] * z[i - 1])) / l[i]

    c = np.zeros(Nint + 1)
    b = np.zeros(Nint)
    d = np.zeros(Nint)
    l[-1] = h[-2] * (2 - mu[-2])
    z[-1] = (alpha[-1] * (h[-2] * z[-2])) / l[-1]
    c[-1] = z[-1]
    for j in range(Nint - 1, -1, -1): # [0, Nint - 1], backwards, Nint iterations
        c[j] = z[j] - (mu[j] * c[j + 1])
        b[j] = ((yint[j + 1] - yint[j]) / h[j]) - (h[j] * (c[j + 1] + 2 * c[j]) * (1/3))
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return [yint, b, c, d]

def cubeEval(a, b, c, d, xj, x):
    return a + b * (x - xj) + c * ((x - xj) ** 2) + d * ((x - xj) ** 3)

def evalCubeSpline(xeval, a, b, f, Nint, opt):
    if (opt == "natural"):
        coeffList = natCubeSplineCoeff(a, b, f, Nint)
    elif (opt == "clamped"):
        coeffList = clampCubeSplineCoeff(a, b, f, Nint)
    else:
        print("Error in determining whether to use natural or clamped spline")
        return None
    aList, bList, cList, dList = coeffList[0], coeffList[1], coeffList[2],\
    coeffList[3]

    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(len(xeval)) 

    for jint in range(Nint):
        '''TODO fix this: find indices of xeval in interval (xint(jint),xint(jint+1))'''
        ind = [i for i in range(len(xeval)) if (xeval[i] >= xint[jint] and xeval[i] <= xint[jint + 1])]
        n = len(ind)

        for kk in range(n):
           '''TODO: use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           yeval[ind[kk]] = cubeEval(aList[jint], bList[jint], cList[jint],\
                                     dList[jint], xint[jint], xeval[ind[kk]])
           
    return yeval

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

# 5 equispace nodes
xEquiInterp5 = equiNodes(-5, 5, 5)
# 10 equispace nodes
xEquiInterp10 = equiNodes(-5, 5, 10)
# 15 equispace nodes
xEquiInterp15 = equiNodes(-5, 5, 15)
# 20 equispace nodes
xEquiInterp20 = equiNodes(-5, 5, 20)

# Lagrange interpolation, n = 5
f1LagrangeEval5 = pLagrange(xEval, f1, xEquiInterp5)
# Lagrange interpolation, n = 10
f1LagrangeEval10 = pLagrange(xEval, f1, xEquiInterp10)
# Lagrange interpolation, n = 15
f1LagrangeEval15 = pLagrange(xEval, f1, xEquiInterp15)
# Lagrange interpolation, n = 10
f1LagrangeEval20 = pLagrange(xEval, f1, xEquiInterp20)

# Hermite interpolation, n = 5
f1HermiteEval5 = pHermite(xEval, f1, xEquiInterp5)
# Hermite interpolation, n = 10
f1HermiteEval10 = pHermite(xEval, f1, xEquiInterp10)
# Hermite interpolation, n = 15
f1HermiteEval15 = pHermite(xEval, f1, xEquiInterp15)
# Hermite interpolation, n = 20
f1HermiteEval20 = pHermite(xEval, f1, xEquiInterp20)

# Natural cubic spline, n = 5
f1NatCubeSplineEval5 = evalCubeSpline(xEval, -5, 5, f1, 4, "natural")
# Natural cubic spline, n = 10
f1NatCubeSplineEval10 = evalCubeSpline(xEval, -5, 5, f1, 9, "natural")
# Natural cubic spline, n = 15
f1NatCubeSplineEval15 = evalCubeSpline(xEval, -5, 5, f1, 14, "natural")
# Natural cubic spline, n = 20
f1NatCubeSplineEval20 = evalCubeSpline(xEval, -5, 5, f1, 19, "natural")

# n = 5 plots
plt.figure("Problem 1) 5 nodes")
# plot actual function
plt.plot(xEval, f1Eval)
# plot Lagrange interpolation
plt.plot(xEval, f1LagrangeEval5)
# plot Hermite interpolation
plt.plot(xEval, f1HermiteEval5)
# plot natural cubic spline
plt.plot(xEval, f1NatCubeSplineEval5)
plt.legend(["Original", "Lagrange", "Hermite", "Natural Cubic"])

# n = 10 plots
plt.figure("Problem 1) 10 nodes")
# plot actual function
plt.plot(xEval, f1Eval)
# plot Lagrange interpolation
plt.plot(xEval, f1LagrangeEval10)
# plot Hermite interpolation
plt.plot(xEval, f1HermiteEval10)
# plot natural cubic spline
plt.plot(xEval, f1NatCubeSplineEval10)
plt.legend(["Original", "Lagrange", "Hermite", "Natural Cubic"])

# n = 15 plots
plt.figure("Problem 1) 15 nodes")
# plot actual function
plt.plot(xEval, f1Eval)
# plot Lagrange interpolation
plt.plot(xEval, f1LagrangeEval15)
# plot Hermite interpolation
plt.plot(xEval, f1HermiteEval15)
# plot natural cubic spline
plt.plot(xEval, f1NatCubeSplineEval15)
plt.legend(["Original", "Lagrange", "Hermite", "Natural Cubic"])

# n = 20 plots
plt.figure("Problem 1) 20 nodes")
# plot actual function
plt.plot(xEval, f1Eval)
# plot Lagrange interpolation
plt.plot(xEval, f1LagrangeEval20)
# plot Hermite interpolation
plt.plot(xEval, f1HermiteEval20)
# plot natural cubic spline
plt.plot(xEval, f1NatCubeSplineEval20)
plt.legend(["Original", "Lagrange", "Hermite", "Natural Cubic"])

plt.show()


def fTest(x):
    return 
