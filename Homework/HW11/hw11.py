######################################################################## imports
########################################################################

import numpy as np
import scipy as sp

#################################################################### subroutines
####################################################################

def compositeGammaTrapezoid(xEval, b, n):
    '''
    Approximate the gamma function, evaluated at xEval, using the composite 
    trapezoidal rule
    Inputs:
        xEval: value to evaluate
        b: right endpoint to use in integral that occurs in gamma function
        definition
        n: number of intervals to use
    Outputs:
        fEval: approximation of the gamma function evaluated at xEval
    '''

    # gamma function: G(x) = Int_{0}^{Inf} (t^{x - 1} * e^{-t}) dt
    # approximation: G(x) ~= Int_{0}^{b} (g(t)) dt
    # where g(t) = t^{xEval - 1} * e^{-t}
    
    # create g(t)
    g = lambda t: (t ** (xEval - 1)) * np.exp(-1 * t)

    # formula to use
    # G(x) ~= (h / 2)(g(a) + 2 * Sum_{j = 1}^{n - 1} (g(x_{j})) + g(b))

    # left endpoint is 0
    a = 0

    # size of intervals
    h = (b - a) / n

    # create (n + 1) nodes for n intervals, including the endpoints
    tNodes = np.linspace(a, b, n + 1)

    # evaluate g at endpoints
    ga = g(a)
    gb = g(b)

    # evaluate summation
    sum = 0
    for node in tNodes:
        # don't add endpoints to sum
        if (node == a or node == b):
            #print("bruh ", node)
            continue
        sum += g(node)

    fEval = (h / 2) * (ga + (2 * sum) + gb)
    return fEval

def gaussLaguerreGamma(xEval):
    '''
    Approximate the gamma function, evaluated at xEval, using the
    Gauss-Laguerre quadrature
    Inputs:
        xEval: value to evaluate
    Outputs:
        fEval: approximation of the gamma function evaluated at xEval
    '''

    # - gamma function: G(x) = Int_{0}^{Inf} (t^{x - 1} * e^{-t}) dt
    # - approximation: G(x) ~= Int_{0}^{b} (g(t)) dt
    #   where g(t) = t^{xEval - 1} * e^{-t}

    # - choose value for n, the number of sample points / weights
    # - Gauss-Laguerre can correctly integrate polynomials of degree (2n - 1) or
    #   less
    # - polynomial being integrated is t^{xEval - 1}
    gg = lambda t: t ** (xEval - 1)
    # - => n = ceil(xEval / 2)
    n = int(np.ceil(xEval / 2))

    # calculate sample points and weights for quadrature
    samplePoints, weights = np.polynomial.laguerre.laggauss(n)

    fEval = 0
    for i in range(0, len(samplePoints)):
        fEval += weights[i] * gg(samplePoints[i])
    
    return fEval

##################################################################### Problem 3)
#####################################################################

######################################################################## part a)
print("\n", "Problem 3a)", "\n")

numIntervals = 100000
rightEndpoint = 100

print("x = 2")
cGT2 = compositeGammaTrapezoid(2, rightEndpoint, numIntervals)
sGF2 = sp.special.gamma(2)
relErr2 = abs(cGT2 - sGF2) / abs(sGF2)
print("Composite trapezoid: ", cGT2)
print("Scipy gamma function: ", sGF2)
print("Relative error: ", relErr2, "\n")

print("x = 4")
cGT4 = compositeGammaTrapezoid(4, rightEndpoint, numIntervals)
sGF4 = sp.special.gamma(4)
relErr4 = abs(cGT4 - sGF4) / abs(sGF4)
print("Composite trapezoid: ", cGT4)
print("Scipy gamma function: ", sGF4)
print("Relative error: ", relErr4, "\n")

print("x = 6")
cGT6 = compositeGammaTrapezoid(6, rightEndpoint, numIntervals)
sGF6 = sp.special.gamma(6)
relErr6 = abs(cGT6 - sGF6) / abs(sGF6)
print("Composite trapezoid: ", cGT6)
print("Scipy gamma function: ", sGF6)
print("Relative error: ", relErr6, "\n")

print("x = 8")
cGT8 = compositeGammaTrapezoid(8, rightEndpoint, numIntervals)
sGF8 = sp.special.gamma(8)
relErr8 = abs(cGT8 - sGF8) / abs(sGF8)
print("Composite trapezoid: ", cGT8)
print("Scipy gamma function: ", sGF8)
print("Relative error: ", relErr8, "\n")

print("x = 10")
cGT10 = compositeGammaTrapezoid(10, rightEndpoint, numIntervals)
sGF10 = sp.special.gamma(10)
relErr10 = abs(cGT10 - sGF10) / abs(sGF10)
print("Composite trapezoid: ", cGT10)
print("Scipy gamma function: ", sGF10)
print("Relative error: ", relErr10, "\n")

######################################################################## part b)
print("Problem 3b)", "\n")

print("x = 2")
print("Composite trapezoid: ", cGT2)
print("Number of function calls: ", numIntervals + 1, "\n")

print("x = 4")
print("Composite trapezoid: ", cGT4)
print("Number of function calls: ", numIntervals + 1, "\n")

print("x = 6")
print("Composite trapezoid: ", cGT6)
print("Number of function calls: ", numIntervals + 1, "\n")

print("x = 8")
print("Composite trapezoid: ", cGT8)
print("Number of function calls: ", numIntervals + 1, "\n")

print("x = 10")
print("Composite trapezoid: ", cGT10)
print("Number of function calls: ", numIntervals + 1, "\n")

######################################################################## part b)
print("Problem 3c)", "\n")

print("x = 2")
gLG2 = gaussLaguerreGamma(2)
print("Gauss-Laguerre: ", gLG2, "\n")

print("x = 4")
gLG4 = gaussLaguerreGamma(4)
print("Gauss-Laguerre: ", gLG4, "\n")

print("x = 6")
gLG6 = gaussLaguerreGamma(6)
print("Gauss-Laguerre: ", gLG6, "\n")

print("x = 8")
gLG8 = gaussLaguerreGamma(8)
print("Gauss-Laguerre: ", gLG8, "\n")

print("x = 10")
gLG10 = gaussLaguerreGamma(10)
print("Gauss-Laguerre: ", gLG10, "\n")