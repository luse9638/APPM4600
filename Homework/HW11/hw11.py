######################################################################## imports
########################################################################

import numpy as np

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

    # evaluate f at endpoints
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

    return (h / 2) * (ga + (2 * sum) + gb)



##################################################################### Problem 3)
#####################################################################

######################################################################## part a)
print("Problem 3a)")

f1 = lambda x: x ** 2
print(compositeGammaTrapezoid(6, 100, 1000))