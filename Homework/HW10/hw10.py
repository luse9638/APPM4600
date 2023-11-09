######################################################################## imports

import numpy as np
import math
import jax
import scipy
import matplotlib.pyplot as plt

#################################################################### subroutines
def Tn(f, n, x0, x):
    '''
    Compute the nth order Taylor polynomial of function f , centered at x0,
    evaluated at x
    Inputs:
        f: function to approximate
        x0: value to center Taylor polynomial
        x: value to evaluate Taylor polynomial at
    Outputs:
        T6x: Taylor polynomial of f evaluated at x
    '''

    # start getting derivatives of f
    fd = f
    
    # start sum at 0
    T6x = 0
    for i in range(0, n + 1): # [0, n]
        T6x += ((fd(x0)) / (math.factorial(i))) *\
            jax.numpy.power((x - x0), i)
        
        # get next derivative of f
        fd = jax.grad(fd)
        
    return T6x

def Pnm(f, coeff, n, m, x):
    '''
    Compute the (n/m) Pade approximation of f created using Taylor Series
    coefficients, evaluated at x
    Inputs:
        f: function to approximate
        coeff: array of Taylor Series coefficients
        n: order of numerator polynomial
        m: order of denominator polynomial
        x: value to evaluate approximation at
    Outputs:
        Pmnx: Pade approximation evalauted at x

    '''
    
    p, q = scipy.interpolate.pade(coeff, m, n)
    return p(x) / q(x)

##################################################################### Problem 1)
print("Problem 1)")
############################### part a)

# f1(x) = sin(x)
f1 = lambda x: jax.numpy.sin(x)

# x-values to evaluate at
x1Eval = jax.numpy.linspace(0, 5, 1000)

# f1(x)
y1Eval = f1(x1Eval)
# T6(x)
y1T6Eval = Tn(f1, 6, 0.0, x1Eval)
# P(3/3)(x)
f1T6Coeff = [0, 1, 0, (-1 / 6), 0, (1 / 120), 0]
y1P33Eval = Pnm(f1, f1T6Coeff, 3, 3, x1Eval)

plt.figure("Problem 1)")

# x1Eval v f1(x1Eval)
plt.plot(x1Eval, y1Eval)
# x1Eval v T6(x1Eval)
plt.plot(x1Eval, y1T6Eval)
# x1Eval v P(3/3)(x1Eval)
plt.plot(x1Eval, y1P33Eval)
plt.legend(["sin(x)", "T6(x)", "P(3/3)(x)"])

plt.show()

