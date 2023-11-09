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

def Pnm(coeff, n, m, x):
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

# f1(x) = sin(x)
f1 = lambda x: jax.numpy.sin(x)

# x-values to evaluate at
x1Eval = jax.numpy.linspace(0, 5, 1000)

# f1(x1Eval)
y1Eval = f1(x1Eval)
# T6(x1Eval)
f1T6Coeff = [0, 1, 0, (-1 / 6), 0, (1 / 120), 0]
y1T6Eval = Tn(f1, 6, 0.0, x1Eval)

plt.figure("Problem 1 Graphs")
plt.title("Sin(x) v its Taylor and some of its Pade Approximations")

# x1Eval v f1(x1Eval)
plt.plot(x1Eval, y1Eval)
# x1Eval v T6(x1Eval)
plt.plot(x1Eval, y1T6Eval)

####################################################################### part a)

# P(3/3)(x1Eval)
y1P33Eval = Pnm(f1T6Coeff, 3, 3, x1Eval)
print(y1P33Eval)

# x1Eval v P(3/3)(x1Eval)
plt.plot(x1Eval, y1P33Eval)

####################################################################### part b)

# P(2/4)(x1eval)
y1P24Eval = Pnm(f1T6Coeff, 2, 4, x1Eval)

# x1Eval v P(2/4)(x1Eval)
plt.plot(x1Eval, y1P24Eval)

####################################################################### part c)

# P(4/2)(x1Eval)
y1P42Eval = Pnm(f1T6Coeff, 4, 2, x1Eval)

# x1Eval v P(4/2)(x1Eval)
plt.plot(x1Eval, y1P42Eval)

plt.legend(["sin(x)", "T6(x)", "P(3/3)(x)", "P(2/4)(x)", "P(4/2)(x)"])

################################################## error for all approximations

plt.figure("Problem 1 Error Graphs")

# T6(x1Eval)
y1T6EvalErr = jax.numpy.absolute(y1Eval - y1T6Eval)
plt.plot(x1Eval, y1T6EvalErr)

# P(3/3)(x1Eval)
y1P33EvalErr = jax.numpy.absolute(y1Eval - y1P33Eval)
plt.plot(x1Eval, y1P33EvalErr)

# P(2/4)(x1Eval)
y1P24EvalErr = jax.numpy.absolute(y1Eval - y1P24Eval)
plt.plot(x1Eval, y1P24EvalErr)

# P(4/2)(x1Eval)
y1P42EvalErr = jax.numpy.absolute(y1Eval - y1P42Eval)
plt.plot(x1Eval, y1P42EvalErr)

plt.title("Errors")
plt.legend(["T6(x)", "P(3/3)(x)", "P(2/4)(x)", "P(4/2)(x)"])

plt.show()

