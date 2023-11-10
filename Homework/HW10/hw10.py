######################################################################## imports
########################################################################

import numpy as np
import math
import jax
import scipy
import matplotlib.pyplot as plt

#################################################################### subroutines
####################################################################

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

def trapezoidal(f, a, b):
    '''
    Approximates the integral of f on [a, b] with the trapezoidal rule
    Inputs:
        f: function
        a: left endpoint
        b: right endpoint
    Return: approximation of integral
    '''

    # interval size
    h = abs(b - a)

    return (h / 2) * (f(a) + f(b))

def simpsons(f, a, b, midpoint):
    '''
    Approximates the integral of f on [a, b] with simpson's rule
    Inputs:
        f: function
        a: left endpoint
        b: right endpoint
    Return: approximation of integral
    '''

    # interval size
    h = abs((b - a)) / 2

    return (h / 3) * (f(a) + (4 * f(midpoint)) + f(b))

def newtonCotes(f, a, b, n, m):
    '''
    Approximates the integral of f on [a, b] with a composite nth order
    Newton-Cotes formula using m intervals, i.e. (m + 1) equally spaced points 
    t_0, ..., t_m, where a = t_0 and b = t_m
    
    For n = 2, this uses composite trapezoidal rule on the intervals 
    [t_0, t_1], ..., [t_i, t_(i + 1)], ..., [t_(m - 1), t_m]
    
    For n = 3, m MUST BE EVEN, this uses composite Simpson's rule on the 
    intervals [t_0, t_2], ... [t_i, t_(i + 2)], ..., [t_(m - 2), t_m]
    '''

    # interval size
    h = (b - a) / m

    # list of nodes
    xNodes = np.linspace(a, b, (m + 1)) # size m + 1, indices run from 0 to m

    integral = 0

    # using trapezoidal rule
    if (n == 2):
        for i in range(0, len(xNodes) - 1): # i in [0, m - 1], m values of i
            integral += trapezoidal(f, xNodes[i], xNodes[i + 1])
        return integral
    # using simpson's rule
    elif (n == 3):
        for i in range(0, len(xNodes) - 2, 2):
            integral += simpsons(f, xNodes[i], xNodes[i + 2], xNodes[i + 1])
        return integral
    else:
        print("Haven't implemented that value of n yet!")
        return -1
    
##################################################################### Problem 1)
#####################################################################

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

######################################################################## part a)

# P(3/3)(x1Eval)
y1P33Eval = Pnm(f1T6Coeff, 3, 3, x1Eval)

# x1Eval v P(3/3)(x1Eval)
plt.plot(x1Eval, y1P33Eval)

######################################################################## part b)

# P(2/4)(x1eval)
y1P24Eval = Pnm(f1T6Coeff, 2, 4, x1Eval)

# x1Eval v P(2/4)(x1Eval)
plt.plot(x1Eval, y1P24Eval)

######################################################################## part c)

# P(4/2)(x1Eval)
y1P42Eval = Pnm(f1T6Coeff, 4, 2, x1Eval)

# x1Eval v P(4/2)(x1Eval)
plt.plot(x1Eval, y1P42Eval)

plt.legend(["sin(x)", "T6(x)", "P(3/3)(x)", "P(2/4)(x)", "P(4/2)(x)"])

################################################### error for all approximations

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

##################################################################### Problem 3)
#####################################################################
print("Problem 3)")

f2 = lambda x: (1) / (1 + x ** 2)



print("Trapezoidal, n = 1,291: ")
print(newtonCotes(f2, -5, 5, 2, 1291))
print("\nSimpsons, n = 108: ")
print(newtonCotes(f2, -5, 5, 3, 108))
print("\nscipy.integrate.quad, tol = 1 x 10^-6: ")
dict1 = scipy.integrate.quad(f2, -5, 5, full_output = 1,\
                           epsabs = 1e-6)
print(dict1[0])
print("Iterations needed: " + str(dict1[2]['neval']))
print("\nscipy.integrate.quad, tol = 1 x 10^-4: ")
dict2 = scipy.integrate.quad(f2, -5, 5, full_output = 1,\
                           epsabs = 1e-4)
print(dict2[0])
print("Iterations needed: " + str(dict2[2]['neval']))
