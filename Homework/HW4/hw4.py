######################################################################## imports

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from newton_example import newton, modNewton

#################################################################### subroutines

def bisection(a, b, f, tol, *args):
    """
    Find root of function f on interval (a, b) \n
        Inputs: \n
            a: left endpoint \n
            b: right endpoint \n
            f: function to find roots of \n
            tol: desire minimum interval size containing a root \n
            realRoot: (optional) will instead run until distance between
            approximated and real root is less than tol \n  
        Outputs: \n
            (c, err, count), where: \n
                c: approximation of root, where f(c) ~= 0 \n
                err: 0 if successful, 1 if no sign change \n
                count: number of iterations run \n
    """

    # keep track of total iterations
    count = 0
    
    # if no sign changes occurs, bisection won't work
    if ((f(a) * f(b)) >= 0): 
        c = b
        err = 1
        return (c, err, count)

    # now that we know a sign change occurs we can proceed
    
    # assume that root is in left interval
    # set initial values for midpoint d, f(a), and f(d)
    d = 0.5 * (a + b)
    fa = f(a)
    fd = f(d)

    # if actual root passed to function, switch to making sure distance between
    # approximation and real root is within tolerance rather than having the
    # interval be within tolerance
    if args:
        realRoot = args[0]
        condition = abs(np.longdouble(realRoot - d))
    else:
        condition = abs(np.longdouble(b - a))

    # continue until interval or is within tolerance
    while (condition > tol):
        # root is actually in right interval
        if ((fd * fa) > 0):
            # move left endpoint to midpoint
            a = np.longdouble(d) 
            fa = np.longdouble(fd)
        else: # we were correct, root is in left interval
            # move right endpoint to midpoint
            b = np.longdouble(d)
        
        # update midpoint
        d = np.longdouble(0.5 * (a + b))
        fd = np.longdouble(f(d))

        # recalculate condition
        if args:
            condition = abs(np.longdouble(realRoot - d))
        else:
            condition = abs(np.longdouble(b - a))

        # update iteration counter
        count += 1

        # track our progress
        print("Iterations ran: " + str(count) + ", Current approximation: "\
               + str(d), end = '\r')

    # once loop ends, return root approximation
    print("")
    c = d
    err = 0
    return (c, err, count)

def fixedpt(f, x0, tol, Nmax):
    """
    Run a fixed point iteration f on (a, b) \n
        Inputs: \n
            f: fixed point iteration \n
            x0: initial guess \n
            Nmax: max iterations to run \n
            tol: stopping tolerance \n 
        Outputs: \n
            (xstar, ier) where: \n
                xstar: approximation of fixed point \n
                ier: 0 if successful, 1 if max iterations reached \n
    """

    count = 0
    while (count < Nmax):
       count = count + 1
       x1 = f(x0)
       if (abs(x1 - x0) < tol):
          print("Iterations ran: " + str(count) + ", Current approximation: "\
               + str(x1), end = '\r')
          print("")
          xstar = x1
          ier = 0
          return (xstar, ier)
       x0 = x1
       print("Iterations ran: " + str(count) + ", Current approximation: "\
               + str(x0), end = '\r')

    print("")
    xstar = x1
    ier = 1
    return (xstar, ier)

def secant(p0, p1, f, tol, Nmax):
    """
    Find root of function f using the secand method \n
        Inputs: \n
            p0, p1: initial guess \n
            f: function \n
            tol: tolerance \n
            Nmax: max number of iterations \n
        Outputs: \n
            (pStar, err, it) where:
                pStar: approximate root \n
                err: 0 if success, 1 if failure \n
                it: number of iterations
    """

    # store approximations here
    pStar = np.zeros([Nmax])
    # lucky guesses !
    if (f(p0) == 0): 
        pStar[0] = p0
        err = 0
        return (pStar, err)
    if (f(p1) == 0):
        pStar[0] = p1
        err = 0
        return (pStar, err)
    
    # time to keep iterating
    for it in range(1, Nmax):
        # no horizontal tangents allowed
        if (abs(f(p1) - f(p0)) == 0): 
            err = 1
            pStar[it] = p1
            return (pStar, err)
        # iterate
        p2 = p0 - ((f(p1) * (p1 - p0)) / (f(p1) - f(p0)))
        # check if we should terminate
        if (abs(f(p2) - f(p1)) < tol):
            pStar[it] = p2
            err = 0
            return (pStar, err)
        # reset values
        p0 = p1
        p1 = p2

    # if we get here, exceeded max iterations
    pStar[it] = p2
    err = 1
    return (pStar, err, it)

############################################################################# 1)

################################### a)

# constants
alpha = 0.138 * 10 ** -6
t_f = 5184000
T_i = 20
T_s = -15

# f(x) = erf(x / 2sqrt(alpha * t_f)) + (T_s) / (T_i - T_s)
f1 = lambda x: sp.special.erf((x) / (2 * np.sqrt(alpha * t_f))) +\
((T_s) / (T_i - T_s))
# f'(x)
df1dx = lambda x: ((1) / (np.sqrt(np.pi * alpha * t_f))) *\
    np.exp((-1 * x**2) / (4 * alpha * t_f))

# 500 points of (x, f(x))
x = np.linspace(0, 1, 500)
y = f1(x)

# y0 = 0
y0 = np.zeros([len(x)])

# plot!
plt.plot(x, y)
plt.plot(x, y0)
plt.title("Problem 1a)")
plt.show()

################################### b)
print("")
print("Problem 1b)")

# tolerance
tolerance = 10 ** -13

# run bisection on interval [0, 1]
(root, error, iterations) = bisection(0, 1, f1, tolerance)
print("Error code: " + str(error))

################################### c)
print("")
print("Problem 1c)")

# Newton's method iteration
n1 = lambda x: x - (f1(x) / df1dx(x))

# run Newton's method with x0 = 0.01
(p1, pstar, info, it) = newton(f1, df1dx, 0.01, tolerance, 1000)
print("Approximate depth with x0 = 0.01: " + str(p1[it]))
print("Error code: " + str(info))

############################################################################# 4)

# define function and derivatives
f2 = lambda x: np.power(np.exp(x) - (3 * np.power(x, 2)), 3)
df2dx = lambda x: 3 * np.power((np.exp(x) - 3 * np.power(x, 2)), 2) *\
    (np.exp(x) - 6 * x)
d2f2dx2 = lambda x: 3 * (np.exp(x) - 3 * np.power(x, 2)) *\
    (90 * np.power(x, 2) - 3 * np.exp(x) * (np.power(x, 2) + 8 * x + 2) + \
                        3 * np.exp(2 * x))

tolerance2 = 10 ** -16

################################### ii)
print("")
print("Problem 4ii)")

# modified function for newton's method
mu = lambda x: f2(x) / df2dx(x)
dmudx = lambda x: 1 - ((f2(x) * d2f2dx2(x)) / (np.power(df2dx(x), 2)))

# run the modified function on newton's method
(p2, pstar, info, it) = newton(mu, dmudx, 4, tolerance2, 7)
for (iter, val) in zip(range(it), p2):
    print("Iteration: " + str(iter) + ", Approximation: " + str(val))

################################## iii)
print("")
print("Problem 4iii)")

# now run the function on modified newton's method
(p3, pstar, info, it) = modNewton(f2, df2dx, 4, 3, tolerance, 6)
for (iter, val) in zip(range(len(p3)), p3):
    print("Iteration: " + str(iter) + ", Approximation: " + str(val))

############################################################################# 5)

# define our function
f3 = lambda x: np.power(x, 6) - x - 1
df3dx = lambda x: 6 * np.power(x, 5) - 1

#################################### a)
print("")
print("Problem 5a)")

# first run Newton's on the function
(pNewt, pstar, info, it1) = newton(f2, df2dx, 4, tolerance, 6)

# then run secant on the function
(pSec, err, it2) = secant(2, 1, f3, tolerance, 20)

for i in range(max(it1, it2)):
    print("Iteration: " + str(i) + ", Newton: " + str(pNewt[i]) + \
          ", Secant: " + str(pSec[i]))





