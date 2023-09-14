# Author: Luke Sellmayer, luse9638

# imports
import numpy as np

## Exercise 4

# 1a) f(x) = x^2(x - 1), (a, b) = (0.5, 2)
# f(a) = f(0.5) = -1/8, f(b) = f(2) = 4
# a sign change exists between the two endpoints so bisection will work

# 1b) (a, b) = (-1, 0.5)
# f(a) = f(-1) = -2, f(b) = f(0.5) = -1/8
# a sign change does not exist between the two endpoints so bisection will not
# work

# 1c) (a, b) = (-1, 2)
# f(a) = f(-1) = -2, f(b) = f(2) = 4
# a sign change exists between the two endpoints so bisection will work


# 2)
# create bisection subroutine
def bisection(a, b, f, tol):
    """
    Find root of function f on interval (a, b)
        Inputs:
            a: left endpoint
            b: right endpoint
            f: function to find roots of
            tol: desire minimum interval size containing a root
        
        Outputs:
            c: approximation of root, where f(c) ~= 0
            err: 0 if successful, 1 if failure
    """

    # if no sign changes occurs, bisection won't work
    if ((f(a) * f(b)) >= 0): 
        c = b
        err = 1
        return ("Err: " + str(err), "Root: " + str(c))

    # now that we know a sign change occurs we can proceed
    
    # assume that root is in left interval
    # set initial values for midpoint d, f(a), and f(d)
    d = 0.5 * (a + b)
    fa = f(a)
    fd = f(d)

    # continue until interval is within tolerance
    while (abs(b - a) > tol):
        # root is actually in right interval
        if ((fd * fa) > 0):
            # move left endpoint to midpoint
            a = d 
            fa = fd
        else: # we were correct, root is in left interval
            # move right endpoint to midpoint
            b = d
        
        # update midpoint
        d = 0.5 * (a + b)
        fd = f(d)

    # once loop ends, return root approximation
    c = d
    err = 0
    return ("Err: " + str(err), "Root: " + str(c))

# tolerance
epsilon = 10 ** (-5)

# 2a) f(x) = (x - 1)(x - 3)(x - 5), (a, b) = (0, 2.4), epsilon = 10^-5

# define our function
f1 = lambda x: (x - 1) * (x - 3) * (x - 5)

# returns 0.9999984741210936, which is expected since there is a root at x = 1
print(bisection(0, 2.4, f1, epsilon)) 

# 2b) f(x) = (x - 1)^2(x - 3), (a, b) = (0, 2)
# define our function
f2 = lambda x: ((x - 1) ** 2) * (x - 3)

# returns an error, which is expected since there is no sign change between a
# and b
print(bisection(0, 2, f2, epsilon))

# 2c) f(x) = sin(x), (a, b) = (0, 0.1) and (a, b) = (0.5, 3pi/4)
# define our function
f3 = lambda x: np.sin(x)

# returns an error, as no sign change occurs in (0, 0.1)
print(bisection(0, 0.1, f3, epsilon))

# returns an error, as no sign change occurs in (0.5, 3pi/4)
print(bisection(0.5, (3 * np.pi) / 4, f3, epsilon))
