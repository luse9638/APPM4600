######################################################################## Imports

import numpy as np

#################################################################### Subroutines

# wrote this instead of downloading the one on canvas because...
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
        condition = abs(realRoot - d)
    else:
        condition = abs(b - a)

    # continue until interval or is within tolerance
    while (condition > tol):
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

        # recalculate condition
        if args:
            condition = abs(realRoot - d)
        else:
            condition = abs(b - a)

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


############################################################################# 1)

################################## c)

# define our function
f1 = lambda x: np.sin(x) - (2 * x) + 1
# tolerance
tolerance = 1 * 10 ** -4 # TODO: bisection runs forever when 
                         # tolerance < 1 * 10 ^ -4
print("")
print("Problem 1c)")
# [a, b] = [-pi, pi], tolerance is 1 * 10^-4, real root is 0.888
(r, error, iterations) = bisection(-1 * np.pi, np.pi, f1, tolerance, 0.888)

############################################################################# 2)

################################## a)

# define our function
f2 = lambda x: (x - 5) ** 9
# tolerance
tolerance = 1 * 10 ** -4
print("")
print("Problem 2a)")
# [a, b] = [4.82, 5.2], tolerance is 1 * 10^-4, real root is 5
bisection(4.82, 5.2, f2, tolerance)

################################### b)

# define our function
f2expanded = lambda x: (x ** 9) - (45 * x ** 8) + (900 * x ** 7) - \
(10500 * x ** 6) + (78750 * x ** 5) - (393750 * x ** 4) + (1312500 * x ** 3) - \
(2812500 * x ** 2) + (3515625 * x) - 1953125
print("")
print("Problem 2b)")
# [a, b] = [4.82, 5.2], tolerance is 1 * 10^-4, real root is 5
bisection(4.82, 5.2, f2expanded, tolerance)
