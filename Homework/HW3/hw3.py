######################################################################## Imports

import numpy as np
import matplotlib.pyplot as plt

#################################################################### Subroutines

# wrote this instead of downloading the one on canvas because... idk
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
        condition = abs(np.longdouble(b - a)) / abs(a)

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
    ''' 
    Does the fixed point iteration of x_(n+1) = -sin(2x_n) + (5x_n / 4) - 3/4 \n
    Inputs: \n
        x0 - initial guess, \n
        Nmax - max number of iterations, \n
        tol - stopping tolerance \n
        realRoot - (optional)

    Outputs: \n
        (xStarVec, ier, count), where: \n
            xStarVec - vector of every approximation calculated, \n
            ier - 0 for success, 1 for error \n
            count - number of iterations run \n
    '''

    # iteration counter
    count = 0

    # vector containing all xstars calculated for each iteration
    xStarVec = np.zeros([Nmax + 1, 1])
    # initialize first value to be x0
    xStarVec[count] = x0
    
    # continue iterating until we reach max iterations
    while (count < Nmax):
       # increment counter
       count = count + 1
       
       # sequence is x_(n + 1) = f(x_n)
       x1 = -1 * np.sin(2 * x0) + ((5 * x0) / (4)) - (3 / 4)
       # store new iteration in xStarVec
       xStarVec[count] = x1
       
       # terminate if within tolerance
       if (((abs(x1 - x0)) / (abs(x0))) < tol):
          print("")
          # success!
          ier = 0
          return (xStarVec, ier, count)
       # update x0 to store x_(n + 1)
       x0 = x1

       print("Iterations ran: " + str(count) + ", Current approximation: "\
               + str(x1), end = '\r')

    # if we get here, we've maxed out our iterations without finding a fixed
    # point
    print("")
    print("ERROR: MAX ITERATIONS EXCEEDED")
    # failure :/
    ier = 1
    return (xStarVec, ier, count)

############################################################################# 1)

################################## c)

# define our function
f1 = lambda x: np.sin(x) - (2 * x) + 1
# tolerance
tolerance = np.longdouble(1 * 10 ** -8) # TODO: bisection runs forever when 
                                        # tolerance < 1 * 10 ^ -4
print("")
print("Problem 1c)")
# [a, b] = [-pi, pi], tolerance is 1 * 10^-5 (approximation distance), real 
# root is approx 0.887862
(r, error, iterations) = bisection(np.longdouble(-1 * np.pi), \
                                   np.longdouble(np.pi), f1, tolerance)

############################################################################# 2)

################################## a)

# define our function
f2 = lambda x: (x - 5) ** 9
# tolerance
tolerance = 1 * 10 ** -4
print("")
print("Problem 2a)")
# [a, b] = [4.82, 5.2], tolerance is 1 * 10^-4 (interval), real root is 5
bisection(4.82, 5.2, f2, tolerance)

################################### b)

# define our function
f2expanded = lambda x: (x ** 9) - (45 * x ** 8) + (900 * x ** 7) - \
(10500 * x ** 6) + (78750 * x ** 5) - (393750 * x ** 4) + (1312500 * x ** 3) - \
(2812500 * x ** 2) + (3515625 * x) - 1953125
print("")
print("Problem 2b)")
# [a, b] = [4.82, 5.2], tolerance is 1 * 10^-4 (interval), real root is 5
bisection(4.82, 5.2, f2expanded, tolerance)

############################################################################# 3)

################################### b)

# define our function
f3 = lambda x: (x ** 3) + x - 4
print("")
print("Problem 3b)")
# tolerance
tolerance = 1 * 10 ** -3
# [a, b] = [1, 4], tolerance is 1 * 10^-3 (approximation distance), real root 
# is 1.379
bisection(1, 4, f3, tolerance, 1.379)

############################################################################# 5)

##################################### a)

# create function
f4 = lambda x: x - (4 * np.sin(2 * x)) - 3

# x, y1 = f4(x), and y2 = 0 vectors to plot
x = np.linspace(-20, 20, 500)
y1 = f4(x)
y2 = np.zeros([len(x), 1])

# (approximate) roots of f4(x) = 0, calculated using Wolfram just to plot
x0 = np.array([-0.898357, -0.544442, 1.73207, 3.16183, 4.51779])
y0 = np.zeros([len(x0), 1])

# plot everything
plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.scatter(x0, y0)
plt.title("Problem 5a)")
plt.show()

###################################### b)
print("")
print("Problem 5b)")

# first root at x ~= -0.898357
print("Root: x ~= -0.898357")
# tolerance and max iterations
tolerance = 1 * 10 ** -10
maxIter = 1000
# guess: x0 = -0.9
(xStar, err, iterations) = fixedpt(f4, -0.9, tolerance, maxIter)
# can't find this root, approximation shoots off to negative infinity :/

# second root at x ~= -0.544442
print("\nRoot: x ~= -0.544442")
# max iterations
maxIter = 1000
# guess: x0 = -0.5
(xStar, err, iterations) = fixedpt(f4, -0.5, tolerance, maxIter)
# found this root, but can't improve precision beyond 5 decimal places

# third root at x ~= 1.73207
print("\nRoot: x ~= 1.73207")
# max iterations
maxIter = 5000
# guess: x0 = 1.8
(xStar, err, iterations) = fixedpt(f4, 1.8, tolerance, maxIter)
# can't find this root, keeps jumping to root at x ~= 3.16183

# fourth root at x ~= 3.16183
print("\nRoot at x ~= 3.16183")
# max iterations
maxIter = 1000
# guess: x0 = 3
(xStar, err, iterations) = fixedpt(f4, 3, tolerance, maxIter)
# found this root, but can't improve beyond 5(?) decimal places of precision

# fifth root at x ~= 4.51779
print("\nRoot at x~= 4.51779")
# max iterations
maxIter = 1000
# guess: x = 5
(xStar, err, iterations) = fixedpt(f4, 5, tolerance, maxIter)
# can't find this root, approximation shoots off to infinity :/




