######################################################################## imports

import numpy as np

#################################################################### subroutines

def modBisection(a, b, f, tol, *args):
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