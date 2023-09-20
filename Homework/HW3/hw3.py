#################################################################### Subroutines

# create bisection subroutine
# wrote this instead of downloading the one on canvas because...
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


############################################################################# 1)
################################## c)