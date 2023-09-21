# import libraries
import numpy as np
    
def driver():

     # test functions 
     
     # correct fixed point is alpha1 = 1.4987....
     f1 = lambda x: 1 + 0.5 * np.sin(x)
     # correct fixed point is alpha2 = 3.09... 
     f2 = lambda x: 3 + 2 * np.sin(x)

     # max iterations and tolerance
     Nmax = 100
     tol = 1e-6

     # test f1
     x0 = 0.0
     [xstar, ier, iterations] = fixedpt(f1, x0, tol, Nmax)
     print('the approximate fixed point is:', xstar[iterations])
     print('f1(xstar):', f1(xstar[iterations]))
     print('Error message reads:', ier)
    
     # test f2
     x0 = 0.0
     [xstar, ier, iterations] = fixedpt(f2, x0, tol, Nmax)
     print('the approximate fixed point is:', xstar[iterations])
     print('f2(xstar):', f2(xstar[iterations]))
     print('Error message reads:', ier)



# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' 
    inputs: \n
    x0 - initial guess, \n
    Nmax - max number of iterations, \n
    tol - stopping tolerance \n
    outputs: \n
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
       x1 = f(x0)
       # store new iteration in xStarVec
       xStarVec[count] = x1
       # TESTING: print each iteration value
       ########## print(xStarVec[count])
       
       # terminate if within tolerance
       if (abs(x1 - x0) / abs(x0) < tol):
          xstar = x1
          # success!
          ier = 0
          return (xStarVec, ier, count)
       # update x0 to store x_(n + 1)
       x0 = x1

    # if we get here, we've maxed out our iterations without finding a fixed
    # point

    # approximation is whatever our last iteration was
    xstar = x1
    # failure :/
    ier = 1
    return (xStarVec, ier, count)

#driver()