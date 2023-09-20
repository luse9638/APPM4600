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
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
    
     # test f2
     x0 = 0.0
     [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
     print('the approximate fixed point is:', xstar)
     print('f2(xstar):', f2(xstar))
     print('Error message reads:', ier)



# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' 
    x0 = initial guess,
    Nmax = max number of iterations,
    tol = stopping tolerance,
    '''

    # iteration counter
    count = 0
    # continue iterating until we reach max iterations
    while (count < Nmax):
       # increment counter
       count = count + 1
       
       # sequence is x_(n + 1) = f(x_n)
       x1 = f(x0)
       # terminate if within tolerance
       if (abs(x1 - x0) < tol):
          xstar = x1
          # success!
          ier = 0
          return [xstar,ier]
       # update x0 to store x_(n + 1)
       x0 = x1

    # if we get here, we've maxed out our iterations without finding a fixed
    # point

    # return whatever our last iteration was
    xstar = x1
    # failure :/
    ier = 1
    return [xstar, ier]

driver()