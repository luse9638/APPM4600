import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 


def driver():
    
    f = lambda x: math.exp(x)
    a = 0
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    #print(yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    plt.figure("Exercise 3.1) y = e^x")
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
     
    err = abs(yeval-fex)
    plt.figure("Exercise 3.1) error for y = e^x")
    plt.plot(xeval,err,'go-')

    g = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1
    x2eval = interpNode1(Neval - 1)
    y2eval = eval_lin_spline(x2eval,Neval,a,b,g,Nint)
    gex = np.zeros(Neval)
    for j in range(Neval):
      gex[j] = g(x2eval[j]) 

    plt.figure("Exercise 3.2) y = 1 / (1 + (10x)^2)")
    plt.plot(x2eval,gex,'ro-')
    plt.plot(x2eval,y2eval,'bs-')
     
    err2 = abs(y2eval-gex)
    plt.figure("Exercise 3.2) error for y = 1 / (1 + (10x)^2)")
    plt.plot(x2eval,err2,'go-')
    #plt.show()
    
def interpNode1(N):
        '''
        Gives interpolation nodes using formula x_i = -1 + (i - 1)h where
        h = 2 / (N - 1), j = 0, 1, ..., N
        Inputs:
            N: creates N + 1 interpolation nodes
        Outputs:
            x1: vector of x values to be used as interpolation nodes
        '''

        x1 = np.zeros((N + 1))
        h = 2.0 / (N - 1)
        
        for i in range(0, N + 1):
            x1[i] = -1 + (i - 1) * h

        return x1
      
def eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    
    for jint in range(Nint):
        '''TODO fix this: find indices of xeval in interval (xint(jint),xint(jint+1))'''
        ind = [i for i in range(len(xeval)) if (xeval[i] >= xint[jint] and xeval[i] <= xint[jint + 1])]
        n = len(ind)
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        for kk in range(n):
           '''TODO: use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           yeval[ind[kk]] = lineEval(a1, fa1, b1, fb1, xeval[ind[kk]])
      
    return yeval

def lineEval(x0, y0, x1, y1, x):
    m = (y1 - y0) / (x1 - x0)
    return m * (x - x1) + y1

def cubicEval(x0, y0, x1, y1, x):
  return True

def cubicCoefficients(n):
  coefList = []
  coefList.append(0) # M0 = 0
  A = np.zeros([n - 1, n - 1])
  for i in range(0, n - 1):
      for j in range(0, n - 1):
          if (i == j):
              A[i][j] = 1. / 3
          if (j == i + 1):
              A[i][j] = 1. / 12
          if (j == i - 1):
              A[i][j] = 1. / 12
  print(A)
          

cubicCoefficients(5)


           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
