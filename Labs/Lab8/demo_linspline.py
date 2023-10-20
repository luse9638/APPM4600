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
    plt.legend(["f(x) = e^x", "linear spline"])
     
    err = abs(yeval-fex)
    plt.figure("Exercise 3.1) error for y = e^x with linear spline")
    plt.plot(xeval,err,'go-')

    g = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1
    xeval =  np.linspace(a,b,Neval)
    y2eval = eval_lin_spline(xeval,Neval,a,b,g,Nint)
    gex = np.zeros(Neval)
    for j in range(Neval):
      gex[j] = g(xeval[j]) 

    plt.figure("Exercise 3.2) f(x) = 1 / (1 + (10x)^2)")
    plt.plot(xeval,gex,'ro-')
    plt.plot(xeval,y2eval,'bs-')
    plt.legend(["f(x)", "linear spline"])
     
    err2 = abs(y2eval-gex)
    plt.figure("Exercise 3.2) error for f(x) = 1 / (1 + (10x)^2) with linear spline")
    plt.plot(xeval,err2,'go-')
    
    plt.figure("Exercise 3.4): f(x) = 1 / (1 + (10x)^2)")
    a = -1
    b = 1
    xeval =  np.linspace(a,b,Neval)
    y3eval = eval_cube_spline(xeval, Neval, a, b, g, Nint)
    plt.plot(xeval, gex)
    plt.plot(xeval, y3eval)
    plt.legend(["f(x)", "cubic spline"])
    
    err3 = abs(y3eval-gex)
    plt.figure("Exercise 3.4): error for f(x) = 1 / (1 + (10x)^2) with cubic spline")
    plt.plot(xeval,err2,'go-')
    plt.show()
    
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

def cubeCoeff(a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)
    
    h = np.zeros(Nint)
    for i in range(0, Nint): # [0, Nint - 1], Nint iterations
        h[i] = xint[i + 1] - xint[i]
    
    alpha = np.zeros(Nint)
    for i in range(1, Nint): # [1, Nint - 1], (Nint - 1) iterations
        alpha[i] = (3 / h[i]) * (yint[i + 1] - yint[i]) -\
            (3 / h[i - 1]) * (yint[i] - yint[i - 1])
    
    l = np.zeros(Nint)
    l[0] = 0
    mu = np.zeros(Nint)
    mu[0] = 0
    z = np.zeros(Nint)
    z[0] = 0
    for i in range(1, Nint): # [1, Nint - 1], Nint - 1 values
        l[i] = 2 * (xint[i + 1] - xint[i - 1]) - (h[i - 1] * mu[i - 1])
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - (h[i - 1] * z[i - 1])) / l[i]

    c = np.zeros(Nint + 1)
    c[-1] = 0
    b = np.zeros(Nint)
    d = np.zeros(Nint)
    for j in range(Nint - 1, -1, -1): # [0, Nint - 1], backwards, Nint iterations
        c[j] = z[j] - (mu[j] * c[j + 1])
        b[j] = ((yint[j + 1] - yint[j]) / h[j]) - (h[j] * (c[j + 1] + 2 * c[j]) * (1/3))
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return [yint, b, c, d]

def cubeEval(a, b, c, d, xj, x):
    return a + b * (x - xj) + c * ((x - xj) ** 2) + d * ((x - xj) ** 3)

def eval_cube_spline(xeval, Neval, a, b, f, Nint):
    coeffList = cubeCoeff(a, b, f, Nint)
    aList, bList, cList, dList = coeffList[0], coeffList[1], coeffList[2],\
    coeffList[3]

    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(Neval) 

    for jint in range(Nint):
        '''TODO fix this: find indices of xeval in interval (xint(jint),xint(jint+1))'''
        ind = [i for i in range(len(xeval)) if (xeval[i] >= xint[jint] and xeval[i] <= xint[jint + 1])]
        n = len(ind)

        for kk in range(n):
           '''TODO: use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           yeval[ind[kk]] = cubeEval(aList[jint], bList[jint], cList[jint],\
                                     dList[jint], xint[jint], xeval[ind[kk]])
           
    return yeval



#   x = np.linalg.solve(A, b)
#   print(x)
      
    


           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
