import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from scipy.integrate import quad

def eval_legendre(n, xVal):
    
    xEval = np.zeros([n + 1])
    xEval[0] = 1
    xEval[1] = xVal

    for i in range(2, len(xEval)):
        xEval[i] = (1 / (n + 1)) * ((2 * n + 1) * xVal * xEval[i - 1] - n * xEval[i - 2])

    return xEval

def driver():
    # function you want to approximate
    f = lambda x: math.exp(x)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 2
    # Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)
    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
        # reate vector with exact values'''
        fex = np.zeros(N+1)
    for kk in range(N+1):
            fex[kk] = f(xeval[kk])
    plt.figure()
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion')
    plt.legend()
    plt.show()
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()
    2
def eval_legendre_expansion(f,a,b,w,n,x):
# This subroutine evaluates the Legendre expansion
# Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab
#p = ...
# initialize the sum to 0
    pval = 0.0
    for j in range(0,n+1):
    # make a function handle for evaluating phi_j(x)
    #phi_j = lambda x: ...
    # make a function handle for evaluating phi_j^2(x)*w(x)
    #phi_j_sq = lambda x: ...
    # use the quad function from scipy to evaluate normalizations
    #norm_fac,err = ...
    # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
    #func_j = lambda x: ...
    # use the quad function from scipy to evaluate coeffs
    #aj,err = ...
    # accumulate into pval
    #pval = pval+aj*p[j]
    return 0

if __name__ == "__main__":
    # run the drivers only if this is called from the command line
    driver()

