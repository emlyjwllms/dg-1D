import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import jacobi

def plegendre(x,porder):
    
    try:
        y = np.zeros((len(x),porder+1))
        dy = np.zeros((len(x),porder+1))
        ddy = np.zeros((len(x),porder+1))
    except TypeError: # if passing in single x-point
        y = np.zeros((1,porder+1))
        dy = np.zeros((1,porder+1))
        ddy = np.zeros((1,porder+1))

    y[:,0] = 1
    dy[:,0] = 0
    ddy[:,0] = 0

    if porder >= 1:
        y[:,1] = x
        dy[:,1] = 1
        ddy[:,1] = 0
    
    for i in np.arange(1,porder):
        y[:,i+1] = ((2*i+1)*x*y[:,i]-i*y[:,i-1])/(i+1)
        dy[:,i+1] = ((2*i+1)*x*dy[:,i]+(2*i+1)*y[:,i]-i*dy[:,i-1])/(i+1)
        ddy[:,i+1] = ((2*i+1)*x*ddy[:,i]+2*(2*i+1)*dy[:,i]-i*ddy[:,i-1])/(i+1)

    # return y,dy,ddy
    return y,dy

def gaussquad1d(pgauss):

    """     
    gaussquad1d calculates the gauss integration points in 1d for [-1,1]
    [x,w]=gaussquad1d(pgauss)

      x:         coordinates of the integration points 
      w:         weights  
      pgauss:         order of the polynomila integrated exactly 
    """

    n = math.ceil((pgauss+1)/2)
    P = jacobi(n, 0, 0)
    x = np.sort(np.roots(P))

    A = np.zeros((n,n))
    for i in range(1,n+1):
        P = jacobi(i-1,0,0)
        A[i-1,:] = np.polyval(P,x)

    r = np.zeros((n,), dtype=float)
    r[0] = 2.0
    w = np.linalg.solve(A,r)

    # map from [-1,1] to [0,1]
    #x = (x + 1.0)/2.0
    #w = w/2.0

    return x, w

if __name__ == "__main__":

    porder = 1
    print('porder: ' + str(porder))

    # quadrature points
    xi, w = gaussquad1d(porder+1)
    Nq = len(xi)
    print("Number of quadrature points: ", str(len(xi)))
    print(str(xi))
    print("Number of quadrature weights: ", str(len(w)))
    print(str(w))
    w = np.array([w])

    phi, dphi = plegendre(xi,porder)

    print(str(phi))

    print(str(dphi))


    phi_right, dphi_right = plegendre(1,porder)

    print(str(phi_right))