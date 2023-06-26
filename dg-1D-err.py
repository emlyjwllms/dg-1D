# 1D DG error analysis for time-dependent IVP problems
# xdot = dx/dt = f(x,t)

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

# xdot function
def f(xv):
    #xdot = np.linspace(t0,tf,porder) # xdot = f = t
    xdot = xv # xdot = f = x
    #xdot = np.sin(xv)
    return xdot


# residual vector
def resid(c,xh0):
    r = np.zeros((porder+1))
    for k in range(porder+1):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        r[k] = dphi_k.T @ np.diag(w) @ phi @ c + phi_k.T @ np.diag(0.5*dt*w) @ f(phi @ c) - phi_k_right.T @ phi_right @ c + phi_k_left.T @ np.array([xh0])

    return r

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


if __name__ == "__main__":

    porders = np.array([1,2,3])
    dts = np.array([1,0.5,0.25])

    x_exact = lambda t: np.exp(t)
    x0 = 1


    err = np.zeros((len(porders),len(dts)))

    for p in range(len(porders)):
        porder = porders[p]

        # quadrature points
        xi, w = gaussquad1d(porder+1)
        Nq = len(xi)
        
        # precompute polynomials
        phi, dphi = plegendre(xi,porder)
        phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
        phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

        for i in range(len(dts)):
            dt = dts[i]
            t = np.arange(0,5,dt)
            N = len(t)

            xh = np.zeros((1)) # elements x quad points X x
            xhq = np.zeros((1)) # all quad points
            tq = np.zeros((1)) # time points that correspond to quad points
            xh[0] = x0

            xh0 = xh[0]

            cguess = np.append(xh0,np.zeros(porder))


            # integrate across elements
            for j in range(1,N): # loop across I_j's
                t0 = t[j-1]
                tf = t[j]
                c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
                xhq = np.append(xhq,phi @ c)
                tq = np.append(tq,dt*xi/2 + (t0+tf)/2)
                xh = np.append(xh,phi_right @ c) # xi = +1
                cguess = c
                xh0 = xh[-1]

            err[p,i] = scipy.linalg.norm(abs(x_exact(t) - xh))

    m = porders+1

    plt.figure()
    #l1, = plt.loglog(dts,dts**(m[0])*np.exp(np.log(err[0,-1])-(m[0])*np.log(dts[-1])),'k--',label=r'$m = p + 1$',zorder=1)
    m1,c1 = np.polyfit(np.log(dts),np.log(err[0,:]),1)
    m2,c2 = np.polyfit(np.log(dts),np.log(err[1,:]),1)
    m3,c3 = np.polyfit(np.log(dts),np.log(err[2,:]),1)
    l1, = plt.loglog(dts,np.exp(c1)*dts**m1,'--',color='tab:blue',label=r'$m_1 = $' + str(m1))
    l2, = plt.loglog(dts,np.exp(c2)*dts**m2,'--',color='tab:orange',label=r'$m_2 = $' + str(m2))
    l3, = plt.loglog(dts,np.exp(c3)*dts**m3,'--',color='tab:green',label=r'$m_3 = $' + str(m3))
    #l2, = plt.loglog(dts,dts**(m[1])*np.exp(np.log(err[1,-1])-(m[1])*np.log(dts[-1])),'k--',zorder=1)
    #l3, = plt.loglog(dts,dts**(m[2])*np.exp(np.log(err[2,-1])-(m[2])*np.log(dts[-1])),'k--',zorder=1)

    h1 = plt.scatter(dts,err[0,:],label=r'p = '+str(porders[0]))
    h2 = plt.scatter(dts,err[1,:],label=r'p = '+str(porders[1]))
    h3 = plt.scatter(dts,err[2,:],label=r'p = '+str(porders[2]))

    plt.title(r"$\dot{x} = f = x, x(0) = 1$")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$e$")
    plt.legend()
    first_legend = plt.legend(handles=[h1,h2,h3], loc='upper left')
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[l1,l2,l3], loc='lower right')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which="both",alpha=0.25)

    plt.savefig('fx_err.png',dpi=300,format='png')

    plt.show()
    
