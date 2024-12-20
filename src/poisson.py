# -*- coding: utf-8 -*-

from plot_graphs import *
from aprox_spacial import *
from scipy.sparse.linalg import bicgstab

def poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt, bicgstab_flag, bicgstab_rtol, bicgstab_atol, bicgstab_maxiter):
    """
    """
    # Second Member (Divergence of Intermediate Velocities)
    b = divergence(u_star, v_star, dx, dy, nx, ny) / dt
    
    b = b.reshape(nx * ny)

    # Resolution on Interior Nodes
    if bicgstab_flag == 1:
        p_star, info_flag = bicgstab(A, b, rtol=bicgstab_rtol, atol=bicgstab_atol, maxiter=bicgstab_maxiter)
    else:
        p_star = A.solve(b)

    p_star = p_star.reshape((ny,nx))

    # Update pressure
    p[1:-1,1:-1] = p_star.copy()
    p[:,0] = p[:,1]
    p[:,nx+1] = p[:,nx]
    p[0,:] = p[1,:]
    p[ny+1,:] = p[ny,:]

    return p
    
