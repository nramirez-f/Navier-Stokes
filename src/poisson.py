from typing import Callable
import numpy as np
from scipy.sparse import identity, lil_matrix
from post_process import *
from aprox_spacial import *
from scipy.sparse.linalg import bicgstab

def poisson_2D(nx, ny, LU, u_star, v_star, p, dx, dy, dt):
    """
    """
    #Second Member (Divergence of Intermediate Velocities)
    b = divergence(u_star, v_star, dx, dy, nx, ny) / dt
    
    b = b.reshape(nx * ny)

    # Resolution on Interior Nodes
    p_star = LU.solve(b)
    #p_star, flag = bicgstab(A, b, rtol=0.001, maxiter=3000)

    p_star = p_star.reshape((ny,nx))

    # Update pressure
    p[1:-1,1:-1] = p_star.copy()
    p[:,0] = p[:,1]
    p[:,nx+1] = p[:,nx]
    p[0,:] = p[1,:]
    p[ny+1,:] = p[ny,:]

    return p
    
