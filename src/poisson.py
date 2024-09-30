from typing import Callable
import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu
from post_process import *
from aprox_spacial import *

def poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt):
    """
    """
    #Second Member (Divergence of Intermediate Velocities)
    b = divergence(u_star, v_star, dx, dy, nx, ny) / dt

    #intermediate_divergence((X[1:ny+1,1:nx+1], Y[1:ny+1, 1:nx+1]), divergence(u_star, v_star, dx, dy, nx, ny))

    b = b.reshape(nx*ny)

    # Resolution on Interior Nodes
    LU = splu(A)
    p_star = LU.solve(b)
    p_star = p_star.reshape((ny,nx))

    # Update pressure
    p[1:ny+1, 1:nx+1] = p_star.copy()
    p[0,:] = p[1,:]
    p[ny+1,:] = p[ny,:]
    p[:,0] = p[:,1]
    p[:,nx+1] = p[:,nx]

    return p
    
