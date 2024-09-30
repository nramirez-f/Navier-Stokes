from typing import Callable
import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu
from post_process import *
from aprox_spacial import *

def poisson_2D(mesh:np.ndarray, u_star, v_star, p, dt):
    """
    Solve the problem:
    
    \Delta p^n+1 =\frac{div \vec{p}^*}{\Delta t}

    whit homogenea neumman condition

    mesh: meshgrid that act as domain
    """
    
    # Mesh
    X,Y = mesh

    nodesy, nodesx = X.shape

    # Number of interior nodes in axis x
    nx = nodesx - 2
    # Number of interior nodes in axis y
    ny = nodesy - 2

    # Hinges
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    
    # Auxiliar Matrixes
    D=identity(nx,dtype='float64',format='csc')
    D = (1/(dy**2)) * D

    M = lil_matrix((nx,nx), dtype='float64')
    d = (-2) * (1 / (dx**2) + 1 / (dy**2))
    sd = 1 / (dx**2)
    M.setdiag(d*np.ones(nx),0)
    M.setdiag(sd*np.ones(nx-1),1)
    M.setdiag(sd*np.ones(nx-1),-1)
    M[0,1] = 2 * sd
    M[nx-1,nx-2] = 2 * sd

    # Change lil to csc
    M = M.tocsc()

    # System Matrix
    A = lil_matrix((nx*ny,nx*ny), dtype='float64')

    # Neumman Conditions (State Duplication)
    # First Row Blocks
    A[0:nx,0:nx] = M
    A[0:nx,nx:2*nx] = 2 * D
    # Last Row Blocks
    A[(ny-1)*nx:ny*nx,(ny-1)*nx:ny*nx] = M
    A[(ny-1)*nx:ny*nx,(ny-2)*nx:(ny-1)*nx] = 2 * D

    for i in range(1, ny-1):
        A[i*nx:(i+1)*nx,(i-1)*nx:i*nx] = D
        A[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = M
        A[i*nx:(i+1)*nx,(i+1)*nx:(i+2)*nx] = D

    A = A.tocsc()

    #Second Member (Divergence of Intermediate Velocities)
    b = divergence(u_star, v_star, dx, dy, nx, ny) / dt

    intermediate_divergence((X[1:ny+1,1:nx+1], Y[1:ny+1, 1:nx+1]), divergence(u_star, v_star, dx, dy, nx, ny))

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
    
