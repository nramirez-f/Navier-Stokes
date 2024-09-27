from typing import Callable
import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu
from post_process import *

def poisson_2D(mesh:np.ndarray, u_star, v_star, dt):
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
    D=identity(nx+2,dtype='float64',format='csc')
    D = (1/(dy**2)) * D

    M = lil_matrix((nx+2,nx+2), dtype='float64')
    d = (-2) * (1 / (dx**2) + 1 / (dy**2))
    sd = 1 / (dx**2)
    M.setdiag(d*np.ones(nx+2),0)
    M.setdiag(sd*np.ones(nx+1),1)
    M.setdiag(sd*np.ones(nx+1),-1)
    M[0,1] = 2 * sd
    M[nx+1,nx] = 2 * sd

    # Change lil to csc
    M = M.tocsc()

    # System Matrix
    N = int((nx+2)*(ny+2))

    A = lil_matrix((N,N), dtype='float64')

    A[0:(nx+2),0:(nx+2)] = M
    A[0:(nx+2),(nx+2):2*(nx+2)] = 2 * D
    A[(ny+1)*(nx+2):(ny+2)*(nx+2),(ny+1)*(nx+2):(ny+2)*(nx+2)] = M
    A[(ny+1)*(nx+2):(ny+2)*(nx+2),(ny)*(nx+2):(ny+1)*(nx+2)] = 2 * D

    for i in range(1,ny+1):
        A[i*(nx+2):(i+1)*(nx+2),(i-1)*(nx+2):i*(nx+2)] = D
        A[i*(nx+2):(i+1)*(nx+2),i*(nx+2):(i+1)*(nx+2)] = M
        A[i*(nx+2):(i+1)*(nx+2),(i+1)*(nx+2):(i+2)*(nx+2)] = D

    A = A.tocsc()

    #Second Member (Divergence of Intermediate Velocities)
    b = np.zeros((ny+2, nx+2))
    for i in range(1, nx+1):
        for j in range(1, ny+1): 
            b[j,i] = 0.5 * (dy * (u_star[j,i+1] - u_star[j,i-1]) + dx * (v_star[j+1,i] - v_star[j-1,i])) / dt

    intermediate_divergence(mesh, b)

    b = b.reshape(N)

    # Resolution
    LU = splu(A)
    p = LU.solve(b)
    p = p.reshape((ny+2,nx+2))

    return p
    
