from typing import Callable
import numpy as np
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import splu

def poisson_2D(mesh:np.ndarray, u_star, v_star, dt):
    """
    Solve the problem:
    
    \Delta p^n+1 =\frac{div \vec{p}^*}{\Delta t}

    whit homogenea neumman condition

    mesh: meshgrid that act as domain
    """
    
    # Mesh
    X,Y = mesh

    Nodesy, Nodesx = X.shape

    # Number of interior nodes in axis x
    Nx = Nodesx - 2
    # Number of interior nodes in axis y
    Ny = Nodesy - 2

    # Hinges
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    
    # Auxiliar Matrixes
    D=identity(Nx+2,dtype='float64',format='csc')
    D = (1/(dy**2)) * D

    M = lil_matrix((Nx+2,Nx+2), dtype='float64')
    d = (-2) * (1 / (dx**2) + 1 / (dy**2))
    sd = 1 / (dx**2)
    M.setdiag(d*np.ones(Nx+2),0)
    M.setdiag(sd*np.ones(Nx+1),1)
    M.setdiag(sd*np.ones(Nx+1),-1)
    M[0,1] = 2 * sd
    M[Nx+1,Nx] = 2 * sd

    # Change lil to csc
    M = M.tocsc()

    # System Matrix
    N = int((Nx+2)*(Ny+2))

    A = lil_matrix((N,N), dtype='float64')

    A[0:(Nx+2),0:(Nx+2)] = M
    A[0:(Nx+2),(Nx+2):2*(Nx+2)] = 2 * D
    A[(Ny+1)*(Nx+2):(Ny+2)*(Nx+2),(Ny+1)*(Nx+2):(Ny+2)*(Nx+2)] = M
    A[(Ny+1)*(Nx+2):(Ny+2)*(Nx+2),(Ny)*(Nx+2):(Ny+1)*(Nx+2)] = 2 * D

    for i in range(1,Ny+1):
        A[i*(Nx+2):(i+1)*(Nx+2),(i-1)*(Nx+2):i*(Nx+2)] = D
        A[i*(Nx+2):(i+1)*(Nx+2),i*(Nx+2):(i+1)*(Nx+2)] = M
        A[i*(Nx+2):(i+1)*(Nx+2),(i+1)*(Nx+2):(i+2)*(Nx+2)] = D

    A = A.tocsc()

    #Second Member (Divergence of Velocities)
    b = np.zeros((Ny+2, Nx+2))
    for i in range(1, Nx+2):
        for j in range(1,Ny+2): 
            b[j,i] = ((dy / 2) * (u_star[j,i+1] - u_star[j,i-1]) + (dx / 2) * (v_star[j+1,i] - v_star[j-1,i])) / dt

    b = b.reshape(N)

    # Resolution
    LU = splu(A)
    p = LU.solve(b)
    p = p.reshape((Ny+2,Nx+2))

    return p
    
