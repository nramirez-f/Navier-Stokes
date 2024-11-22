import numpy as np
from scipy.sparse import identity, lil_matrix
import time
from scipy.sparse.linalg import splu

# Aproximation of Laplacian of pressure
def poisson_system(nx, ny, dx, dy):

    # Coefficients
    aX = dy / dx
    aY = dx / dy

    # Interior Matrixes
    aP = (-1) * (2 * aX + 2 * aY)

    M = lil_matrix((nx,nx), dtype='float64')
    M.setdiag(aX *np.ones(nx-1),-1)
    M.setdiag(aP * np.ones(nx),0)
    M.setdiag(aX * np.ones(nx-1),1)

    # First and Last Column
    M[0, 0] = (-1) * (aX + 2 * aY)
    M[nx-1, nx-1] = (-1) * (aX + 2 * aY)

    # Change lil to csc
    M = M.tocsc()

    Id = identity(nx,dtype='float64',format='csc')
    D = aY * Id

    # System Matrix
    A = lil_matrix((nx*ny,nx*ny), dtype='float64')

    # Interior Row Blocks
    for i in range(1, ny-1):
        A[i*nx:(i+1)*nx,(i-1)*nx:i*nx] = D
        A[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = M
        A[i*nx:(i+1)*nx,(i+1)*nx:(i+2)*nx] = D

    # First Row Block
    A[0:nx,0:nx] = M
    A[0:nx,nx:2*nx] = D

    # Last Row Block
    A[(ny-1)*nx:ny*nx,(ny-2)*nx:(ny-1)*nx] = D
    A[(ny-1)*nx:ny*nx,(ny-1)*nx:ny*nx] = M

    A = A.tocsc()

    LU = splu(A)

    return LU

# Aproximation of F(u)
def conv_diff_u(u, v, Re, nx, ny, dx, dy):

    # Output
    Fu = np.zeros((ny+2, nx+2))

    # Auxiliar matrixes
    a_w = np.zeros((ny+2,nx+1))
    a_e = np.zeros((ny+2,nx+1))
    a_n = np.zeros((ny+2,nx+1))
    a_s = np.zeros((ny+2,nx+1))
    a_p = np.zeros((ny+2,nx+1))

    for j in range(1, ny+1):
        for i in range(1, nx):

            a_w[j,i] = dy * (1 / (Re * dx) + 0.25 * (u[j,i] + u[j, i-1]))
            a_e[j,i] = dy * (1 / (Re * dx) - 0.25 * (u[j,i+1] + u[j, i]))
            a_n[j,i] = dx * (1 / (Re * dy) - 0.25 * (v[j,i+1] + v[j, i]))
            a_s[j,i] = dx * (1 / (Re * dy) + 0.25 * (v[j-1,i+1] + v[j-1, i]))

    a_p = a_w + a_e + a_n + a_s

    for j in range(1, ny+1):
        for i in range(1, nx):

            Fu[j,i] =  a_w[j,i] * u[j,i-1] + a_e[j,i] * u[j,i+1] + a_n[j,i] * u[j+1,i] + a_s[j,i] * u[j-1,i] - a_p[j,i] * u[j,i]

    return Fu

# Aproximation of F(v)
def conv_diff_v(u, v, Re, nx, ny, dx, dy):

    # Output
    Fv = np.zeros((ny+2, nx+2))

    # Auxiliar matrixes
    a_w = np.zeros((ny+1,nx+2))
    a_e = np.zeros((ny+1,nx+2))
    a_n = np.zeros((ny+1,nx+2))
    a_s = np.zeros((ny+1,nx+2))
    a_p = np.zeros((ny+1,nx+2))

    for j in range(1,ny):
        for i in range(1, nx):

            a_w[j,i]  = dy * (1 / (Re * dx) + 0.25 * (u[j+1,i-1] + u[j, i-1]))
            a_e[j,i]  = dy * (1 / (Re * dx) - 0.25 * (u[j+1,i] + u[j, i]))
            a_n[j,i]  = dx * (1 / (Re * dy) - 0.25 * (v[j+1,i] + v[j, i]))
            a_s[j,i]  = dx * (1 / (Re * dy) + 0.25 * (v[j,i] + v[j-1, i]))

    a_p = a_w + a_e + a_n + a_s

    for j in range(1,ny):
        for i in range(1, nx):

            Fv[j,i] =  a_w[j,i] * v[j,i-1] + a_e[j,i] * v[j,i+1] + a_n[j,i] * v[j+1,i] + a_s[j,i] * v[j-1,i] - a_p[j,i] * v[j,i]

    return Fv

def gradient(p, nx, ny, dx, dy):
    """
    Calculus of gradient of pressure
    """
    grad_pu = np.zeros((ny+2, nx+2))
    grad_pv = np.zeros((ny+2, nx+2))

    grad_pu[:,1:] = 0.5 * (p[:,1:] - p[:,:-1]) * dy
    grad_pv[1:,:] = 0.5 * (p[1:,:] - p[:-1,:]) * dx

    return (grad_pu, grad_pv)

def divergence(u_star, v_star, dx, dy, nx, ny):
    
    result = np.zeros((ny,nx)) 
    
    result = 0.5 * (dy * (u_star[1:ny+1,1:nx+1] - u_star[1:ny+1,0:nx]) + dx * (v_star[1:ny+1,1:nx+1] - v_star[0:ny,1:nx+1]))

    return  result

