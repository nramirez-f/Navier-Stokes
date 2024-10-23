import numpy as np
from scipy.sparse import identity, lil_matrix

# Aproximation of Laplacian of pressure
def poisson_system(nx, ny, dx, dy):
    
    # Auxiliar Matrixes
    D=identity(nx,dtype='float64',format='csc')
    D = (dx / dy) * D

    M = lil_matrix((nx,nx), dtype='float64')
    d = (-2) * ((dy / dx) + (dx / dy))
    sd = dy / dx
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
    Ms = lil_matrix((nx,nx), dtype='float64')
    Ms.setdiag(d*np.ones(nx),0)
    Ms.setdiag(sd*np.ones(nx-1),1)
    A[0:nx,0:nx] = Ms
    A[0:nx,nx:2*nx] = 2 * D

    # Last Row Blocks
    Mn = lil_matrix((nx,nx), dtype='float64')
    Mn.setdiag(d*np.ones(nx),0)
    Mn.setdiag(sd*np.ones(nx-1),-1)
    A[(ny-1)*nx:ny*nx,(ny-1)*nx:ny*nx] = Mn
    A[(ny-1)*nx:ny*nx,(ny-2)*nx:(ny-1)*nx] = 2 * D

    for i in range(1, ny-1):
        A[i*nx:(i+1)*nx,(i-1)*nx:i*nx] = D
        A[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = M
        A[i*nx:(i+1)*nx,(i+1)*nx:(i+2)*nx] = D
   
    A = A.tocsc()

    return A

# Aproximation of F(u) by finite volumes
def conv_diff_u(u, v, Re, mesh):

    # Recovery dimensions
    X,Y = mesh
    nodes_y, nodes_x = X.shape
    # Interior nodes
    nx = nodes_x - 2
    ny = nodes_y - 2

    # Output
    Fu = np.zeros((nodes_y, nodes_x))

    # Hinges
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    for i in range(1, nx):
        for j in range(1, ny+1):

            u_w = (u[j,i] + u[j, i-1]) * 0.5
            u_e = (u[j,i+1] + u[j, i]) * 0.5
            v_n = (v[j,i+1] + v[j, i]) * 0.5
            v_s = (v[j-1,i+1] + v[j-1, i]) * 0.5

            a_w = dy * (1 / (Re * dx) + u_w * 0.5)
            a_e = dy * (1 / (Re * dx) - u_e * 0.5)
            a_n = dx * (1 / (Re * dy) - v_n * 0.5)
            a_s = dx * (1 / (Re * dy) + v_s * 0.5)
            a_p = a_w + a_e + a_n + a_s

            Fu[j,i] =  a_w * u[j,i-1] + a_e * u[j,i+1] + a_n * u[j+1,i] + a_s * u[j-1,i] - a_p * u[j,i]

    return Fu

# Aproximation of F(v) by finite volumes
def conv_diff_v(u, v, Re, mesh):

    # Recovery dimensions
    X,Y = mesh
    nodes_y, nodes_x = X.shape
    # Interval numbers
    nx = nodes_x - 1
    ny = nodes_y - 1

    # Output
    Fv = np.zeros((nodes_y, nodes_x))

    # Hinges
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    for i in range(1, nx):
        for j in range(1,ny):

            u_w = (u[j+1,i-1] + u[j, i-1]) * 0.5
            u_e = (u[j+1,i] + u[j, i]) * 0.5
            v_n = (v[j+1,i] + v[j, i]) * 0.5
            v_s = (v[j,i] + v[j-1, i]) * 0.5

            a_w = dy * (1 / (Re * dx) + u_w * 0.5)
            a_e = dy * (1 / (Re * dx) - u_e * 0.5)
            a_n = dx * (1 / (Re * dy) - v_n * 0.5)
            a_s = dx * (1 / (Re * dy) + v_s * 0.5)
            a_p = a_w + a_e + a_n + a_s

            Fv[j,i] =  a_w * v[j,i-1] + a_e * v[j,i+1] + a_n * v[j+1,i] + a_s * v[j-1,i] - a_p * v[j,i]

    return Fv

def gradient(p, mesh):
    """
    Calculus of gradient of pressures

    """
    # Recovery dimensions
    X,Y = mesh
    nodes_y, nodes_x = X.shape
    # Interior nodes
    nx = nodes_x - 2
    ny = nodes_y - 2

    # Hinges
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    # Gradient Components
    grad_pu = np.zeros((ny+2, nx+2))
    grad_pv = np.zeros((ny+2, nx+2))

    # Es division pero no se por que funciona mejor producto
    grad_pu[1:-1,1:-1] = 0.5 * (p[1:-1,2:] - p[1:-1,1:-1]) * dx
    grad_pv[1:-1,1:-1] = 0.5 * (p[2:,1:-1] - p[1:-1,1:-1]) * dy
    
    ## Boundary conditions ##
    grad_pu[1:-1, 0] = grad_pu[1:-1, 1]
    grad_pu[1:-1, nx+1] = grad_pu[1:-1, -1]
    grad_pv[1:-1, 0] = grad_pv[1:-1, 1]
    grad_pv[1:-1, nx+1] = grad_pv[1:-1, -1]

    grad_pu[0, 1:-1] = grad_pu[1, 1:-1]
    grad_pu[ny+1, 1:-1] = grad_pu[-1, 1:-1]
    grad_pv[0, 1:-1] = grad_pv[1, 1:-1]
    grad_pv[ny+1, 1:-1] = grad_pv[-1, 1:-1]

    # Corners conditions
    grad_pu[0,0] = 0.5 * (grad_pu[1,0] + grad_pu[0,1])
    grad_pu[0,nx+1] = 0.5 * (grad_pu[0,-1] + grad_pu[1,nx+1])
    grad_pu[ny+1,0] = 0.5 * (grad_pu[ny,0] + grad_pu[ny+1,1])
    grad_pu[ny+1,nx+1] = 0.5 * (grad_pu[ny,nx+1] + grad_pu[ny+1,nx])

    grad_pv[0,0] = 0.5 * (grad_pv[1,0] + grad_pv[0,1])
    grad_pv[0,nx+1] = 0.5 * (grad_pv[0,-1] + grad_pv[1,nx+1])
    grad_pv[ny+1,0] = 0.5 * (grad_pv[ny,0] + grad_pv[ny+1,1])
    grad_pv[ny+1,nx+1] = 0.5 * (grad_pv[ny,nx+1] + grad_pv[ny+1,nx])

    return (grad_pu, grad_pv)

def divergence(u_star, v_star, dx, dy, nx, ny):
    
    result = np.zeros((ny,nx)) 
    
    result = 0.5 * (dy * (u_star[1:ny+1,1:nx+1] - u_star[1:ny+1,0:nx]) + dx * (v_star[1:ny+1,1:nx+1] - v_star[0:ny,1:nx+1]))

    return  result

