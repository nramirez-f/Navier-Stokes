import numpy as np

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

            u_w = (u[j,i] + u[j, i-1]) / 2
            u_e = (u[j,i+1] + u[j, i]) / 2
            v_n = (v[j,i+1] + v[j, i]) / 2
            v_s = (v[j-1,i+1] + v[j-1, i]) / 2

            a_w = dy * (1 / (Re * dx) + u_w / 2)
            a_e = dy * (1 / (Re * dx) + u_e / 2)
            a_n = dx * (1 / (Re * dy) + v_s / 2)
            a_s = dx * (1 / (Re * dy) + v_n / 2)
            a_p = a_w + a_e + a_n + a_s

            Fu[j,i] =  a_w * u[j,i-1] + a_e * u[j,i+1] + a_n * u[j+1,i] + a_s * u[j-1,i] + a_p * u[j,i]

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

            u_w = (u[j+1,i-1] + u[j, i-1]) / 2
            u_e = (u[j+1,i] + u[j, i]) / 2
            v_n = (v[j+1,i] + v[j, i]) / 2
            v_s = (v[j,i] + v[j-1, i]) / 2

            a_w = dy * (1 / (Re * dx) + u_w / 2)
            a_e = dy * (1 / (Re * dx) + u_e / 2)
            a_n = dx * (1 / (Re * dy) + v_s / 2)
            a_s = dx * (1 / (Re * dy) + v_n / 2)
            a_p = a_w + a_e + a_n + a_s

            Fv[j,i] =  a_w * v[j,i-1] + a_e * v[j,i+1] + a_n * v[j+1,i] + a_s * v[j-1,i] + a_p * v[j,i]

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
    grad_pu = np.zeros((nodes_y, nodes_x))
    grad_pv = np.zeros((nodes_y, nodes_x))

    for i in range(1, nx+1):
        for j in range(1,ny+1):
            grad_pu[j,i] = (p[j,i+1] - p[j,i-1]) / (2 * dx)
            grad_pv[j,i] = (p[j+1,i] - p[j-1,i]) / (2 * dy)
    
    ## Boundary conditions ##
    # U
    for j in range(ny+2):
        grad_pu[j,0] = (p[j,1] - p[j,0]) / dx
        grad_pu[j,nx+1] = (p[j,nx+1] - p[j,nx]) / dx
    # V
    for i in range(nx+2):
        grad_pv[0,i] = (p[1,i] - p[0,i]) / dy
        grad_pv[ny+1,i] = (p[ny+1,i] - p[ny,i]) / dy

    return (grad_pu, grad_pv)  
