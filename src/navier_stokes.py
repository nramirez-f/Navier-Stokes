import numpy as np
from typing import Callable
from aprox_temporal import *
from post_process import *
from poisson import poisson_2D
import time


def navier_stokes_2D(mesh:np.ndarray, gu:Callable[[np.ndarray, np.ndarray], np.ndarray], gv:Callable[[np.ndarray, np.ndarray], np.ndarray],  Re:float, T:float, pressure_tolerance:float = 0.001, max_steps_pressure:int = 3000):
    """
    mesh: Domain of the problem
    gu: Dirichlet boundary conditions for component u
    gv: Dirichlet boundary conditions for component v
    Re: Reynols Number (Note: has to be less than 2300 for laminar regime)
    T: Final time
    """

    ## Recovery Domain ##
    X,Y = mesh
    nodes_y, nodes_x = X.shape
    # Interval numbers
    nx = nodes_x - 1
    ny = nodes_y - 1
        
    ## Variables ##
    # Velocity Components 
    u = np.zeros((nodes_y, nodes_x))
    v = np.zeros((nodes_y, nodes_x))
    # Intermediate Velocity Components
    u_star = np.zeros((nodes_y, nodes_x))
    v_star = np.zeros((nodes_y, nodes_x))
    # Pressure
    p = np.zeros((nodes_y, nodes_x))

    ## Dirichlet Boundary Conditions ##
    # Top Boundary
    u[ny, :] = gu(X[ny, :], Y[ny, :])
    v[ny, :] = gv(X[ny, :], Y[ny, :])
    # Botton Boundary
    u[0, :] = gu(X[0, :], Y[0, :])
    v[0, :] = gv(X[0, :], Y[0, :])
    # Right Boundary
    u[:, nx] = gu(X[:, nx], Y[:, nx])
    v[:, nx] = gv(X[:, nx], Y[:, nx])
    # Left Boundary
    u[:, 0] = gu(X[:, 0], Y[:, 0])
    v[:, 0] = gv(X[:, 0], Y[:, 0])

    ## Hinges ##
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    ## Calculate of dt (CFL Condition) ##
    norm_2_u =  np.sum((dy * dx * u**2).reshape(nodes_x * nodes_y))**(1 / 2)
    norm_2_v =  np.sum((dy * dx * v**2).reshape(nodes_x * nodes_y))**(1 / 2)
    norm_vec_u = (norm_2_u + norm_2_v)**(1 / 2)
    dt = dx / norm_vec_u
    
    ## Time Iteraion ##
    t = 0
    u_old = u.copy()
    v_old = v.copy()
    while  (t < T):
        print(f'Working on time {t}')
        if (t == 0):
            # Intermediate Velocity
            print('Calculating intermediate velocity...')
            u_star, v_star, Fu_old, Fv_old = intermediate_velocity_euler(u_old , v_old, Re, mesh, dt)

            # Pressure as Poisson Solution
            print('Solving Poisson Equation...')
            p = poisson_2D(mesh, u_star, v_star, dt) 

            # Correction of Intermediate Velocity
            print('Correcting the velocity...')
            u = u_old - gradient(p, mesh)
            v = v_old - gradient(p, mesh)

            ## Impose Dirichlet Boundary Conditions ##
            print('Imposing Dirichlet Conditions...')
            # Top Boundary
            u[ny, :] = gu(X[ny, :], Y[ny, :])
            v[ny, :] = gv(X[ny, :], Y[ny, :])
            # Botton Boundary
            u[0, :] = gu(X[0, :], Y[0, :])
            v[0, :] = gv(X[0, :], Y[0, :])
            # Right Boundary
            u[:, nx] = gu(X[:, nx], Y[:, nx])
            v[:, nx] = gv(X[:, nx], Y[:, nx])
            # Left Boundary
            u[:, 0] = gu(X[:, 0], Y[:, 0])
            v[:, 0] = gv(X[:, 0], Y[:, 0])

        else:
            # Intermediate Velocity
            print('Calculating intermediate velocity...')
            u_star, v_star, Fu_old, Fv_old = intermediate_velocity_adamsB(u , v, Re, mesh, Fu_old, Fv_old, dt)

            # Pressure as Poisson Solution
            print('Solving Poisson Equation...')
            p = poisson_2D(mesh, u_star, v_star, dt)

            # Correction of Intermediate Velocity
            print('Correcting the velocity...')
            u = u_old - gradient(p, mesh)
            v = v_old - gradient(p, mesh)

            ## Impose Dirichlet Boundary Conditions ##
            print('Imposing Dirichlet Conditions...')
            # Top Boundary
            u[ny, :] = gu(X[ny, :], Y[ny, :])
            v[ny, :] = gv(X[ny, :], Y[ny, :])
            # Botton Boundary
            u[0, :] = gu(X[0, :], Y[0, :])
            v[0, :] = gv(X[0, :], Y[0, :])
            # Right Boundary
            u[:, nx] = gu(X[:, nx], Y[:, nx])
            v[:, nx] = gv(X[:, nx], Y[:, nx])
            # Left Boundary
            u[:, 0] = gu(X[:, 0], Y[:, 0])
            v[:, 0] = gv(X[:, 0], Y[:, 0])

        plot_vector_field_contour(mesh, u, v, p, t)

        u_old = u.copy()
        v_old = v.copy()            
        time.sleep(0.5)
        t += dt
 
    
