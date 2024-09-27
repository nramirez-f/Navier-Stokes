import numpy as np
from typing import Callable
from aprox_temporal import *
from post_process import *
from poisson import poisson_2D
import time


def navier_stokes_2D(mesh:np.ndarray, gu:Callable[[np.ndarray, np.ndarray], np.ndarray], gv:Callable[[np.ndarray, np.ndarray], np.ndarray],  Re:float, convergence_criteria:float = 5e-6):
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
    # Interior points
    nx = nodes_x - 2
    ny = nodes_y - 2

    print(f'Interior nodes axis X: {nx}')   
    print(f'Interior nodes axis Y: {ny}')
    print(f'Cells Numbers: {(nx+1) * (ny+1)}')   

    ## Variables ##
    # Initial Velocity Components (suppose flow is rest)
    u = np.zeros((ny+2, nx+2))
    v = np.zeros((ny+2, nx+2))
    p = np.zeros((ny+2, nx+2))

    ## Dirichlet Boundary Conditions ##
    # Top Boundary
    u[ny+1, :] = gu(X[ny+1, :], Y[ny+1, :])
    v[ny+1, :] = gv(X[ny+1, :], Y[ny+1, :])
    # Botton Boundary
    u[0, :] = gu(X[0, :], Y[0, :])
    v[0, :] = gv(X[0, :], Y[0, :])
    # Right Boundary
    u[:, nx+1] = gu(X[:, nx+1], Y[:, nx+1])
    v[:, nx+1] = gv(X[:, nx+1], Y[:, nx+1])
    # Left Boundary
    u[:, 0] = gu(X[:, 0], Y[:, 0])
    v[:, 0] = gv(X[:, 0], Y[:, 0])

    ## Hinges ##
    dx = np.diff(X[0,:])[0]
    dy = np.diff(Y[:, 0])[0]

    print(f'dx: {dx}')
    print(f'dy: {dy}')

    ## Calculate of dt (CFL Condition) ##
    norm_2_u =  np.sqrt(np.sum((dy * dx * u**2).reshape(nodes_x * nodes_y)))
    norm_2_v =  np.sqrt(np.sum((dy * dx * v**2).reshape(nodes_x * nodes_y)))
    norm_vec_u = np.sqrt(norm_2_u**2 + norm_2_v**2)
    security_factor = 0.5
    cfl_top = dx / norm_vec_u
    dt = security_factor * cfl_top

    print(f'dt: {dt}')

    ## Plot Initial Conditions ##
    velocities_contour(mesh, u, v)
    
    ## Time Iteraion ##
    t = dt
    n = 1
    error_u = 1
    error_v = 1
    u_old = u.copy()
    v_old = v.copy()
    while  (n < 3):
        print(f'Working on time {t}...')
        if (n == 1):
            # Intermediate Velocity
            print('Calculating intermediate velocity...')
            u_star, v_star, Fu_old, Fv_old = intermediate_velocity_euler(u_old , v_old, Re, mesh, dt)

            velocities_contour(mesh, u, v, 'Intermediate velocities (Euler)')

            # Pressure as Poisson Solution
            print('Solving Poisson Equation...')
            p = poisson_2D(mesh, u_star, v_star, dt)

            pressure_contour(mesh, p)

            # Correction of Intermediate Velocity
            print('Correcting the velocity...')
            grad_pu, grad_pv = gradient(p, mesh)
            u = u_old - grad_pu
            v = v_old - grad_pv

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

            velocities_contour(mesh, u, v, 'Intermediate velocities (Adams Bashford)')

            # Pressure as Poisson Solution
            print('Solving Poisson Equation...')
            p = poisson_2D(mesh, u_star, v_star, dt)

            pressure_contour(mesh, p)

            # Correction of Intermediate Velocity
            print('Correcting the velocity...')
            grad_pu, grad_pv = gradient(p, mesh)
            u = u_old - grad_pu
            v = v_old - grad_pv

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

        ## Calculate Errors ##
        print('Calculating Errors...')
        norm_2_u =  np.sqrt(np.sum((dy * dx * u**2).reshape(nodes_x * nodes_y)))
        norm_2_v =  np.sqrt(np.sum((dy * dx * v**2).reshape(nodes_x * nodes_y)))

        error_u = np.sqrt(np.sum((dy * dx * np.abs(u - u_old)**2).reshape(nodes_x * nodes_y))) / norm_2_u
        error_v = np.sqrt(np.sum((dy * dx * np.abs(v - v_old)**2).reshape(nodes_x * nodes_y))) / norm_2_v

        print('Copiyng Solutions...')
        u_old = u.copy()
        v_old = v.copy()

        
        if np.any(np.abs(u) > 1e6):
            print("U too large!!")
            break
        elif np.any(np.abs(v) > 1e6):
            print("V too large!!")
            break
        elif np.any(np.abs(p) > 1e6):
            print("P too large!!")
            break
        
        variables_contour(mesh, u, v, p, t)
        
        t += dt
        n += 1
    
