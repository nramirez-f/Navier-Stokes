import numpy as np
from typing import Callable
from aprox_temporal import *
from post_process import *
from poisson import poisson_2D
import time


def navier_stokes_2D(mesh:np.ndarray, gu:Callable[[np.ndarray, np.ndarray], np.ndarray], gv:Callable[[np.ndarray, np.ndarray], np.ndarray],  Re:float, convergence_criteria:float = 5e-6, max_iterations:int = 100000):
    """
    mesh: Domain of the problem
    gu: Dirichlet boundary conditions for component u
    gv: Dirichlet boundary conditions for component v
    Re: Reynols Number (Note: has to be less than 2300 for laminar regime)
    T: Final time
    """
    start_time = time.time()

    ## Recovery Domain ##
    X,Y = mesh
    nodes_y, nodes_x = X.shape
    # Interior points
    nx = nodes_x - 2
    ny = nodes_y - 2  

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

    ## Calculate the matrix of the poisson problem
    A = poisson_system(nx, ny, dx, dy)

    ## Calculate of dt (CFL Condition) ##
    norm_2_u =  np.sqrt(np.sum((dy * dx * (u**2)).reshape((nx+2) * (ny+2))))
    norm_2_v =  np.sqrt(np.sum((dy * dx * (v**2)).reshape((nx+2) * (ny+2))))
    norm_vel = np.sqrt(norm_2_u**2 + norm_2_v**2)
    security_factor = 0.5
    dt_top = dx / norm_vel
    dt = security_factor * dt_top
    dt = 5e-3 # dt of mathlab code

    print('****************************************')
    print(f'dt: {dt}')
    print(f'dt Max(CFL): {dt_top}')
    print(f'dx: {dx}')
    print(f'dy: {dy}')
    print(f'Re: {Re}')
    print('****************************************')

    ## Time Iteraion ##
    t = dt
    n = 1
    relative_error_u = 1
    relative_error_v = 1
    u_old = u.copy()
    v_old = v.copy()
    
    while  ((relative_error_v > convergence_criteria or relative_error_u > convergence_criteria) and n < max_iterations):

        if (n == 1):
            # Intermediate Velocity
            u_star, v_star, Fu, Fv = intermediate_velocity_euler(u , v, Re, mesh, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt)

            # Correction of Intermediate Velocity
            grad_pu, grad_pv = gradient(p, mesh)
            u = u_star - dt * grad_pu
            v = v_star - dt * grad_pv

            ## Impose Dirichlet Boundary Conditions ##
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
            u_star, v_star, Fu, Fv = intermediate_velocity_adamsB(u , v, Re, mesh, Fu_old, Fv_old, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt)

            # Correction of Intermediate Velocity
            grad_pu, grad_pv = gradient(p, mesh)
            u = u_star - dt * grad_pu
            v = v_star - dt * grad_pv

            ## Impose Dirichlet Boundary Conditions ##
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

        ## Calculate Norms & Errors ##
        norm_2_u =  np.sqrt(np.sum((dy * dx * u**2).reshape((nx+2) * (ny+2))))
        norm_2_v =  np.sqrt(np.sum((dy * dx * v**2).reshape((nx+2) * (ny+2))))
        norm_2_p =  np.sqrt(np.sum((dy * dx * p**2).reshape((nx+2) * (ny+2))))

        relative_error_u = np.sqrt(np.sum((dy * dx * np.abs(u - u_old)**2).reshape((nx+2) * (ny+2)))) / norm_2_u
        relative_error_v = np.sqrt(np.sum((dy * dx * np.abs(v - v_old)**2).reshape((nx+2) * (ny+2)))) / norm_2_v

        ## Update
        u_old = u.copy()
        v_old = v.copy()
        Fu_old = Fu.copy()
        Fv_old = Fv.copy()

        ## Manage stops conditions
        if np.any(norm_2_u > 1e6):
            print("U too large!!")
            break
        elif np.any(norm_2_v > 1e6):
            print("V too large!!")
            break
        elif np.any(norm_2_p > 1e6):
            print("P too large!!")
            break
        
        if (relative_error_v < convergence_criteria and relative_error_u < convergence_criteria):
            print('Reach to convergence Criteria')
            break

        if (n > max_iterations):
            print('Reach to max iterations')
            break
        
        t += dt
        n += 1

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)} horas, {int(minutes)} minutos, {seconds:.2f} segundos.")

    variables_contour(mesh, u, v, p, t, n)

    
