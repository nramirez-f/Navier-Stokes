import numpy as np
from typing import Callable
from aprox_temporal import *
from post_process import *
from poisson import poisson_2D
import time
from datetime import datetime
import pyvista as pv
import os


def navier_stokes_2D(x0, xf, nx, y0, yf, ny, Re:float, save:int = 0, convergence_criteria:float = 5e-6, max_iterations:int = 100000):
    """
    mesh: Domain of the problem
    gu: Dirichlet boundary conditions for component u
    gv: Dirichlet boundary conditions for component v
    Re: Reynols Number (Note: has to be less than 2300 for laminar regime)
    T: Final time
    """
    start_time = time.time()
    if save == 1:
        os.makedirs('simulations', exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d-%H:%M")
        output_dir = f'simulations/nv2d-{current_time}'
        os.makedirs(output_dir, exist_ok=True)

    # Hinges
    dx = (xf - x0) / nx
    dy = (yf - y0) / ny
    
    ## Pyvista Configuration ##
    if save == 1:
        grid = pv.StructuredGrid()
        x = np.linspace(0-dx, 1+dx, nx+2)
        y = np.linspace(0-dy, 1+dy, ny+2)
        mesh = np.meshgrid(x,y)
        X, Y = mesh
        grid.points = np.c_[X.ravel(), Y.ravel(), np.zeros((ny+2) * (nx+2))]
        grid.dimensions = [ny+2, nx+2, 1]  

    ## Variables ##
    # Initial Velocity Components (suppose flow is rest)
    u = np.zeros((ny+2, nx+2))
    v = np.zeros((ny+2, nx+2))
    p = np.zeros((ny+2, nx+2))

    ## Dirichlet Boundary Conditions ##
    # Botton Boundary
    u[0, :] = 0
    v[0, :] = 0
    # Right Boundary
    u[:, nx+1] = 0
    v[:, nx+1] = 0
    # Left Boundary
    u[:, 0] = 0
    v[:, 0] = 0
    # Top Boundary
    u[ny+1, :] = 1
    v[ny+1, :] = 0

    ## Calculate the matrix of the poisson problem
    LU = poisson_system(nx, ny, dx, dy)

    ## Calculate of dt (CFL Condition) ##
    """ norm_2_u =  np.sqrt(np.sum((dy * dx * (u**2)).reshape((nx+2) * (ny+2))))
    norm_2_v =  np.sqrt(np.sum((dy * dx * (v**2)).reshape((nx+2) * (ny+2))))
    norm_vel = np.sqrt(norm_2_u**2 + norm_2_v**2)
    security_factor = 0.5
    dt_top = dx / norm_vel
    dt = security_factor * dt_top """
    dt = 5e-3

    """ if save != 1:
        variables_contour(dx, dy, nx, ny, u, v, p, 0, 0) """

    print('****************************************')
    print('** Driven Cavity **')
    print('****************************************')
    print(f'dt: {dt}')
    print(f'problem_nodes_x: {nx}')
    print(f'nodes_x: {nx+2}')
    print(f'dx: {dx}')
    print(f'problem_nodes_y: {ny}')
    print(f'nodes_y: {ny+2}')
    print(f'dy: {dy}')
    print(f'Re: {Re}')
    print('****************************************')

    ## Time Iteraion ##
    t = dt
    n = 1
    error_u = 1
    error_v = 1
    u_0 = u.copy()
    v_0 = v.copy()
    
    while  (error_v > convergence_criteria):

        if (n == 1):
            # Intermediate Velocity
            u_star, v_star, Fu, Fv = intermediate_velocity_euler(u , v, Re, nx, ny, dx, dy, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, LU, u_star, v_star, p, dx, dy, dt)

            # Correction of Intermediate Velocity
            grad_pu, grad_pv = gradient(p, nx, ny, dx, dy)
            u = u_star - dt * grad_pu
            v = v_star - dt * grad_pv

            ## Impose Dirichlet Boundary Conditions ##
            # Botton Boundary
            u[0, :] = 0
            v[0, :] = 0
            # Right Boundary
            u[:, nx+1] = 0
            v[:, nx+1] = 0
            # Left Boundary
            u[:, 0] = 0
            v[:, 0] = 0
            # Top Boundary
            u[ny+1, :] = 1
            v[ny+1, :] = 0

        else:
            # Intermediate Velocity
            u_star, v_star, Fu, Fv = intermediate_velocity_adamsB(u , v, Re, nx, ny, dx, dy, Fu_0, Fv_0, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, LU, u_star, v_star, p, dx, dy, dt)

            # Correction of Intermediate Velocity
            grad_pu, grad_pv = gradient(p, nx, ny, dx, dy)
            u = u_star - dt * grad_pu
            v = v_star - dt * grad_pv

            ## Impose Dirichlet Boundary Conditions ##
            # Botton Boundary
            u[0, :] = 0
            v[0, :] = 0
            # Right Boundary
            u[:, nx+1] = 0
            v[:, nx+1] = 0
            # Left Boundary
            u[:, 0] = 0
            v[:, 0] = 0
            # Top Boundary
            u[ny+1, :] = 1
            v[ny+1, :] = 0

        ## Calculate Norms & Errors ##
        norm_u =  np.linalg.norm(u)
        norm_v =  np.linalg.norm(v)
        norm_p =  np.linalg.norm(p)

        error_u =  np.linalg.norm(u - u_0) / norm_u
        error_v = np.linalg.norm(v - v_0) / norm_v

        ## Generate files with Pyvista ## 
        if (save == 1 and n % 1000 == 0):
            # Añadir campos de datos de velocidad y presión
            grid["velocity"] = np.c_[u.ravel(), v.ravel(), np.zeros((ny+2) * (nx+2))]
            grid["pressure"] = p.ravel()

            filename = f"{output_dir}/ns_n{n:06d}.vtk"
            grid.save(filename)

        ## Update
        u_0 = u.copy()
        v_0 = v.copy()
        Fu_0 = Fu.copy()
        Fv_0 = Fv.copy()

        ## Manage stops conditions on error
        if np.any(norm_u > 1e6):
            print("U too large!!")
            break
        elif np.any(norm_v > 1e6):
            print("V too large!!")
            break
        elif np.any(norm_p > 1e6):
            print("P too large!!")
            break
        
        if (error_v < convergence_criteria):
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

    if save != 1:
        variables_contour(dx, dy, nx, ny, u, v, p, t, n)

    
