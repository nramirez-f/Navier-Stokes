# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable
from aprox_temporal import *
from post_process import *
from poisson import poisson_2D
import time
from datetime import datetime
import pyvista as pv
import os
from colorama import Fore, Style


def navier_stokes_2D(x0, xf, nx, y0, yf, ny, Re:float, simulation_name:str, convergence_criteria:float=5e-6,
                      bicgstab_flag:int = 0, bicgstab_rtol:float = 0.001, bicgstab_atol:float = 0, bicgstab_maxiter:int=3000,
                      sns_step:int=1000, max_iterations:int = 200000):
    """
    mesh: Domain of the problem
    gu: Dirichlet boundary conditions for component u
    gv: Dirichlet boundary conditions for component v
    Re: Reynols Number (Note: has to be less than 2300 for laminar regime)
    T: Final time
    """
    start_time = time.time()
    os.makedirs('simulations', exist_ok=True)
    current_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    simulation_dir = f'simulations/{simulation_name}_{current_time}'
    vtk_dir = f'{simulation_dir}/vtk'
    simulation_info_file = f'{simulation_dir}/info.txt'
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(vtk_dir, exist_ok=True)

    # Hinges
    dx = (xf - x0) / nx
    dy = (yf - y0) / ny
    
    ## Pyvista Configuration ##
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
    A = poisson_system(nx, ny, dx, dy, bicgstab_flag)

    ## Calculate of dt (CFL Condition) ##
    """ norm_2_u =  np.sqrt(np.sum((dy * dx * (u**2)).reshape((nx+2) * (ny+2))))
    norm_2_v =  np.sqrt(np.sum((dy * dx * (v**2)).reshape((nx+2) * (ny+2))))
    norm_vel = np.sqrt(norm_2_u**2 + norm_2_v**2)
    security_factor = 0.5
    dt_top = dx / norm_vel
    dt = security_factor * dt_top """
    dt = 5e-3

    if bicgstab_flag == 1:
        laplacian_pressure_solver = f'Bicgstab (rtol={bicgstab_rtol}, atol={bicgstab_atol}, maxiter={bicgstab_maxiter})'
    else:
        laplacian_pressure_solver = 'LU'
    simulation_info = f'*** Navier-Stokes 2D Simulation ***\n\nSimulation Name: {simulation_name}\nTime Step: {dt}\nVolumes in x-Axis: {nx}\nVolumes in y-Axis: {ny}\nReynols Number: {Re}\nConvergence Criteria: {convergence_criteria}\nMax Iterations: {max_iterations}\nLaplacian Pressure Solver: {laplacian_pressure_solver}'
    with open(simulation_info_file, 'w') as file:
        file.write(simulation_info)
    
    print(f'{simulation_info}\n')

    ## Time Iteraion ##
    t = dt
    n = 1
    error_u = 1
    error_v = 1
    u_0 = u.copy()
    v_0 = v.copy()
    p_0 = p.copy()

    while  ((error_v > convergence_criteria or error_u > convergence_criteria) and n < max_iterations):

        if (n == 1):
            # Intermediate Velocity
            u_star, v_star, Fu, Fv = intermediate_velocity_euler(u, v, Re, nx, ny, dx, dy, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt, bicgstab_flag, bicgstab_rtol, bicgstab_atol, bicgstab_maxiter)

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
            u_star, v_star, Fu, Fv = intermediate_velocity_adamsB(u, v, Re, nx, ny, dx, dy, Fu_0, Fv_0, dt)

            # Pressure as Poisson Solution
            p = poisson_2D(nx, ny, A, u_star, v_star, p, dx, dy, dt, bicgstab_flag, bicgstab_rtol, bicgstab_atol, bicgstab_maxiter)

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
        error_p = np.linalg.norm(p - p_0) / norm_p

        ## Generate files with Pyvista ## 
        if (n % sns_step == 0):
            grid["velocity"] = np.c_[u.ravel(), v.ravel(), np.zeros((ny+2) * (nx+2))]
            grid["pressure"] = p.ravel()

            filename = f"{vtk_dir}/n_{n:06d}.vtk"
            grid.save(filename)

        ## Update
        u_0 = u.copy()
        v_0 = v.copy()
        p_0 = p.copy()
        Fu_0 = Fu.copy()
        Fv_0 = Fv.copy()

        ## Info of progress
        if (n % 1000 == 0):
            percentage = n / max_iterations
            bar_length = 30
            filled_length = int(bar_length * percentage)
            bar = bar = f"{Fore.GREEN}{'█' * filled_length}{Style.RESET_ALL}{'░' * (bar_length - filled_length)}"
            print(f"\rIterations: [{bar}] {n} / {max_iterations} Time: {t:.2f}s Rel_Err_U: {error_u:.10f} Relative Err_V: {error_v:.10f} ", end="", flush=True)

        ## Manage stops conditions on error
        if np.any(norm_u > 1e6):
            info_message = '\nRunning Error: U too large'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(info_message)
            break
        elif np.any(norm_v > 1e6):
            info_message = '\nRunning Error: V too large'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(info_message)
            break
        elif np.any(norm_p > 1e6):
            info_message = '\nRunning Error: V too large'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(info_message)
            break
        
        if (error_v < convergence_criteria and error_u < convergence_criteria):
            # Info
            info_message = f'\nReach to Convergence Criteria (Rel_Err_U: {error_u}, Rel_Err_V: {error_v})'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(f'\n{info_message}')


            # Save vtk
            grid["velocity"] = np.c_[u.ravel(), v.ravel(), np.zeros((ny+2) * (nx+2))]
            grid["pressure"] = p.ravel()

            filename = f"{vtk_dir}/n_{n:06d}.vtk"
            grid.save(filename)

            # Image
            variables_contour(dx, dy, nx, ny, u, v, p, t, n, f'Steady State t = {t:.2f}s n = {n}', f'{simulation_dir}/steady_state.png')
            break
        
        n += 1
        
        if (n > max_iterations):
            # Info
            info_message = '\nReach to Convergence Criteria'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(f'\n{info_message}')
            
            # Save vtk
            grid["velocity"] = np.c_[u.ravel(), v.ravel(), np.zeros((ny+2) * (nx+2))]
            grid["pressure"] = p.ravel()

            filename = f"{vtk_dir}/n_{n:06d}.vtk"
            grid.save(filename)

            # Image
            variables_contour(dx, dy, nx, ny, u, v, p, t, n, f'Last Iteration t = {t:.2f}s n = {n}', f'{simulation_dir}/last_iteration.png')
            break
            
        t += dt

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time_info = f'Total Simulation Time: {int(hours)} horas, {int(minutes)} minutos, {seconds:.2f} segundos'
    with open(simulation_info_file, 'a') as file:
        file.write(total_time_info)
    print(f'\n{total_time_info}')
