# -*- coding: utf-8 -*-

import numpy as np
from aprox_temporal import *
from plot_graphs import *
from poisson import poisson_2D
import time
from datetime import datetime
import os
from colorama import Fore, Style


def navier_stokes_2D(x0:float, xf:float, nx:int, y0:float, yf:float, ny:int, Re:float, nu:float, simulation_name:str,
                     security_factor:float=0, dt:float = 0,
                     convergence_criteria:float=5e-6, max_iterations:int = 200000, sns_flag = 0, sns_step:int=1000,
                     bicgstab_flag:int = 0, bicgstab_rtol:float = 0.001, bicgstab_atol:float = 0, bicgstab_maxiter:int=3000):
    """
    Simulates the 2D Navier-Stokes equations for incompressible fluid flow within a bounded domain.

    Parameters:
    - x0, xf (float): Start and end coordinates of the domain in the x direction.
    - y0, yf (float): Start and end coordinates of the domain in the y direction.
    - nx, ny (int): Number of volumes in the x and y directions, respectively.
    - Re (float): Reynolds number, governing the flow's inertial and viscous forces.
    - nu (float): Dinamic viscosity.
    - simulation_name (str): Name for the simulation; used to organize output files.
    - security_factor (float, optional): Safety factor for calculating the time step. Defaults to 0.
    - dt (float, optional): Time step for the simulation. If <= 0, CFL condition determines dt.
    - convergence_criteria (float, optional): Relative error threshold for stopping. Defaults to 5e-6.
    - bicgstab_flag (int, optional): If 1, solves the Poisson equation using BiCGSTAB; else uses LU. Defaults to 0.
    - bicgstab_rtol (float, optional): Relative tolerance for BiCGSTAB. Defaults to 0.001.
    - bicgstab_atol (float, optional): Absolute tolerance for BiCGSTAB. Defaults to 0.
    - bicgstab_maxiter (int, optional): Maximum iterations for BiCGSTAB. Defaults to 3000.
    - sns_step (int, optional): Number of iterations between saving snapshots. Defaults to 1000.
    - max_iterations (int, optional): Maximum allowed iterations for the simulation. Defaults to 200000.

    This function:
    1. Initializes the velocity (`u`, `v`) and pressure (`p`) fields with Dirichlet boundary conditions.
    2. Solves the Navier-Stokes equations iteratively using temporal approximations and the Poisson equation.
    3. Dynamically updates the velocity and pressure fields until convergence criteria are met or maximum iterations are reached.
    4. Saves simulation snapshots at regular intervals and generates final contour and field plots of the steady-state or last iteration.
    5. Handles errors such as divergence in velocity or pressure and logs simulation progress and results.

    Output:
    - Simulation data and snapshots are saved in a directory named after the simulation.
    - Results include velocity and pressure field plots, simulation info, and total execution time.
    """
    start_time = time.time()
    os.makedirs('simulations', exist_ok=True)
    current_time = datetime.now().strftime("%d%m%Y%H%M%S")
    simulation_dir = f'simulations/{simulation_name}_{current_time}'
    simulation_info_file = f'{simulation_dir}/info.txt'
    snapshots_dir = f'{simulation_dir}/snapshots'
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    # Hinges
    dx = (xf - x0) / nx
    dy = (yf - y0) / ny
     
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

    ## Save Initial Conditions ##
    if (sns_flag == 1):
        np.save(f"{snapshots_dir}/u_sns_{0}.npy", u)
        np.save(f"{snapshots_dir}/v_sns_{0}.npy", v)
        np.save(f"{snapshots_dir}/p_sns_{0}.npy", p)

    ## Calculate the matrix of the poisson problem
    A = poisson_system(nx, ny, dx, dy, bicgstab_flag)

    norm_u =  np.linalg.norm(u)
    #  Time Step by CFL condition
    if (dt <= 0):
        dt_a = 0.35 * (dx / norm_u)
        dt_b = 0.2 * ((dx * dx) / nu)
        dt_cfl = min(dt_a, dt_b)
        if (security_factor <= 0):
            security_factor = 1
        dt = security_factor * dt_cfl

    # Simulation Info
    if bicgstab_flag == 1:
        laplacian_pressure_solver = f'Bicgstab (rtol={bicgstab_rtol}, atol={bicgstab_atol}, maxiter={bicgstab_maxiter})'
    else:
        laplacian_pressure_solver = 'LU'

    simulation_info = f'*** Navier-Stokes 2D Simulation ***\n\nSimulation Name: {simulation_name}\nTime Step: {dt}\nVolumes in x-Axis: {nx}\nVolumes in y-Axis: {ny}\nReynols Number: {Re}\nConvergence Criteria: {convergence_criteria}\nMax Iterations: {max_iterations}\nLaplacian Pressure Solver: {laplacian_pressure_solver}\nSnapshot Step: {sns_step}'
    
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

    while  ((error_v > convergence_criteria or error_u > convergence_criteria) and n <= max_iterations):

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

        ## Update
        u_0 = u.copy()
        v_0 = v.copy()
        p_0 = p.copy()
        Fu_0 = Fu.copy()
        Fv_0 = Fv.copy()

        ## Info of progress
        if (n % sns_step == 0):
            percentage = n / max_iterations
            bar_length = 30
            filled_length = int(bar_length * percentage)
            bar = bar = f"{Fore.GREEN}{'█' * filled_length}{Style.RESET_ALL}{'░' * (bar_length - filled_length)}"
            print(f"\rIterations: [{bar}] {n} / {max_iterations} Time: {t:.2f}s Rel_Err_U: {error_u:.10f} Relative Err_V: {error_v:.10f} ", end="", flush=True)

            ## Save iteration ##
            if (sns_flag == 1):
                sns_n = n // sns_step
                np.save(f"{snapshots_dir}/u_sns_{sns_n}.npy", u)
                np.save(f"{snapshots_dir}/v_sns_{sns_n}.npy", v)
                np.save(f"{snapshots_dir}/p_sns_{sns_n}.npy", p)

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
            info_message = f'\nReach to Convergence Criteria (Rel_Err_U: {error_u}, Rel_Err_V: {error_v}, Rel_Err_P: {error_p})'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(f'\n{info_message}')

            # Image
            plot_contour(dx, dy, nx, ny, u, v, p, f'Steady State t = {t:.2f}s n = {n}', f'{simulation_dir}/steady_state_contour.png')
            plot_field(dx, dy, nx, ny, u, v, p, f'Steady State t = {t:.2f}s n = {n}', f'{simulation_dir}/steady_state_field.png')
            break
        
        n += 1
        
        if (n > max_iterations):
            # Info
            info_message = f'\nReach to Max iterations (Rel_Err_U: {error_u}, Rel_Err_V: {error_v}, Rel_Err_P: {error_p})'
            with open(simulation_info_file, 'a') as file:
                file.write(info_message + '\n')
            print(f'\n{info_message}')

            # Image
            plot_contour(dx, dy, nx, ny, u, v, p, f'Last Iteration t = {t:.2f}s n = {n}', f'{simulation_dir}/last_iteration_contour.png')
            plot_field(dx, dy, nx, ny, u, v, p, f'Last Iteration t = {t:.2f}s n = {n}', f'{simulation_dir}/last_iteration_field.png')
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
