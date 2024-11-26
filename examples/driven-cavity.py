######################################
###### DRIVEN CAVITY EXAMPLE #########
#####################################

from navier_stokes import navier_stokes_2D

## Mesh Variables ##

# Interval in axis X
x0 = 0
xf = 1

# volumes in axis X
nx = 30

# Interval in axis Y
y0 = 0
yf = 1

# volumes in axis Y
ny = 30

## Phisical Variables ##

# Dinamic Viscosity
nu = 0.01
# Characteristic Velocity
U = 1
# Characteristic Lenght (meters)
L = 1

iteration_list = [0,1]

dt_list = [0, 5e-3]

nu_list = [0.01, 0.02, 0.001, 1/1800, 0.002]

for itf in iteration_list:
    for dt in dt_list:
        for nu in nu_list:
            # Reynols Number
            Re = (L * U) / nu

            navier_stokes_2D(x0, xf, nx, y0, yf, ny, Re, nu, f'Driven-Cavity_n{nx}X{ny}_Re{Re}_dt{dt}_itf{itf}',
                            security_factor=0, dt = dt,
                            convergence_criteria=5e-6, max_iterations = 500000, sns_flag = 0, sns_step = 1000,
                            bicgstab_flag = itf, bicgstab_rtol = 0.001, bicgstab_atol = 0, bicgstab_maxiter = 3000)
