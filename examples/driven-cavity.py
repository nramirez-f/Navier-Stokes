######################################
###### DRIVEN CAVITY EXAMPLE #########
#####################################

from navier_stokes import navier_stokes_2D

## Mesh Variables ##

# Interval in axis X
x0 = 0
xf = 1

# volumes in axis X
nx = 10

# Interval in axis Y
y0 = 0
yf = 1

# volumes in axis Y
ny = 10

## Phisical Variables ##

# Dinamic Viscosity
nu = 0.2390
# Characteristic Velocity
U = 0.1
# Density
rho = 933
# Characteristic Lenght (meters)
L = 1
# Reynols Number
Re = 100

navier_stokes_2D(x0, xf, nx, y0, yf, ny, Re, f'Driven-Cavity_n{nx}X{ny}_Re{Re}_sf0_cc{5e-6}_itf{0}', security_factor=0, dt=5e-3, convergence_criteria=5e-6, bicgstab_flag=0)

""" dimensions = [10, 20, 30, 40, 50]

Re_numbers = [100, 500, 1000, 1500, 2000]

iterative_flag = [0, 1]

convergence_criteria = [5e-6, 1e-6]

Security_Factors = [1, 0.75, 0.5, 0.25]

for n, Re, itf, cc in zip(dimensions, Re_numbers, iterative_flag, convergence_criteria):

    navier_stokes_2D(x0, xf, n, y0, yf, n, Re, f'Driven-Cavity_n{n}X{n}_Re{Re}_sf0_cc{cc}_itf{itf}', security_factor=0, dt=5e-3, convergence_criteria=cc, bicgstab_flag=itf)

    for sf in Security_Factors:
        navier_stokes_2D(x0, xf, n, y0, yf, n, Re, f'Driven-Cavity_n{n}X{n}_Re{Re}_sf{sf}_cc{cc}_itf{itf}', security_factor={sf}, dt=0, convergence_criteria=cc, bicgstab_flag=itf) """