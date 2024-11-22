######################################
###### DRIVEN CAVITY EXAMPLE #########
#####################################

from navier_stokes import navier_stokes_2D

## Mesh Variables ##

# Interval in axis X
x0 = 0
xf = 1

# volumes in axis X
nx = 8

# Interval in axis Y
y0 = 0
yf = 1

# volumes in axis Y
ny = 8

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

navier_stokes_2D(x0, xf, nx, y0, yf, ny, Re, 'Driven-Cavity')
