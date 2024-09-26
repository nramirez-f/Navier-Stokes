######################################
###### DRIVEN CAVITY EXAMPLE #########
#####################################

import numpy as np
from navier_stokes import navier_stokes_2D

## Mesh Variables ##
# Interval in axis X
x0 = 0
xf = 1
# Domain length in axis X
Lx = np.abs(xf - x0)
# Subintervals in axis X
nx = 10
# Interval in axis Y
y0 = 0
yf = 1
# Domain length in axis X
Ly = np.abs(yf - y0)
# Subintervals in axis Y
ny = 10

## Phisical Variables ##
# Dinamic Viscosity
nu = 0.2390
# Characteristic Velocity
U = 0.1
# Density
rho = 933
# Diameter
D = (Lx * Ly) * 4 / (Lx + Lx + Ly + Ly)
# Characteristic Lenght
L = D * rho
# Reynols Number
Re = L * U / nu

## Mesh ##
x = np.linspace(x0, xf, nx+1)
y = np.linspace(y0, yf, ny+1)
mesh = np.meshgrid(x,y)

## Dirichlet Boundary Conditions ##
# U
def gu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 1, 0)
# V
def gv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)

navier_stokes_2D(mesh, gu, gv, Re, 2)
