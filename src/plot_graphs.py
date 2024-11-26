# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_contour(dx: float, dy: float, nx: int, ny: int, u: np.ndarray, v: np.ndarray, p: np.ndarray, title: str, save_path: str):
    """
    Generates a contour plot for the velocity components (u, v) and pressure (p) fields,
    including level lines with labeled values.
    """
    x = np.linspace(0 - dx, 1 + dx, nx + 2)
    y = np.linspace(0 - dy, 1 + dy, ny + 2)
    mesh = np.meshgrid(x, y)
    X, Y = mesh

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot U
    contour1 = axes[0].contourf(X, Y, u, cmap='viridis')  # Filled contours
    level_lines_u = axes[0].contour(X, Y, u, colors='black', linewidths=0.5)  # Line contours
    axes[0].clabel(level_lines_u, inline=True, fmt='%.2f', fontsize=8)  # Label lines
    axes[0].set_title('U')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(contour1, ax=axes[0])

    # Plot V
    contour2 = axes[1].contourf(X, Y, v, cmap='viridis')
    level_lines_v = axes[1].contour(X, Y, v, colors='black', linewidths=0.5)
    axes[1].clabel(level_lines_v, inline=True, fmt='%.2f', fontsize=8)
    axes[1].set_title('V')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    fig.colorbar(contour2, ax=axes[1])

    # Plot P
    contour3 = axes[2].contourf(X, Y, p, cmap='viridis')
    level_lines_p = axes[2].contour(X, Y, p, colors='black', linewidths=0.5)
    axes[2].clabel(level_lines_p, inline=True, fmt='%.2f', fontsize=8)
    axes[2].set_title('P')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    fig.colorbar(contour3, ax=axes[2])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_field(dx:float, dy:float, nx:int, ny:int, u: np.ndarray, v: np.ndarray, p: np.ndarray, title: str, save_path: str):
    """
    Generates a combined plot of the pressure field (contour plot) and velocity field (vector quiver plot).

    Parameters:
    - dx, dy (float): Grid spacing in the x and y directions, respectively.
    - nx, ny (int): Number of grid points in the x and y directions.
    - u, v (np.ndarray): Velocity field components in the x and y directions.
    - p (np.ndarray): Pressure field values.
    - title (str): Title for the figure.
    - save_path (str): Path to save the generated plot as an image file.

    This function creates a single plot with:
    - A contour plot of the pressure field `p`.
    - A quiver plot of the velocity field `(u, v)` overlaid on the pressure contour.

    The velocity vectors are scaled dynamically to highlight the magnitude of the velocity field.
    The plot includes a color bar for the pressure contour and labels for clarity.
    The final figure is saved as a high-resolution image at the specified path.
    """
    x = np.linspace(0 - dx, 1 + dx, nx + 2)
    y = np.linspace(0 - dy, 1 + dy, ny + 2)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 10))

    pressure_contour = ax.contourf(X, Y, p, levels=20, cmap='viridis')
    fig.colorbar(pressure_contour, ax=ax, label="P")

    norm = np.sqrt(u**2 + v**2)

    ax.quiver(
        X[1:-1], Y[1:-1],
        u[1:-1], v[1:-1],
        color='white', 
        scale=None,
        scale_units='xy',
        pivot='tail',
        label='Velocity Field',
        cmap='cool', 
        clim=[0, np.max(norm)]
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
