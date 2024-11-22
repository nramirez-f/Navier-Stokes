import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field_contour(mesh, u: np.ndarray, v: np.ndarray, p: np.ndarray, t:float):
    """
    """
    X,Y = mesh
    
    plt.clf()

    plt.figure(figsize=(8, 6))
    
    contour = plt.contourf(X, Y, p, cmap='viridis')
    plt.colorbar(contour)
    
    quiv = plt.quiver(X, Y, u, v, color='white', units='width')
    plt.quiverkey(quiv, 0.9, 0.9, 2, r'm/s', labelpos='E', coordinates='figure')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Aprox at time {t}')
    
    plt.pause(5)

def velocities_contour(mesh, u: np.ndarray, v: np.ndarray, title:str='Initial Conditions'):
    """
    """
    X, Y = mesh
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot U
    contour1 = axes[0].contourf(X, Y, u, cmap='viridis')
    axes[0].set_title('U (m/s)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(contour1, ax=axes[0])

    # Plot V
    contour2 = axes[1].contourf(X, Y, v, cmap='viridis')
    axes[1].set_title('V (m/s)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    fig.colorbar(contour2, ax=axes[1])

    plt.suptitle(title, fontsize=16)
    plt.pause(5)
    plt.close(fig)

def intermediate_divergence(mesh, div:np.ndarray):
    """
    """
    X, Y = mesh
    
    plt.figure(figsize=(6, 6))

    # Plot Div
    contour = plt.contourf(X, Y, div, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.suptitle('Divergence', fontsize=16)
    plt.pause(5)
    plt.close('All')

def pressure_contour(mesh, p:np.ndarray):
    """
    """
    X, Y = mesh
    
    plt.figure(figsize=(6, 6))

    # Plot P
    contour = plt.contourf(X, Y, p, cmap='viridis')
    plt.colorbar(contour)
    plt.title('P (Pa)')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.suptitle('Pressure', fontsize=16)
    plt.pause(5)
    plt.close('All')

def variables_contour(dx, dy, nx, ny, u: np.ndarray, v: np.ndarray, p: np.ndarray, t: float, n:int):
    """
    """

    x = np.linspace(0-dx, 1+dx, nx+2)
    y = np.linspace(0-dy, 1+dy, ny+2)
    mesh = np.meshgrid(x,y)
    X, Y = mesh
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot U
    contour1 = axes[0].contourf(X, Y, u, cmap='viridis')
    axes[0].set_title('U (m/s)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(contour1, ax=axes[0])

    # Plot V
    contour2 = axes[1].contourf(X, Y, v, cmap='viridis')
    axes[1].set_title('V (m/s)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    fig.colorbar(contour2, ax=axes[1])

    # Plot P
    contour3 = axes[2].contourf(X, Y, p, cmap='viridis')
    axes[2].set_title('P (Pa)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    fig.colorbar(contour3, ax=axes[2])

    plt.suptitle(f'Approximation at time {t} (n={n})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.pause(3)
    #plt.close(fig)
    plt.show()

def mesh_to_txt(u, v, p, n):
    """
    """
    directory = 'iterations' 
    with open(f'{directory}/U_n{n}.txt', 'w') as f:
        for row in u:
            row_string = '\t'.join(f'{val:.6f}' for val in row)
            f.write(row_string + '\n')
    
    with open(f'{directory}/V_n{n}.txt', 'w') as f:
        for row in v:
            row_string = '\t'.join(f'{val:.6f}' for val in row)
            f.write(row_string + '\n')

    with open(f'{directory}/P_n{n}.txt', 'w') as f:
        for row in p:
            row_string = '\t'.join(f'{val:.6f}' for val in row)
            f.write(row_string + '\n')