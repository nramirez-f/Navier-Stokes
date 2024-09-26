import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field_contour(mesh, u: np.ndarray, v: np.ndarray, p: np.ndarray, t:float):
    """
    """
    X,Y = mesh
    
    plt.figure(figsize=(8, 6))
    
    contour = plt.contourf(X, Y, p, cmap='viridis')
    plt.colorbar(contour)
    
    plt.quiver(X, Y, u, v, color='white')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Aprox on time {t}')
    
    plt.show()