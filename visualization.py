import numpy as np
import matplotlib.pyplot as plt

def plot_sparsity(A, threshold=1e-15, title="Sparsity Pattern", filename="sparsity_pattern.png"):
    """
    Produces a visualization of the non-zero elements in matrix A.
    
    Parameters:
        A (np.ndarray): The input matrix.
        threshold (float): Elements with absolute value below this are treated as zero.
        title (str): Title for the plot.
        filename (str): The file path to save the resulting image.
    """
    # Create a mask for non-zero elements based on the practitioner's threshold
    # This is more useful than a hard 0 check for numerical matrices.
    mask = np.abs(A) > threshold
    
    plt.figure(figsize=(8, 8))
    # 'spy' is the direct equivalent to MATLAB's function
    plt.spy(mask, markersize=5, color='darkblue')
    
    plt.title(f"{title}\n(Non-zero threshold: {threshold})")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_convergence(history, title="Convergence History", filename="convergence.png"):
    """
    Plots the convergence of the SVD/QR algorithm.
    
    Parameters:
        history (list): A list of values representing the error or 
                        max off-diagonal element at each iteration.
        title (str): Title of the plot.
        filename (str): Path to save the image.
    """
    plt.figure(figsize=(10, 6))
    
    # Use log scale for the y-axis as convergence is often exponential
    plt.semilogy(history, color='tab:red', linewidth=2, marker='o', markersize=4)
    
    plt.title(title)
    plt.xlabel("Iteration / Sweep Number")
    plt.ylabel("Max Off-diagonal Element (Log Scale)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for machine epsilon reference
    plt.axhline(y=1e-15, color='black', linestyle=':', label='Machine Epsilon')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
