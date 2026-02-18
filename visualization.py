import matplotlib.pyplot as plt
import numpy as np

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
    # 'spy' is the direct equivalent to Matlab's function
    plt.spy(mask, markersize=5, color='darkblue')
    
    plt.title(f"{title}\n(Non-zero threshold: {threshold})")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()