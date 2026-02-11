import numpy as np
import time
from spectralkit.svd_solver import simple_svd
from spectralkit.eigen_solver import compute_eigen

def run_benchmarks(sizes=[10, 50, 100, 200]):
    print(f"{'Size':<10} | {'Method':<15} | {'Rel. Error':<15} | {'Time (s)':<10}")
    print("-" * 60)

    for n in sizes:
        # Generate a challenging test matrix (ill-conditioned)
        A = np.random.randn(n, n)
        
        # Benchmark Custom SVD
        start = time.time()
        sigma_custom = simple_svd(A)
        custom_time = time.time() - start
        
        # Benchmark NumPy (Reference)
        start = time.time()
        sigma_numpy = np.linalg.svd(A, compute_uv=False)
        numpy_time = time.time() - start
        
        # Calculate Relative Error
        sigma_custom.sort()
        sigma_numpy.sort()
        rel_error = np.linalg.norm(sigma_custom - sigma_numpy) / np.linalg.norm(sigma_numpy)
        
        print(f"{n:<10} | {'Custom SVD':<15} | {rel_error:<15.2e} | {custom_time:<10.4f}")
        print(f"{n:<10} | {'NumPy SVD':<15} | {'0.00e+00':<15} | {numpy_time:<10.4f}")
        print("-" * 60)

if __name__ == "__main__":
    run_benchmarks()