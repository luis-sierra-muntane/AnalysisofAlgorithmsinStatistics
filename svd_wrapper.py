import argparse
import numpy as np
import sys
import os

# Import the user's custom implementation
# We wrap this in a try-block to ensure the file exists
try:
    import svd_initial_implementation as custom_svd
except ImportError:
    print("\n[Error] Could not import 'svd_initial_implementation.py'.")
    print("Please ensure the file is in the same directory as this wrapper.\n")
    sys.exit(1)

def load_matrix_from_file(file_path):
    """Loads a matrix from a text or CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Try loading as standard whitespace-delimited
        return np.loadtxt(file_path)
    except ValueError:
        try:
            # If that fails, try loading as CSV
            return np.loadtxt(file_path, delimiter=',')
        except Exception as e:
            raise ValueError(f"Could not parse matrix file. Ensure it is numeric. Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="CLI Wrapper for Custom SVD Implementation."
    )
    
    # Input methods (Mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file', '-f', 
        type=str, 
        help="Path to a text/csv file containing the matrix."
    )
    group.add_argument(
        '--random', '-r', 
        type=str, 
        help="Generate a random matrix of size M,N (e.g., '5,4')."
    )
    group.add_argument(
        '--demo', '-d', 
        action='store_true', 
        help="Run a hardcoded 4x3 demo matrix."
    )

    # Tuning parameters
    parser.add_argument('--tol', type=float, default=1e-10, help="Convergence tolerance (default: 1e-10)")
    parser.add_argument('--iter', type=int, default=1000, help="Max iterations (default: 1000)")
    parser.add_argument('--compare', action='store_true', help="Compare results with NumPy SVD")

    args = parser.parse_args()
    
    matrix = None

    # --- 1. Load Data ---
    try:
        if args.demo:
            print("--- Running Demo Mode ---")
            matrix = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0]
            ])
            
        elif args.random:
            try:
                rows, cols = map(int, args.random.split(','))
                print(f"--- Generating Random Matrix ({rows}x{cols}) ---")
                matrix = np.random.randn(rows, cols)
            except ValueError:
                print("[Error] Random format must be 'rows,cols' (e.g., -r 10,5)")
                sys.exit(1)
                
        elif args.file:
            print(f"--- Loading Matrix from {args.file} ---")
            matrix = load_matrix_from_file(args.file)

    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    print(f"Input Matrix Shape: {matrix.shape}")
    
    # --- 2. Run Custom SVD ---
    print("\nComputing SVD...")
    try:
        # Call the function from the imported file
        singular_values = custom_svd.simple_svd(matrix, tol=args.tol, max_iter=args.iter)
        
        # Sort descending for standard presentation
        singular_values.sort()
        singular_values = singular_values[::-1]
        
        print("\n" + "="*30)
        print("  CALCULATED SINGULAR VALUES")
        print("="*30)
        print(singular_values)
        
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        sys.exit(1)

    # --- 3. Comparison (Optional) ---
    if args.compare:
        print("\n" + "-"*30)
        print("  NUMPY REFERENCE CHECK")
        print("="*30)
        np_vals = np.linalg.svd(matrix, compute_uv=False)
        np_vals.sort()
        np_vals = np_vals[::-1]
        
        print(f"NumPy:  {np_vals}")
        
        # Calculate Error
        if len(singular_values) == len(np_vals):
            mse = np.mean((singular_values - np_vals)**2)
            print(f"\nMean Squared Error: {mse:.6e}")
        else:
            print("\nDimension mismatch in singular values (cannot compute error).")

if __name__ == "__main__":
    main()
