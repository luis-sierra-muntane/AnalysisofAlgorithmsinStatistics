import numpy as np

def get_norm(x):
    """Computes Euclidean norm safely."""
    return np.linalg.norm(x)

def sign(x):
    """Returns 1.0 if x >= 0, else -1.0. (Matches LAPACK convention)"""
    return 1.0 if x >= 0 else -1.0

def householder_bidiagonalization(A_in):
    """
    Phase 1: Reduces general matrix A to upper bidiagonal form B.
    Returns the diagonal (d) and super-diagonal (e).
    
    Note: For a full SVD, we would accumulate the Householder 
    vectors into U and V. For clarity, we just return the bands.
    """
    A = A_in.copy().astype(float)
    m, n = A.shape
    
    # We will overwrite A with the bidiagonal form and return the diagonals
    # d = diagonal elements, e = super-diagonal elements
    d = np.zeros(min(m, n))
    e = np.zeros(min(m, n) - 1)
    
    for k in range(min(m, n)):
        # --- 1. Eliminate column k below diagonal (Left Householder) ---
        x = A[k:, k]
        alpha = -sign(x[0]) * get_norm(x)
        
        if get_norm(x) > 1e-15: # Avoid div by zero
            # Construct Householder vector v
            v = x.copy()
            v[0] -= alpha
            v = v / get_norm(v)
            
            # Apply H = I - 2vv' to A from the left: A = H A
            # (only affects rows k to m)
            A[k:, k:] -= 2 * np.outer(v, (v.T @ A[k:, k:]))
            
        d[k] = A[k, k] # Store diagonal
        
        # --- 2. Eliminate row k to the right of super-diag (Right Householder) ---
        if k < n - 2:
            x = A[k, k+1:]
            alpha = -sign(x[0]) * get_norm(x)
            
            if get_norm(x) > 1e-15:
                v = x.copy()
                v[0] -= alpha
                v = v / get_norm(v)
                
                # Apply H = I - 2vv' to A from the right: A = A H
                # (only affects cols k+1 to n)
                A[:, k+1:] -= 2 * (A[:, k+1:] @ np.outer(v, v.T))
            
            e[k] = A[k, k+1] # Store super-diagonal

    # Capture the last super-diagonal element if square or wide
    if m <= n and m > 1:
        e[-1] = A[m-2, m-1]
        
    return d, e

def golub_kahan_svd_step(d, e):
    """
    Phase 2: Performs one pass of the Implicit QR (Golub-Kahan) algorithm.
    It 'chases the bulge' down the bidiagonal matrix.
    """
    m = len(d)
    n = len(e)
    
    # We work with copies to avoid mutating input directly during calc
    d = d.copy()
    e = e.copy()
    
    # --- Wilkinson Shift ---
    # Based on the bottom 2x2 block to accelerate convergence
    # [ d_m-2   e_m-2 ]
    # [   0     d_m-1 ]
    # The shift is the eigenvalue of T = B^T B closest to t_nn
    
    # Bottom 2x2 block of B^T B
    dm = d[-1]; dm1 = d[-2]; em1 = e[-2]
    
    # Calculating the shift mu usually involves finding eigenvalues of
    # [ dm1^2 + em1^2    dm1*em1 ]
    # [ dm1*em1          dm^2    ]
    # For simplicity in this demo, we can just use the corner element 
    # as a naive shift or 0 for "zero-shift QR" if stability is tricky.
    # Here is a robust Wilkinson shift approximation:
    t_mm = dm**2
    t_mm1 = dm1**2 + em1**2
    t_m1m = dm1 * em1
    
    delta = (t_mm1 - t_mm) / 2.0
    sign_delta = 1.0 if delta >= 0 else -1.0
    mu = t_mm - (t_m1m**2) / (delta + sign_delta * np.sqrt(delta**2 + t_m1m**2))
    
    # --- Initial Rotation (Givens) ---
    # We want to annihilate y in [x, y]^T where:
    y = d[0] * e[0]
    x = d[0]**2 - mu
    
    # Chase the bulge
    for k in range(n):
        # 1. Generate Givens rotation to annihilate the bulge
        # In a real implementation, use explicit hypot(x,y) for stability
        r = np.hypot(x, y)
        c = x / r
        s = y / r
        
        # Apply Givens from Right (affects cols k, k+1)
        # Updates B[k, k], B[k, k+1] ...
        if k > 0:
            e[k-1] = r # The super-diagonal element from previous step
        
        dk = d[k]
        ek = e[k]
        dk1 = d[k+1]
        
        # Matrix mult logic for right rotation:
        # [ d_k    e_k ] [ c  -s ]
        # [  0    d_k+1] [ s   c ]
        d[k] = c * dk - s * ek
        e[k] = s * dk + c * ek
        d[k+1] = c * dk1 # The 0 becomes s*d_k1, but wait, next step handles left rot
        bulge = -s * dk1 # This is the "bulge" at (k+1, k) introduced
        d[k+1] = c * dk1

        # 2. Generate Givens rotation to annihilate the NEW bulge at (k+1, k)
        # We apply this from the Left.
        x = d[k]
        y = bulge 
        r = np.hypot(x, y)
        c = x / r
        s = y / r
        
        d[k] = r
        
        # Apply Givens from Left (affects rows k, k+1)
        # [ c   s ] [ e_k      0     ]
        # [-s   c ] [ d_k+1  e_k+1?  ]
        
        # Current values after right-rotation
        old_ek = e[k]
        old_dk1 = d[k+1]
        
        # If k < n-1, there is a next e[k+1]
        if k < n - 1:
            old_ek1 = e[k+1]
            e[k] = c * old_ek - s * old_dk1
            d[k+1] = s * old_ek + c * old_dk1
            
            # The bulge is now pushed to (k, k+2) -- wait, no
            # The bulge moves to the "next" x and y for the next iteration
            bulge = -s * old_ek1
            e[k+1] = c * old_ek1
            
            # Prepare x and y for the NEXT right-rotation
            x = e[k]
            y = bulge
        else:
            e[k] = c * old_ek - s * old_dk1
            d[k+1] = s * old_ek + c * old_dk1
            
    return d, e

def simple_svd(A, tol=1e-10, max_iter=1000):
    """
    Driver function combining Bidiagonalization + Iterative QR
    """
    # 1. Bidiagonalize
    d, e = householder_bidiagonalization(A)
    
    # 2. Iterative Diagonalization (Golub-Kahan)
    # We iterate until the super-diagonal e is effectively zero
    for _ in range(max_iter):
        # Check convergence: if e is small, we are done
        if np.max(np.abs(e)) < tol:
            break
            
        # In a real implementation, we would split the matrix if a middle 
        # e[i] is zero (deflation). Here we just run the step on the full chain.
        d, e = golub_kahan_svd_step(d, e)
        
    return np.abs(d) # Singular values are always positive

# --- Example Usage ---
np.set_printoptions(precision=4, suppress=True)

# Create a random matrix
A_test = np.random.randn(5, 4)

# 1. Run our Custom Implementation
sigma_custom = simple_svd(A_test)
sigma_custom.sort() # Sort for comparison (QR doesn't guarantee order)

# 2. Run NumPy's LAPACK Wrapper
sigma_numpy = np.linalg.svd(A_test, compute_uv=False)
sigma_numpy.sort()

print("Custom SVD Sigma:", sigma_custom[::-1]) # Reverse to show Descending
print("NumPy  SVD Sigma:", sigma_numpy[::-1])