def implicit_qr_step(d, e):
    """
    Generalized Implicit QR step with Wilkinson shift.
    Applicable to SVD (bidiagonal) or EVD (tridiagonal).
    """
    n = len(d)
    # 1. Compute Wilkinson Shift (based on bottom 2x2)
    # Using your existing logic...
    mu = compute_wilkinson_shift(d[-2], d[-1], e[-1])
    
    # 2. Initial bulge creation
    y = d[0] * e[0]
    x = d[0]**2 - mu
    
    for k in range(n - 1):
        c, s = apply_givens(x, y)
        
        # In SVD: Apply Right then Left rotations to maintain bidiagonal structure.
        # In EVD: Apply Similarity (G A G^T) to maintain tridiagonal structure.
        
        # ... update d, e and chase the bulge ...