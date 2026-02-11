import numpy as np

def get_householder(x):
    """Computes the Householder vector v and the resulting alpha."""
    norm_x = np.linalg.norm(x)
    if norm_x < 1e-15:
        return None, x[0]
    
    # LAPACK-style sign selection for numerical stability
    v = x.copy()
    alpha = -(1.0 if x[0] >= 0 else -1.0) * norm_x
    v[0] -= alpha
    v /= np.linalg.norm(v)
    return v, alpha

def apply_givens(x, y):
    """Returns (c, s) such that [[c, s], [-s, c]] @ [x, y]^T = [r, 0]^T."""
    r = np.hypot(x, y)
    if r < 1e-15:
        return 1.0, 0.0
    return x / r, y / r