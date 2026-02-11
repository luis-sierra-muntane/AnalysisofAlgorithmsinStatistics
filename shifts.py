import numpy as np

def compute_wilkinson_shift(d_m1, d_m, e_m1):
    """
    Computes the Wilkinson shift for the bottom 2x2 block:
    [ d_m1^2 + e_m1^2    d_m1 * e_m1 ]
    [ d_m1 * e_m1        d_m^2       ]
    """
    # Bottom 2x2 of B.T @ B
    a = d_m1**2 + (e_m1**2 if e_m1 else 0)
    b = d_m1 * e_m1
    c = d_m**2
    
    delta = (a - c) / 2.0
    # Stability: ensure we don't divide by zero
    denom = np.abs(delta) + np.sqrt(delta**2 + b**2)
    if denom == 0:
        return c
        
    return c - (b**2) / (np.sign(delta) * denom if delta != 0 else denom)