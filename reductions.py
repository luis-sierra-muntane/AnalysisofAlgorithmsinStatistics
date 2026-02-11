def reduce_to_bidiagonal(A_in):
    """SVD Phase 1: Bidiagonalization."""
    A = A_in.copy().astype(float)
    m, n = A.shape
    d, e = np.zeros(min(m, n)), np.zeros(min(m, n) - 1)

    for k in range(min(m, n)):
        # Column elimination (Left)
        v, alpha = get_householder(A[k:, k])
        if v is not None:
            A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
        d[k] = alpha

        # Row elimination (Right)
        if k < n - 2:
            v, alpha = get_householder(A[k, k+1:])
            if v is not None:
                A[k:, k+1:] -= 2 * (A[k:, k+1:] @ np.outer(v, v))
            e[k] = alpha
    return d, e