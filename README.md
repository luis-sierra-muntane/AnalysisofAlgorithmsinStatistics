# Analysis of Algorithms in Statistics
Modern statistics runs on matrix decompositions the same way modern cities run on electricity: quietly, constantly, and absolutely everywhere. Whether you’re fitting regressions, running PCA, exploring latent structure, or stabilizing ill-conditioned problems, you’re depending on a handful of numerical algorithms that do the real heavy lifting: QR, SVD, eigendecompositions, Cholesky, and their friends. In this project we take a look at how these algorithms actually work and why they’re the trustworthy backbone of statistical computation, addressing current limitations and opportunities for our own contributions.

### Files:

`transformations.py`	Householder and Givens kernels.
`reductions.py`	Phase 1: Bidiagonal and Tridiagonal reductions.
`svd_solver.py`	Implicit QR SVD pipeline.
`eigen_solver.py`	EVD pipeline (Symmetric and Hessenberg).
`visualization.py`	Structural inspection tools (e.g., `plot_sparsity`).

### Goals

In order to make these algorithms applicable to really large matrices, we want to randomize the SVD and QR algorithms by using Sketching techniques. These work by multiplying the original matrices by a smaller sketching matrix, thus reducing the dimension of the data matrix and making the subsequent computations much cheaper. The price to pay for doing this is reduced signal accuracy for downstream tasks, but hopefully the loss in accuracy is offset by the gains in speed. Another goal is to quantify and analyze this tradeoff between accuracy and speed.