# ===============================================================================================================
# SOURCE: https://github.com/eXascaleInfolab/bench-vldb20
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# The code was transformed from the C++ original version to a python version
# ===============================================================================================================


"""NumPy/SciPy translation of the supplied C++ SVT implementation.

Input convention: (rows, columns), with NaN denoting missing entries.
The algorithm returns the FULL reconstructed matrix, not only filled gaps.

Faithful to SVT.cpp / SVT.h, including the rank >= 1 rule, the SVD
thresholds, the stopping conditions, and the final NaN sentinel.

Numerical identity with Armadillo is not guaranteed: its sparse svds solver,
SVD backend, and floating-point summation can differ from SciPy's.
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import svds

TOL = 1e-4
MAX_ITER = 100
INCRE = 5
R_INC = 4
_SENTINEL = np.sqrt(np.finfo(np.float64).max / 100000.0)


def update_sparse(shape, y, omega):
    """C++ SVT::updateSparse: Y[omega[k]] = y[k], column-major indices."""
    rows = omega % shape[0]
    cols = omega // shape[0]
    return sparse.csc_matrix((y, (rows, cols)), shape=shape)


def x_on_omega(u_s, v, omega):
    """C++ SVT::XonOmega, including its FULLMATRIX decision."""
    m, n = u_s.shape[0], v.shape[0]
    if m * n < 50 * 50 or omega.size == m * n:
        return (u_s @ v.T).ravel(order="F")[omega]
    row = omega % m
    col = omega // m
    return np.einsum("ij,ij->i", u_s[row], v[col])


def _economy_svd(a):
    """Return descending singular values and corresponding U, V (not Vh)."""
    u, s, vh = linalg.svd(a, full_matrices=False)
    return u, s, vh.T


def _sparse_svd(y, s):
    """Sparse largest-s singular triplets, sorted in descending order."""
    min_dim = min(y.shape)
    if s == min_dim:
        # scipy.sparse.linalg.svds requires k < min(shape), unlike Armadillo's
        # svds. Dense fallback computes the same full set, not bitwise equal.
        return _economy_svd(y.toarray())
    u, sing, vh = svds(y, k=s, which="LM", return_singular_vectors=True)
    order = np.argsort(sing)[::-1]
    return u[:, order], sing[order], vh[order, :].T


def do_svt(x, tau_scale, *, max_iter=MAX_ITER, tol=TOL):
    """C++ SVT::doSVT: returns a new float64 reconstruction matrix.

    Parameters max_iter and tol expose C++ compile-time constants for testing;
    leave them at their defaults to match SVT.h.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim != 2:
        raise ValueError("x must have two dimensions")
    n1, n2 = x.shape
    if n1 == 0 or n2 == 0:
        raise ValueError("SVT needs a nonempty matrix")
    if max_iter < 1:
        raise ValueError("SVT needs at least one iteration")

    r = 3
    if n2 > 30:
        r = (n2 - 1) // 10 + 1
    small_scale = n1 * n2 < 100 * 100

    df = float(r * (n1 + n2 - r))
    m = min(5.0 * df, float(np.floor(0.99 * (n1 * n2) + 0.5)))
    p = m / (n1 * n2)
    tau = 5.0 * np.sqrt(float(n1 * n2)) * tau_scale
    delta = 1.2 / p

    # find_finite() in C++ uses Armadillo's column-major linear indices.
    omega = np.flatnonzero(np.isfinite(x.ravel(order="F")))
    b = x.ravel(order="F")[omega]

    # C++ allocates data of size n_observed, but populates it by testing only
    # !isnan. For +/-inf this overruns the allocation: not safely reproducible.
    if np.isinf(x).any():
        raise ValueError("The C++ implementation has undefined behavior on +/-inf; use finite values and NaN")

    y_mat = update_sparse(x.shape, b, omega)
    rank = 0

    # k0=ceil(tau / (delta * ||Y||_2)) in the C++ code.
    # For empty/all-zero observations, the original has division by zero or
    # NaN-to-integer conversion. Reject those undefined cases explicitly.
    norm_y = float(linalg.svdvals(y_mat.toarray())[0]) if small_scale else float(svds(y_mat, k=1, return_singular_vectors=False)[0]) if min(x.shape) > 1 else float(linalg.svdvals(y_mat.toarray())[0])
    if norm_y == 0.0 or omega.size == 0:
        raise ValueError("C++ SVT initialization is undefined when all observations are zero or absent")
    k0 = np.ceil(tau / (delta * norm_y))
    norm_b = np.linalg.norm(b)
    y = k0 * delta * b
    y_mat = update_sparse(x.shape, y, omega)

    for _ in range(max_iter):
        s = min(rank + R_INC, min(n1, n2))
        if small_scale:
            u, sing, v = _economy_svd(y_mat.toarray())
        else:
            ok = False
            while not ok:
                u, sing, v = _sparse_svd(y_mat, s)
                if sing.size < s:
                    ok = True
                else:
                    ok = sing[s - 1] <= tau or s == min(n1, n2)
                s = min(s + INCRE, min(n1, n2))

        rank = max(int(np.count_nonzero(sing > tau)), 1)
        u = u[:, :rank]
        v = v[:, :rank]
        sing = sing[:rank] - tau  # DO NOT clamp negative values to zero.

        u_s = u * sing[np.newaxis, :]
        pred = x_on_omega(u_s, v, omega)
        rel_res = np.linalg.norm(pred - b) / norm_b
        if rel_res < tol:
            break
        if rel_res > 1e5:
            break
        y = y + delta * (b - pred)
        y_mat = update_sparse(x.shape, y, omega)

    out = u_s @ v.T
    out[np.isnan(out)] = _SENTINEL
    return out


def recoverySVT(matrix, tauScale):
    """Python equivalent of the C++ recoverySVT entry point (array output)."""
    return do_svt(matrix, tauScale)


# Conventional Python alias.
svt = do_svt
