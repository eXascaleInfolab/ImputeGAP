# ===============================================================================================================
# SOURCE: https://github.com/eXascaleInfolab/bench-vldb20
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# The code was transformed from the C++ original version to a python version
# ===============================================================================================================


"""
Direct NumPy/SciPy translation of the supplied C++ DynaMMo implementation.
Input convention: rows are observed variables, columns are time steps.
The supplied C++ interpolation routine is translated literally, including
its missing-block endpoint behavior. Armadillo and NumPy random generators
differ, so numerical identity requires supplying the same initial model.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve


@dataclass
class DynaMMoModel:
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    mu0: np.ndarray
    Q0: np.ndarray


def _interpolate(x):
    """Translate Algebra::Algorithms::interpolate(X, true, &non_observed).

    Deliberately retains the original C++ endpoint indexing: val2 reads
    X[i, block_end], not X[i, block_end + 1]. Consequently, for ordinary
    NaN gaps, this is forward fill, with leading gaps replaced by zero.
    This is *not* NumPy's usual linear interpolation.
    """
    n = x.shape[1]
    for row in x:
        missing_indices = np.flatnonzero(~np.isfinite(row))
        k = 0
        while k < len(missing_indices):
            start = int(missing_indices[k])
            end = start
            k += 1
            while k < len(missing_indices) and missing_indices[k] <= end + 1:
                end = int(missing_indices[k])
                k += 1

            val1 = float(row[start - 1]) if start > 0 else float('nan')
            # C++ uses `block_end < X.n_cols` (always true here), then
            # reads X.at(i, block_end), i.e. the last missing entry.
            val2 = float(row[end]) if end < n else float('nan')
            if np.isnan(val1) and np.isnan(val2):
                val1 = 0.0
                step = 0.0
            elif np.isnan(val1):
                val1 = val2
                step = 0.0
            elif np.isnan(val2):
                step = 0.0
            else:
                step = (val2 - val1) / float(end - start + 2)
            for j in range(end - start + 1):
                row[start + j] = val1 + step * float(j + 1)
    return x


def _is_tiny(sigma):
    # Armadillo norm(matrix, 1) is the maximum absolute column sum.
    return np.linalg.norm(sigma, ord=1) < 1e-10 or np.any(np.diag(sigma) < 1e-10)


def forward(x, model, n, h, fast=False):
    ih = np.eye(h)
    mu = [None] * n
    v = [None] * n
    p = [None] * n
    mu[0] = model.mu0.copy()
    v[0] = model.Q0.copy()

    if fast:
        inv_r = np.linalg.inv(model.R)
        inv_rc = inv_r @ model.C
        inv_crc = model.C.T @ inv_rc

    for i in range(n):
        if i == 0:
            kp = model.Q0
            mu[i] = model.mu0.copy()
        else:
            p[i - 1] = model.A @ v[i - 1] @ model.A.T + model.Q
            kp = p[i - 1]
            mu[i] = model.A @ mu[i - 1]

        if fast:
            sol = solve((np.linalg.inv(kp) + inv_crc).T, inv_rc.T, assume_a='gen').T
            inv_sig = inv_r - sol @ inv_rc.T
        else:
            sigma_c = model.C @ kp @ model.C.T + model.R
            inv_sig = np.linalg.inv(sigma_c)

        k = kp @ model.C.T @ inv_sig
        u_c = model.C @ mu[i]
        delta = x[:, i] - u_c
        mu[i] = mu[i] + k @ delta
        v[i] = (ih - k @ model.C) @ kp

    return mu, v, p


def backward(mu, v, p, model, n):
    ez = [None] * n
    ezz = [None] * n
    ez1z = [None] * n

    ez[n - 1] = mu[n - 1].copy()
    vhat = v[n - 1].copy()
    ezz[n - 1] = vhat + np.outer(ez[n - 1], ez[n - 1])

    for ii in range(n - 1, 0, -1):
        i = ii - 1
        j = solve(p[i].T, (v[i] @ model.A.T).T, assume_a='gen').T
        ez[i] = mu[i] + j @ (ez[i + 1] - model.A @ mu[i])
        ez1z[i] = vhat @ j.T + np.outer(ez[i + 1], ez[i])
        vhat = v[i] + j @ (vhat - p[i]) @ j.T
        ezz[i] = vhat + np.outer(ez[i], ez[i])

    return ez, ezz, ez1z


def mle_lds(n, m, h, x, model, ez, ezz, ez1z):
    sz1z = np.zeros((h, h))
    szz = np.zeros((h, h))
    sxz = np.zeros((m, h))

    for i in range(n - 1):
        sz1z = sz1z + ez1z[i]

    for i in range(n):
        szz += ezz[i]
        sxz += np.outer(x[:, i], ez[i])

    szz_n = szz - ezz[n - 1]
    model.mu0 = ez[0].copy()
    model.Q0 = ezz[0] - np.outer(ez[0], ez[0])
    model.Q0 = np.eye(h) * (np.trace(model.Q0) / float(h))

    model.A = solve(szz_n.T, sz1z.T, assume_a='gen').T

    delta = ((np.trace(szz) - np.trace(ezz[0])
              - 2 * np.trace(model.A @ sz1z.T)
              + np.trace(model.A @ szz_n @ model.A.T)) / float(n - 1)) / float(h)
    model.Q = np.eye(h) * delta

    model.C = solve(szz.T, sxz.T, assume_a='gen').T

    delta = (np.trace(x @ x.T)
             - 2 * np.trace(model.C @ sxz.T)
             + np.trace(model.C @ szz @ model.C.T)) / float(n) / float(m)
    model.R = np.eye(m) * delta


def estimate_missing(n, ez, model):
    return np.column_stack([model.C @ ez[i] for i in range(n)])


def dynammo(x, h=0, max_iter=100, fast=False, *, seed=18931, initial_model=None, interpolate=None):
    """Impute a (variables, time) matrix, returning a new float64 array.

    Parameters mirror doDynaMMo(X, H, maxIter, FAST). ``initial_model``
    allows externally supplied Armadillo initialization for reproducibility.
    The bundled interpolation is a literal translation of the supplied C++.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim != 2:
        raise ValueError('x must be a two-dimensional matrix')
    m, n = x.shape
    if m == 0 or n < 2:
        raise ValueError('DynaMMo requires at least one row and two time steps')
    if h == 0:
        h = m
    if h < 0 or max_iter < 0:
        raise ValueError('h and max_iter must be nonnegative')

    # The C++ calls arma_rng::set_seed(18931). NumPy's generator
    # does not reproduce Armadillo's exact random matrices.
    rng = np.random.default_rng(seed)
    if initial_model is None:
        model = DynaMMoModel(
            np.eye(h) + rng.standard_normal((h, h)),
            np.eye(m, h) + rng.standard_normal((m, h)),
            np.eye(h), np.eye(m), rng.standard_normal(h), np.eye(h))
    else:
        model = DynaMMoModel(*(np.array(getattr(initial_model, field), dtype=np.float64, copy=True) for field in ('A', 'C', 'Q', 'R', 'mu0', 'Q0')))

    missing = ~np.isfinite(x)
    x = _interpolate(x) if interpolate is None else interpolate(x)
    if x.shape != (m, n):
        raise ValueError('interpolate must return a matrix of the same shape')

    iteration = 0
    while (iteration < max_iter and
           not (_is_tiny(model.Q0) or _is_tiny(model.Q) or _is_tiny(model.R))):
        iteration += 1
        mu, v, p = forward(x, model, n, h, fast)
        ez, ezz, ez1z = backward(mu, v, p, model, n)
        mle_lds(n, m, h, x, model, ez, ezz, ez1z)
        y = estimate_missing(n, ez, model)
        x[missing] = y[missing]

    # SharedLibDynaMMo::verifyRecovery replaces NaNs (not infinities).
    x[np.isnan(x)] = np.sqrt(np.finfo(np.float64).max / 100000.0)
    return x


# Optional naming compatibility with the C++ entry point.
def recoveryDynaMMo(matrix, H=0, maxIter=100, fast=False):
    return dynammo(matrix, h=H, max_iter=maxIter, fast=fast)
