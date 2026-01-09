# ===============================================================================================================
# SOURCE: https://github.com/KoheiObata/MissNet/
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.dm.sanken.osaka-u.ac.jp/~yasushi/publications/missnet.pdf
# ===============================================================================================================

import numpy as np
import collections

try:
    from collections.abc import Mapping
except ImportError:  # very old Python
    from collections import Mapping

import warnings

def empirical_covariance(X, assume_centered=False):
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.ndim == 2:
        if assume_centered:
            covariance = np.dot(X.T, X) / X.shape[0]
        else:
            covariance = np.cov(X.T, bias=1)

    if X.ndim == 3:
        covariance = 0
        for i in range(X.shape[1]):
            X_temp = X[:,i,:]
            if assume_centered:
                covariance += np.dot(X_temp.T, X_temp) / X_temp.shape[0]
            else:
                covariance += np.cov(X_temp.T, bias=1)
        covariance /= X.shape[1]

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance

def prox_laplacian(a, lamda):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return a / (1 + 2. * lamda)

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order='K')
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn('Array type is integer, np.dot may overflow. '
                      'Data should be float type to avoid this issue',
                      UserWarning)
    return np.dot(x, x)


def init_precision(emp_cov, mode='empirical'):
    if isinstance(mode, np.ndarray):
        return mode.copy()

    if mode == 'empirical':
        n_times, _, n_features = emp_cov.shape
        covariance_ = emp_cov.copy()
        covariance_ *= 0.95
        # covariance_ *= 1
        K = np.empty_like(emp_cov)
        for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
            c.flat[::n_features + 1] = e.flat[::n_features + 1]
            K[i] = np.linalg.pinv(c, hermitian=True)
    elif mode == 'zeros':
        K = np.zeros_like(emp_cov)

    return K


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

convergence = namedtuple_with_defaults(
    'convergence', 'obj rnorm snorm e_pri e_dual precision')

def fast_logdet(A):
    """Compute log(det(A)) for A symmetric.

    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.

    Parameters
    ----------
    A : array-like
        The matrix.
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld
def logl(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)
def loss(S, K, n_samples=None):
    """Loss function for time-varying graphical lasso."""
    if n_samples is None:
        n_samples = np.ones(S.shape[0])
    return sum(
        -ni * logl(emp_cov, precision)
        for emp_cov, precision, ni in zip(S, K, n_samples))
def l1_norm(precision):
    """L1 norm."""
    return np.abs(precision).sum()
def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return l1_norm(precision) - np.abs(np.diag(precision)).sum()

def objective(n_samples, S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)

    if isinstance(alpha, np.ndarray):
        obj += sum(l1_od_norm(a * z) for a, z in zip(alpha, Z_0))
    else:
        obj += alpha * sum(map(l1_od_norm, Z_0))

    if isinstance(beta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(beta, map(psi, Z_2 - Z_1)))
    else:
        obj += beta * sum(map(psi, Z_2 - Z_1))

    return obj


def prox_logdet(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = (-es + np.sqrt(np.square(es) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))

def soft_thresholding(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)

def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """
    Parameters
    ----------
    rho : float
    """
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho

def time_graphical_lasso(
        emp_cov, alpha=0.01, rho=1, beta=1, max_iter=100, n_samples=None,
        verbose=False, psi='laplacian', tol=1e-4, rtol=1e-4,
        return_history=False, return_n_iter=True, mode='admm',
        compute_objective=True, stop_at=None, stop_when=1e-4,
        update_rho_options=None, init='empirical', init_inv_cov=None):

    # psi, prox_psi, psi_node_penalty = check_norm_prox(psi)
    psi=squared_norm
    prox_psi=prox_laplacian
    psi_node_penalty=False

    Z_0 = init_precision(emp_cov, mode=init)
    if isinstance(init_inv_cov, np.ndarray):
        Z_0[0,:,:]=init_inv_cov

    Z_1 = Z_0.copy()[:-1]
    Z_2 = Z_0.copy()[1:]

    U_0 = np.zeros_like(Z_0)
    U_1 = np.zeros_like(Z_1)
    U_2 = np.zeros_like(Z_2)

    Z_0_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    if n_samples is None:
        n_samples = np.ones(emp_cov.shape[0])

    checks = [
        convergence(
            obj=objective(
                n_samples, emp_cov, Z_0, Z_0, Z_1, Z_2, alpha, beta, psi))
    ]
    for iteration_ in range(max_iter):
        # update K
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        A += A.transpose(0, 2, 1)
        A /= 2.

        A *= -rho * divisor[:, None, None] / n_samples[:, None, None]
        A += emp_cov

        K = np.array(
            [
                prox_logdet(a, lamda=ni / (rho * div))
                for a, div, ni in zip(A, divisor, n_samples)
            ])

        if isinstance(init_inv_cov, np.ndarray):
            K[0,:,:]=init_inv_cov

        # update Z_0
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # other Zs
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(
                np.concatenate((A_1, A_2), axis=1), lamda=.5 * beta / rho,
                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        if isinstance(init_inv_cov, np.ndarray):
            Z_0[0,:,:]=init_inv_cov
            Z_1[0,:,:]=init_inv_cov

        # update residuals
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(K - Z_0) + squared_norm(K[:-1] - Z_1) +
            squared_norm(K[1:] - Z_2))

        snorm = rho * np.sqrt(
            squared_norm(Z_0 - Z_0_old) + squared_norm(Z_1 - Z_1_old) +
            squared_norm(Z_2 - Z_2_old))

        obj = objective(
            n_samples, emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi) \
            if compute_objective else np.nan

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
                np.sqrt(
                    squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)),
                np.sqrt(
                    squared_norm(K) + squared_norm(K[:-1]) +
                    squared_norm(K[1:]))),
            e_dual=np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho *
            np.sqrt(squared_norm(U_0) + squared_norm(U_1) + squared_norm(U_2)),
            # precision=Z_0.copy()
        )
        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        checks.append(check)
        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(
            rho, rnorm, snorm, iteration=iteration_,
            **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U_0 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new


    covariance_ = np.array([np.linalg.pinv(x, hermitian=True) for x in Z_0])
    return_list = [Z_0, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    return return_list


class TVGL():
    def __init__( self, alpha=0.01, beta=1., mode='admm', rho=1., tol=1e-4, rtol=1e-4, psi='laplacian', max_iter=100, verbose=False, assume_centered=False, return_history=False, update_rho_options=None, compute_objective=True, stop_at=None, stop_when=1e-4, suppress_warn_list=False, init='empirical'):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.rho = rho
        self.tol = tol
        self.rtol = rtol
        self.psi = psi
        self.max_iter = max_iter
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.return_history = return_history
        self.update_rho_options = update_rho_options
        self.compute_objective = compute_objective
        self.stop_at = stop_at
        self.stop_when = stop_when
        self.suppress_warn_list = suppress_warn_list
        self.init = init

    def _fit(self, emp_cov, n_samples, init_inv_cov=None):
        """Fit the TimeGraphicalLasso model to X.

        # Parameters
        # ----------
        # emp_cov : ndarray, shape (n_time, n_features, n_features)
        #     Empirical covariance of data.

        """

        out = time_graphical_lasso( emp_cov, alpha=self.alpha, rho=self.rho, beta=self.beta, mode=self.mode, n_samples=n_samples, tol=self.tol, rtol=self.rtol, psi=self.psi, max_iter=self.max_iter, verbose=self.verbose, return_n_iter=True, return_history=self.return_history, update_rho_options=self.update_rho_options, compute_objective=self.compute_objective, stop_at=self.stop_at, stop_when=self.stop_when, init=self.init, init_inv_cov=init_inv_cov)

        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.n_iter_ = out
        return self

    def fit(self, X, y):
        """Fit the TimeGraphicalLasso model to X.

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions, n_users)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each sample.

        """

        self.classes_, n_samples = np.unique(y, return_counts=True)



        if X.ndim==3:
            # n_samples*=X.shape[1]
            pass

        emp_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl], assume_centered=self.assume_centered)
                for cl in self.classes_
            ])

        a = self._fit(emp_cov, n_samples)
        return a

    def fit_stream(self, X, y, init_inv_cov):
        self.classes_, n_samples = np.unique(y, return_counts=True)

        emp_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl], assume_centered=self.assume_centered)
                for cl in self.classes_
            ])

        emp_cov=np.insert(emp_cov,0,emp_cov[0,:,:],axis=0)
        n_samples=np.insert(n_samples,0,n_samples[0],axis=0)

        a = self._fit(emp_cov, n_samples, init_inv_cov)
        return a
