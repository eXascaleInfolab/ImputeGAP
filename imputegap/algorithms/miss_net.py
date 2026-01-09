import time

import numpy as np

from imputegap.tools import utils
from imputegap.wrapper.AlgoPython.MissNet.recovMissNet import recovMissNet

def miss_net(incomp_data, n_components=15, alpha=0.5, beta=1, n_cl=1, max_iter=100, tol=5, random_init=False, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using the Multivariate Recurrent Neural Network (MRNN) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    n_components : int
        number of latent dimentions. (default 15)

    alpha : float, optional
        Trade-off parameter controlling the contribution of contextual matrix
        and time-series. If alpha = 0, network is ignored. (default 0.5)

    beta : float, optional
        Regularization parameter for sparsity. (default 0.1)

    n_cl : int, optional
        Number of clusters. (default 1)

    max_iter : int, optional
        Maximum number of iterations for convergence. (default 20)

    tol : float, optional
        Tolerance for early stopping criteria.  (default 5)

    random_init : bool, optional
        Whether to use random initialization for latent variables. (default False)

    tr_ratio: float, optional
        Split ratio between training and testing sets (default is 0.9).

    logs : bool, optional
        Whether to log the execution time (default is True).

    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
        >>> recov_data = miss_net(incomp_data, alpha=0.5, beta=0.1, n_cl=1, max_iter=20, tol=5, random_init=False)
        >>> print(recov_data)

    References
    ----------
    Kohei Obata, Koki Kawabata, Yasuko Matsubara, and Yasushi Sakurai. 2024. Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24). Association for Computing Machinery, New York, NY, USA, 2296–2306. https://doi.org/10.1145/3637528.3671760

    """

    start_time = time.time()  # Record start time

    recov_data = recovMissNet(X=incomp_data, n_components=n_components, alpha=alpha, beta=beta, n_cl=n_cl, max_iter=max_iter, tol=tol, random_init=random_init, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation miss net - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data