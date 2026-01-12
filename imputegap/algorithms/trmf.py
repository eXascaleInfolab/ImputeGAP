import time

#from imputegap.wrapper.AlgoPython.trmf.tmrfRecovery import recoveryTRMF
from imputegap.wrapper.AlgoPython.Trmf.recovTRMF import recovTRMF


def trmf(incomp_data, lags, K, lambda_f, lambda_x, lambda_w, eta, alpha, max_iter, logs=True, verbose=True):
    """
    Perform imputation using the Temporal Regularized Matrix Factorization (TRMF) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    lags : array-like, optional
        Set of lag indices to use in model.

    K : int, optional
        Length of latent embedding dimension

    lambda_f : float, optional
        Regularization parameter used for matrix F.

    lambda_x : float, optional
        Regularization parameter used for matrix X.

    lambda_w : float, optional
        Regularization parameter used for matrix W.

    alpha : float, optional
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.

    eta : float, optional
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int, optional
        Number of iterations of updating matrices F, X and W.

    verbose : bool, optional
        Whether to display the contamination information (default is True).

    logs : bool, optional
        Whether to log the execution time (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The MRNN algorithm is a machine learning-based approach for time series imputation, where missing values are recovered using a recurrent neural network structure.

    This function logs the total execution time if `logs` is set to True.

    Example
    -------
        >>> recov_data = trmf(incomp_data, lags=[], K=-1, lambda_f=1.0, lambda_x=1.0, lambda_w=1.0, eta=1.0, alpha=1000.0, max_iter=100)
        >>> print(recov_data)

    References
    ----------
    H.-F. Yu, N. Rao, and I. S. Dhillon, "Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction," in *Advances in Neural Information Processing Systems*, vol. 29, 2016. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf
    """
    start_time = time.time()  # Record start time

    recov_data = recovTRMF(data=incomp_data, lags=lags, K=K, lambda_f=lambda_f, lambda_x=lambda_x, lambda_w=lambda_w, eta=eta, alpha=alpha, max_iter=max_iter, reversed=False, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation trmf - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
