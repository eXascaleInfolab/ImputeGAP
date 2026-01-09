# ===============================================================================================================
# SOURCE: https://github.com/SemenovAlex/trmf/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
# ===============================================================================================================

import numpy as np

from imputegap.wrapper.AlgoPython.TRMF.trmf import trmf


def recovTRMF(data, lags=[1,7], K=4, lambda_f=1.0, lambda_x=1.0, lambda_w=1.0, eta=1.0, alpha=1000.0, max_iter=5000, reversed=False, verbose=True):
    """Temporal Regularized Matrix Factorization : https://github.com/SemenovAlex/trmf

    Parameters
    ----------
    data : numpy.ndarray
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

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.
    """

    if reversed:
        data = data.T

    ts_m = np.copy(data)
    recov = np.copy(data)
    m_mask = np.isnan(data)

    if verbose:
        print(f"(IMPUTATION) TRMF\n\tMatrix: {data.shape[0]}, {data.shape[1]}\n\tlags: {lags}\n\tK: {K}\n\tlambda_f: {lambda_f}\n\tlambda_x: {lambda_x}\n\tlambda_w: {lambda_w}\n\teta: {eta}\n\talpha: {alpha}\n\tmax_iter: {max_iter}\n")
        print("\ncall: trmf.impute(params={'lags':", lags,", 'K':", K,", 'lambda_f': ", lambda_f,", 'lambda_x':", lambda_x,", 'lambda_w':", lambda_w,", 'eta': ", eta, ", 'alpha':", alpha,", 'max_iter':", max_iter,"})\n")


    model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter)
    model.fit(ts_m)
    data_imputed = model.impute_missings()

    if reversed:
        data_imputed = np.array(data_imputed).T
    else:
        data_imputed = np.array(data_imputed)

    if verbose:
        print(f"{data_imputed.shape = }\n")

    recov[m_mask] = data_imputed[m_mask]

    return recov