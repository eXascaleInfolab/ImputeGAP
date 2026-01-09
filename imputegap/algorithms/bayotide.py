import time

#from imputegap.wrapper.AlgoPython.BayOTIDE.BayOTIDE import recovBayOTIDE
from imputegap.wrapper.AlgoPython.BayOTIDE.recovBayOTIDE import recovBayOTIDE


def bay_otide(incomp_data, K_trend=20, K_season=2, n_season=5, K_bias=1, time_scale=1, a0=0.6, b0=2.5, v=0.5, num_fold=1, tr_ratio=0.8, logs=True, verbose=False):
    """
    BayOTIDE class to impute missing values using Bayesian Online Multivariate Time series Imputation with functional decomposition

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    K_trend : int, (optional) (default: 20)
        Number of trend factors.

    K_season : int, (optional) (default: 2)
        Number of seasonal factors.

    n_season : int, (optional) (default: 5)
        Number of seasonal components per factor.

    K_bias : int, (optional) (default: 1)
        Number of bias factors.

    time_scale : float, (optional) (default: 1)
        Time scaling factor.

    a0 : float, (optional) (default: 0.6)
        Hyperparameter for prior distribution.

    b0 : float, (optional) (default: 2.5)
        Hyperparameter for prior distribution.

    v : float, (optional) (default: 0.5)
        Variance parameter.

    num_fold: int, optional
         Number of fold used for training (default is 1).

    tr_ratio : float, (optional) (default: 0.6)
        Ratio of the training set for the model.

    config : dict, (optional) (default: None)
        Dictionary containing all configuration parameters, that will replace all other parameters (see documentation).

    args : object, (optional) (default: None)
        Arguments containing all configuration parameters, that will replace all other parameters (see documentation).

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
        >>> recov_data = bay_otide(incomp_data, K_trend=20, K_season=2, n_season=5, K_bias=1, time_scale=1, a0=0.6, b0=2.5, v=0.5, num_fold=1, tr_ratio=0.8)
        >>> print(recov_data)

    References
    ----------
    S. Fang, Q. Wen, Y. Luo, S. Zhe, and L. Sun, "BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition," CoRR, vol. abs/2308.14906, 2024. [Online]. Available: https://arxiv.org/abs/2308.14906.
    https://github.com/xuangu-fang/BayOTIDE
    """
    start_time = time.time()  # Record start time

    #recov_data = recovBayOTIDE(incomp_m=incomp_data, K_trend=K_trend, K_season=K_season, n_season=n_season, K_bias=K_bias, time_scale=time_scale, a0=a0, b0=b0, v=v, num_workers=0, tr_ratio=tr_ratio, verbose=verbose)
    recov_data = recovBayOTIDE(incomp_data=incomp_data, K_trend=K_trend, K_season=K_season, n_season=n_season, K_bias=K_bias, time_scale=time_scale, a0=a0, b0=b0, v=v, num_fold=num_fold, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation bay_otide - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
