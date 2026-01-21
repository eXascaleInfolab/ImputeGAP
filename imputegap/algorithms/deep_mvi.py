import time

#from imputegap.wrapper.AlgoPython.DeepMVI.recovery import deep_mvi_recovery
from imputegap.wrapper.AlgoPython.DeepMVI.recovDeepMVI import deep_mvi_recovery


def deep_mvi(incomp_data, max_epoch=100, patience=2, lr=0.001, batch_size=16, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using the DEEP MVI (Deep Multivariate Imputation) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    max_epoch : int, optional
        Limit of training epoch (default is 1000)
    patience : int, optional
        Number of threshold error that can be crossed during the training (default is 2)
    lr : float, optional
        Learning rate of the training (default is 0.001)
    batch_size: int, optional
        Number of samples processed before the model's weights are updated (default is 16)
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
        >>> recov_data = deep_mvi(incomp_data, 1000, 2, 0.001, 16)
        >>> print(recov_data)

    References
    ----------
    P. Bansal, P. Deshpande, and S. Sarawagi. Missing value imputation on multidimensional time series. arXiv preprint arXiv:2103.01600, 2023
    https://github.com/pbansal5/DeepMVI
    """
    start_time = time.time()  # Record start time

    recov_data = deep_mvi_recovery(input=incomp_data, max_epoch=max_epoch, patience=patience, lr=lr, batch_size=batch_size, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation deep mvi - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
