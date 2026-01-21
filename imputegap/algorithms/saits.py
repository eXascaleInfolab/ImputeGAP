import time

#from imputegap.wrapper.AlgoPython.SAITS.run_models import recov_saits
from imputegap.wrapper.AlgoPython.SAITS.recovSAITS import recov_saits

def saits(incomp_data, seq_len, epochs, batch_size, sliding_windows, n_head, num_workers, tr_ratio, seed=26, logs=True, verbose=True):
    """
    Perform imputation using the SAITS (Self-Attention-based Imputation for Time Series Imputation) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_len : int, optional
        Length of the input sequence for temporal modeling (default: 1).

    epochs : int, optional
        Number of training epochs (default: 10).

    batch_size : int, optional, default=32
        The number of samples per training batch.

    sliding_windows: int, optional
            Stride between consecutive training windows (default is 1). If set to -1, the window size is equal to seq_len.
            Use values ≥ 1 for univariate datasets (window strategy) and -1 for multivariate datasets (sample strategy).

    n_head : int, optional
        head num of self-attention (default: 4)

    num_workers: int, optional
        Number of worker for multiprocess (default is 0).

    seed : int, optional
        Random seed for reproducibility (default: 42).

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
        >>> recov_data = saits(incomp_data)
        >>> print(recov_data)

    References
    ----------
    Wenjie Du, David Coté, Yan Liu. SAITS: Self-attention-based imputation for time series. arXiv, 2023
    https://github.com/WenjieDu/SAITS
    """
    start_time = time.time()  # Record start time

    recov_data = recov_saits(incomp_data=incomp_data, seq_len=seq_len, epochs=epochs, batch_size=batch_size, n_head=n_head, sliding_windows=sliding_windows, num_workers=num_workers, seed=seed, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation saits - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
