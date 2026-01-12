import time

from imputegap.wrapper.AlgoPython.HkmfT_d.recovHKMFT import recovHKMFT

def hkmf_t(incomp_data, tags=None, seq_len=24, blackouts_begin=None, blackouts_end=None, epochs=30, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    tags : numpy.ndarray, optional
        An array containing tags that provide additional structure or metadata about
        the input data. If None, no tags are used (default is None).

    seq_length : int, optional
        Length of the input sequence used by the model. Defines the number of time steps processed at once (default 24).

    blackouts_begin : int, optional
        position of the blackout for the validation (default is None).
        if None, based on the training ratio

    blackouts_end : int, optional
        position of the blackout for the validation (default is None).
        if None, based on the training ratio

    epochs : int, optional
        The maximum number of training epochs for the Hankel Matrix Factorization algorithm.
        If convergence is reached earlier, the process stops (default is 30).

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
        >>> recov_data = hkmf_t(incomp_data, tags=None, blackouts_begin=None, blackouts_end=None, epochs=10)
        >>> print(recov_data)

    References
    ----------
    L. Wang, S. Wu, T. Wu, X. Tao and J. Lu, "HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization," in IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 11, pp. 3582-3593, 1 Nov. 2021, doi: 10.1109/TKDE.2020.2971190. keywords: {Time series analysis;Matrix decomposition;Market research;Meteorology;Sparse matrices;Indexes;Software;Tagged time series;missing value imputation;blackouts;hankel matrix factorization}
    https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
    """
    start_time = time.time()  # Record start time

    recov_data = recovHKMFT(dataset=incomp_data, tags=tags, seq_len=seq_len, blackouts_begin=blackouts_begin, blackouts_end=blackouts_end, max_epoch=epochs, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation hkmf_t - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
