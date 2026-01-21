import time

from imputegap.wrapper.AlgoPython.MPIN.recovMPIN import recovMPIN


def mpin(incomp_data=None, window=6, incre_mode="alone", base="SAGE", epochs=15, num_of_iteration=5, k=10, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using the Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    window : int, optional
        The size of the sliding window for processing data streams (default is 6).
    incre_mode : str, optional
        The mode of incremental learning. Options are: 'alone',  'data', 'state', 'state+transfer', 'data+state', 'data+state+transfer' (default is "alone").
    base : str, optional
        The base model used for graph representation and message propagation. Common options include "SAGE" and "GCN" (default is "SAGE").
    epochs : int, optional
        The number of epochs to run the training process (default is 200).
    num_of_iteration : int, optional
        The number of iteration of the whole training (default is 5).
    k : int, optional
        The number of neighbors to consider during message propagation (default is 10).
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
        >>> recov_data = mpin(incomp_data,  window=6, incre_mode="alone", base="SAGE", epochs=15, num_of_iteration=5, k=10)
        >>> print(recov_data)

    References
    ----------
    Li, X., Li, H., Lu, H., Jensen, C.S., Pandey, V. & Markl, V. Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). arXiv (2023). https://arxiv.org/abs/2311.07344
    https://github.com/XLI-2020/MPIN
    """
    start_time = time.time()  # Record start time

    recov_data = recovMPIN(incomp_data=incomp_data, window=window, incre_mode=incre_mode, base=base, epochs=epochs, num_of_iter=num_of_iteration, k=k, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation mpin - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
