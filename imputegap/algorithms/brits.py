import time
#from imputegap.wrapper.AlgoPython.BRITS.runnerBRITS import brits_recovery
from imputegap.wrapper.AlgoPython.BRITS.recovBRITS import recovBRITS


def brits(incomp_data, model="rits_i", seq_len=24, epochs=10, batch_size=7, sliding_windows=1, hidden_layers=64, impute_weight=0.3, num_workers=0, seed=42, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using the BRITS algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    model : str
        Specifies the type of model to use for the imputation. Options may include predefined models like 'brits', 'brits-i'.

    seq_len : int
        Length of the input sequence used by the model. Defines the number of time steps processed at once (default 24).

    epochs : int
        Number of epochs for training the model. Determines how many times the algorithm processes the entire dataset during training.

    batch_size : int
        Size of the batches used during training. Larger batch sizes can speed up training but may require more memory.

    sliding_windows: int, optional
        Stride between consecutive training windows (default is 1). If set to 0, the window size is equal to seq_len.
        Use values ≥ 1 for univariate datasets (window strategy) and 0 for multivariate datasets (sample strategy).

    hidden_layers : int
        Number of units in the hidden layer of the model. Controls the capacity of the neural network to learn complex patterns.

    impute_weight : float
        Weight of the imputation term (default: 0.3).

    num_workers: int, optional
         Number of worker for multiprocess (default is 0).

    tr_ratio: float, optional
         Split ratio between training and testing sets (default is 0.9).

    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The BRITS algorithm is a machine learning-based approach for time series imputation, where missing values are recovered using a recurrent neural network structure.

    This function logs the total execution time if `logs` is set to True.

    Example
    -------
        >>> recov_data = brits(incomp_data=incomp_data, model="brits")
        >>> print(recov_data)

    References
    ----------
    Cao, W., Wang, D., Li, J., Zhou, H., Li, L. & Li, Y. BRITS: Bidirectional Recurrent Imputation for Time Series. Advances in Neural Information Processing Systems, 31 (2018). https://proceedings.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf
    """
    start_time = time.time()  # Record start time

    #recov_data = brits_recovery(incomp_data=incomp_data, model=model, epoch=epoch, batch_size=batch_size, nbr_features=nbr_features, hidden_layers=hidden_layers, seq_length=seq_length, tr_ratio=tr_ratio, num_workers=num_workers, seed=42, verbose=verbose)
    recov_data = recovBRITS(incomp_data=incomp_data, seq_len=seq_len, model_name=model, epochs=epochs, batch_size=batch_size, sliding_windows=sliding_windows, impute_weight=impute_weight, hid_size=hidden_layers, num_workers=num_workers, seed=seed, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation brits - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
