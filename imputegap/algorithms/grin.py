import time

#from imputegap.wrapper.AlgoPython.GRIN.recoveryGRIN import recoveryGRIN
from imputegap.wrapper.AlgoPython.GRIN.runnerGRIN import runGRIN


def grin(incomp_data, seq_len=1, sim_type="corr", epochs=50, batch_size=32, sliding_windows=1, alpha=10.0, patience=40, num_workers=0, tr_ratio=0.7, logs=True, verbose=True):
    """
    Perform imputation using the Multivariate Recurrent Neural Network (MRNN) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_len : int, optional, default=1
        Length of the input sequence used by the model. Defines the number of time steps inside each sample.

    sim_type : string, optional, default="corr"
        Select the function used to compute the similarity measure: (uniform, corr, dcrnn, stcn)

    epochs : int, optional, default=20
            The maximum number of training epochs.

    batch_size : int, optional, default=32
        The number of samples per training batch.

    sliding_windows: int, optional
        Stride between consecutive training windows (default is 1). If set to 0, the window size is equal to seq_len.
        Use values ≥ 1 for univariate datasets (window strategy) and 0 for multivariate datasets (sample strategy).

    alpha : float, optional, default=10.0
        The weight assigned to the adversarial loss term during training.

    patience : int, optional, default=4
        Number of epochs without improvement before early stopping is triggered.

    num_workers : int, optional, default=0
        The number of worker processes for data loading.

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
        >>> recov_data = grin(incomp_data)
        >>> print(recov_data)

    References
    ----------
    A. Cini, I. Marisca, and C. Alippi, "Multivariate Time Series Imputation by Graph Neural Networks," CoRR, vol. abs/2108.00298, 2021
    https://github.com/Graph-Machine-Learning-Group/grin
    """
    start_time = time.time()  # Record start time

    #recov_data = recoveryGRIN(input=incomp_data, d_hidden=d_hidden, lr=lr, batch_size=batch_size, window=window, alpha=alpha, patience=patience, epochs=epochs, workers=workers, tr_ratio=tr_ratio, verbose=verbose)
    recov_data = runGRIN(incomp_data=incomp_data, seq_len=seq_len, sim_type=sim_type, epochs=epochs, batch_size=batch_size, sliding_windows=sliding_windows, alpha=alpha, patience=patience, num_workers=num_workers, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation grin - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
