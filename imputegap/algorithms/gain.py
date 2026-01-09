import time
from imputegap.wrapper.AlgoPython.GAIN.recovGAIN import recovGAIN


def gain(incomp_data, batch_size=-1, epochs=100, alpha=10, hint_rate=0.9, tr_ratio=0.9, logs=True, verbose=True):
    """
    Perform imputation using the Multivariate Recurrent Neural Network (MRNN) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    batch_size : int, optional
        Number of samples in each mini-batch during training. Default is 32.
    hint_rate : float, optional
        Probability of providing hints for the missing data during training. Default is 0.9.
    alpha : float, optional
        Hyperparameter that controls the balance between the adversarial loss and the reconstruction loss. Default is 10.
    epoch : int, optional
        Number of training epochs. Default is 100.
    tr_ratio: float, optional
        Split ratio between training and testing sets (default is 0.9).
    logs : bool, optional
        Whether to log execution details (e.g., training progress and execution time). Default is True.
    verbose : bool, optional
        Whether to display the contamination information (default is True).


    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.


    Example
    -------
        >>> recov_data = gain(incomp_data, batch_size=32, hint_rate=0.9, alpha=10, epoch=100)
        >>> print(recov_data)

    References
    ----------
    J. Yoon, W. R. Zame and M. van der Schaar, "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks," in IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1477-1490, May 2019, doi: 10.1109/TBME.2018.2874712. keywords: {Time measurement;Interpolation;Estimation;Medical diagnostic imaging;Correlation;Recurrent neural networks;Biomedical measurement;Missing data;temporal data streams;imputation;recurrent neural nets}
    """
    start_time = time.time()  # Record start time

    recov_data = recovGAIN(ts_m=incomp_data, batch_size=batch_size, iterations=epochs, alpha=alpha, hint_rate=hint_rate, tr_ratio=tr_ratio, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation gain - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
