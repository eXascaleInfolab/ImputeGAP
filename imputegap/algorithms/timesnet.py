import time

from imputegap.wrapper.AlgoPython.TimesNet.recovTimesNet import recovTimesNet


def timesnet(incomp_data, seq_len=-1, batch_size=-1, epochs=10, gpt_layers=6, num_workers=0, tr_ratio=0.9, seed=42, logs=True, verbose=True):
    """
    Perform imputation using TimesNet algorithm.


    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_len : int, optional
        Length of the input sequence for the encoder. If -1, it will be automatically determined (default: -1).

    batch_size : int, optional
        Number of samples per batch during training/inference. If -1, it will be auto-set (default: -1).

    epochs : int, optional
        Number of epoch for training the model (default: 10).

    gpt_layers : int, optional
        Number of layers in the transformer/generator component (default: 6).

    num_workers: int, optional
         Number of worker for multiprocess (default is 0).

    tr_ratio: float, optional
         Split ratio between training and testing sets (default is 0.9).

    seed : int, optional
        Random seed for reproducibility (default: 2021).

    logs : bool, optional
        Whether to print/log execution time and key events (default: True).

    verbose : bool, optional
        Whether to print detailed output information during execution (default: True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values filled in.

    Example
    -------
        >>> imputed = timesnet(incomp_data, seq_len=96, batch_size=16)
        >>> print(imputed.shape)

    References
    ----------
    Cheng, Jinguo and Yang, Chunwei and Cai, Wanlin and Liang, Yuxuan and Wen, Qingsong and Wu, Yuankai: "NuwaTS: Mending Every Incomplete Time Series", arXiv'2024
    https://github.com/Chengyui/NuwaTS/tree/master
    """
    start_time = time.time()  # Record start time

    recov_data = recovTimesNet(ts_m=incomp_data, seq_len=seq_len, batch_size=batch_size, epochs=epochs, gpt_layers=gpt_layers, model="TimesNet", num_workers=num_workers, tr_ratio=tr_ratio, seed=seed, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation timesnet - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
