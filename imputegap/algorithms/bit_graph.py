import time

from imputegap.wrapper.AlgoPython.BitGraph.recovBitGRAPH import recovBitGRAPH


def bit_graph(incomp_data, seq_len=24, sliding_windows=1, kernel_size=2, kernel_set=[1], epochs=50, batch_size=32, subgraph_size=5, num_workers=0, tr_ratio=0.7, seed=42, logs=True, verbose=True):
    """
    Perform imputation using Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_len : int, optional
        Length of the input sequence for temporal modeling (default: 1).

    sliding_windows: int, optional
        Stride between consecutive training windows (default is 1). If set to 0, the window size is equal to seq_len.
        Use values ≥ 1 for univariate datasets (window strategy) and 0 for multivariate datasets (sample strategy).

    kernel_size : int, optional
        Size of the kernel used during training (most be smaller the seq_len). Default is 2.

    kernel_set : list, optional
        Set of kernel sizes used in the model for graph convolution operations (default: [1]).

    epochs : int, optional
        Number of training epochs (default: 10).

    batch_size : int, optional
        Size of each batch (default: 32).

    subgraph_size : int, optional
        The size of each subgraph used in message passing within the graph network (default: 5).

    num_workers: int, optional
         Number of worker for multiprocess (default is 0).

    tr_ratio: float, optional
        Split ratio between training and testing sets (default is 0.9).

    seed : int, optional
        Random seed for reproducibility (default: 42).

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
        >>> recov_data = bit_graph(incomp_data)
        >>> print(recov_data)

    References
    ----------
    X. Chen1, X. Li, T. Wu, B. Liu and Z. Li, BIASED TEMPORAL CONVOLUTION GRAPH NETWORK FOR TIME SERIES FORECASTING WITH MISSING VALUES
    https://github.com/chenxiaodanhit/BiTGraph
    """
    start_time = time.time()  # Record start time

    #recov_data = recoveryBitGRAPH(input=incomp_data, node_number=node_number, kernel_set=kernel_set, dropout=dropout, subgraph_size=subgraph_size, node_dim=node_dim, seq_len=seq_len, lr=lr, batch_size=batch_size, tr_ratio=tr_ratio, epoch=epoch, num_workers=num_workers, seed=seed, verbose=verbose)
    recov_data = recovBitGRAPH(ts_m=incomp_data, seq_len=seq_len, sliding_windows=sliding_windows, kernel_size=kernel_size, kernel_set=kernel_set, epochs=epochs, batch_size=batch_size, subgraph_size=subgraph_size, tr_ratio=tr_ratio, num_workers=num_workers, seed=seed, verbose=verbose)
    #recov_data = run(ts_m=incomp_data, seq_len=8, pred_len=1, kernel_size=2, epochs=100)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation bit graph - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
