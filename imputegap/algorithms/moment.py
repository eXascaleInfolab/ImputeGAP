import time
from imputegap.wrapper.AlgoPython.Moment.recovMoment import recovery_moment

def moment(incomp_data, model="training-shot", model_size="high", use_mean=False, seq_len=-1, sliding_windows=1, seed=26, logs=True, verbose=True):
    """
    Perform imputation using the SAITS (Self-Attention-based Imputation for Time Series Imputation) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    model: str, optional
        Type of the model (default: 'training-shot') / "zero-shot", "train" .

    model_size: str, optional
        Size of the LLM model (default: large) / "small", "medium", "large".

    use_mean: bool, optional
        Replace NaNs with mean values (default: False).

    seq_len : int, optional
        Length of the input sequence for temporal modeling (default: -1).

    sliding_windows: int, optional
            Stride between consecutive training windows (default is 1). If set to -1, the window size is equal to seq_len.
            Use values ≥ 1 for univariate datasets (window strategy) and -1 for multivariate datasets (sample strategy).

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
        >>> recov_data = moment(incomp_data)
        >>> print(recov_data)

    References
    ----------
    MOMENT: A Family of Open Time-series Foundation Models
    Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski Proceedings of the 41st International Conference on Machine Learning, PMLR 235:16115-16152, 2024.
    https://proceedings.mlr.press/v235/goswami24a.html
    """
    start_time = time.time()  # Record start time

    recov_data, _ = recovery_moment(incomp_data=incomp_data, model=model, model_size=model_size, use_mean=use_mean, seq_len=seq_len, sliding_windows=sliding_windows, seed=seed, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation moment - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
