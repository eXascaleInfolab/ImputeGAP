import numpy as np
import time


def mean_impute_by_series(incomp_data, logs=True, verbose=True):
    """
    Impute NaN values with the mean value of the time series by series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series by series.

    Example
    -------
        >>> incomp_data = np.array([[5, 2, np.nan], [3, np.nan, 6]])
        >>> recov_data = mean_impute_by_series(incomp_data)
        >>> print(recov_data)
        array([[5., 2., 3.5],
               [3., 4.5, 6.]])

    """
    start_time = time.time()  # Record start time

    recov_data = np.copy(incomp_data)
    n_rows, n_cols = incomp_data.shape

    # Iterate over each row (time series)
    for current_series in range(n_cols):
        s = incomp_data[:, current_series]
        mean_value = np.nanmean(s)
        recov_data[np.isnan(s), current_series] = mean_value

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation mean impute (by series) - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
