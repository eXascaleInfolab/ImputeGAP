import numpy as np


def min_impute(incomp_data, params=None):
    """
    Impute NaN values with the minimum value of the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the minimum value from the time series.

    Notes
    -----
    This function finds the minimum non-NaN value in the time series and replaces all NaN values with this minimum value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
        >>> incomp_data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
        >>> recov_data = min_impute(incomp_data)
        >>> print(recov_data)
        array([[1., 2., 1.],
               [4., 1., 6.]])

    """

    # logic
    min_value = np.nanmin(incomp_data)

    # Imputation
    recov_data = np.nan_to_num(incomp_data, nan=min_value)

    return recov_data
