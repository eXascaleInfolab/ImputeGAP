import numpy as np


def zero_impute(incomp_data, params=None):
    """
    Impute missing values (NaNs) with zeros in the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series matrix with missing values represented as NaNs.
    params : dict, optional
        Optional parameters for the algorithm. This is not used in the current implementation but can be passed for future extensions (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where all NaN values have been replaced by zeros.

    Notes
    -----
    This simple imputation strategy replaces all missing values (NaNs) with zeros. This can be useful for initializing datasets where more complex imputation methods will follow.

    Example
    -------
        >>> incomp_data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
        >>> recov_data = zero_impute(incomp_data)
        >>> print(recov_data)
        array([[1., 2., 0.],
               [4., 0., 6.]])

    """
    recov_data = np.nan_to_num(incomp_data, nan=0)

    return recov_data
