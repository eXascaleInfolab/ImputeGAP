import numpy as np


def recovMeanImpute(incomp_data, logic="test"):
    """
    Impute NaN values with the mean value of the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).

    logic = str, optional
        define the logic of the mean impute

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series.

    Notes
    -----
    This function finds the non-NaN value in the time series and replaces all NaN values with this mean value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
        >>> incomp_data = np.array([[5, 2, np.nan], [3, np.nan, 6]])
        >>> recov_data = mean_impute(incomp_data)
        >>> print(recov_data)
        array([[5., 2., 4.],
               [3., 4., 6.]])

    """

    if logic == "mean":
        # core of the algorithm
        mean_value = np.nanmean(incomp_data)

        # Imputation
        return np.nan_to_num(incomp_data, nan=mean_value)

    else:
        return np.nan_to_num(incomp_data, nan=0)
