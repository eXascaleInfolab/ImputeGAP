import numpy as np

from imputegap.wrapper.AlgoPython.MeanImpute.recovMeanImpute import recovMeanImpute

def mean_impute(incomp_data, logic="test", logs=None, verbose=True):
    """
    Impute NaN values with the mean value of the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).

    logic = str, optional
        define the logic of the mean impute

    logs : bool, optional
        Whether to print/log execution time and key events (default: True).

    verbose : bool, optional
        Whether to print detailed output information during execution (default: True).


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

    # information for the user =========================================================================================
    if verbose:
        print(f"\n(IMPUTATION) MeanImpute\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tlogic: {logic}\n\tlogs: {logs}\n\tverbose: {verbose}\n")


    # if you have multiple files, and you want to store them into ./wrapper ============================================
    if logic is not None:
        recov_data = recovMeanImpute(incomp_data, logic=logic)


    # if you want to put your algorithm intoput into a single file =====================================================
    else:
        # core of the algorithm
        mean_value = np.nanmean(incomp_data)

        # Imputation
        recov_data = np.nan_to_num(incomp_data, nan=mean_value)

    return recov_data
