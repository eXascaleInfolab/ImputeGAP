import numpy as np

from imputegap.tools import utils # access into imputegap utilities

# create a directory for your algorithm and a recov python file with its logic
from imputegap.wrapper.AlgoPython.MeanImpute.recovMeanImpute import recovMeanImpute

def mean_impute(incomp_data, logic="test", logs=None, use_cuda=0, verbose=True):
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

    use_cuda: int, optional, default 0
        CUDA usage mode: - 0: Force CPU. - 1: Use CUDA if available, otherwise fall back to CPU. - 2: Automatically detect CUDA availability.

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
    use_cuda = utils.check_cuda(use_cuda=use_cuda, verbose=verbose)  # optional, handle cuda inside the library

    # information for the user (prints and console) ====================================================================
    if verbose:
        print(f"\n(IMPUTATION) MeanImpute\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tlogic: {logic}\n\tlogs: {logs}\n\tverbose: {verbose}\n\t{use_cuda}:use_cuda\n")


    # if you have multiple files, and you want to store them into ./wrapper ============================================
    if logic is not None:
        recov_data = recovMeanImpute(incomp_data, logic=logic)  # adapt the import in the top of the file, look line 4==


    # if you want to put your algorithm code into a single file ========================================================
    else:
        # core of the algorithm
        mean_value = np.nanmean(incomp_data)

        # Imputation
        recov_data = np.nan_to_num(incomp_data, nan=mean_value)

    return recov_data # the return "recov_data" must be in the same format and shape as "incomp_data"
