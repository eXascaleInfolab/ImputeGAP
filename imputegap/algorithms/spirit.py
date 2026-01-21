import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils

def native_spirit(__py_matrix, __py_k, __py_w, __py_lambda, __verbose=True):
    """
    Perform matrix imputation using the SPIRIT algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_k : int
        The number of eigencomponents (principal components) to retain for dimensionality reduction.
        Example: 2, 5, 10.
    __py_w : int
        The window size for capturing temporal dependencies.
        Example: 5 (short-term), 20 (long-term).
    __py_lambda : float
        The forgetting factor controlling how quickly past data is "forgotten".
        Example: 0.8 (fast adaptation), 0.95 (stable systems).
    __verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values filled.

    References
    ----------
    S. Papadimitriou, J. Sun, and C. Faloutsos. Streaming pattern discovery in multiple time-series. In Proceedings of the 31st International Conference on Very Large Data Bases, Trondheim, Norway, August 30 - September 2, 2005, pages 697–708, 2005.
    """

    shared_lib = utils.load_share_lib("lib_spirit", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_k > 0);
    assert (__py_w > 0);
    assert (__py_lambda >= 0);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_k = __native_c_types_import.c_ulonglong(__py_k);
    __ctype_w = __native_c_types_import.c_ulonglong(__py_w);
    __ctype_lambda = __native_c_types_import.c_double(__py_lambda);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoverySPIRIT(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_k, __ctype_w, __ctype_lambda);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def spirit(incomp_data, k, w, lambda_value, logs=True, verbose=True, lib_path=None):
    """
    SPIRIT algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    k : int
        The number of eigencomponents (principal components) to retain for dimensionality reduction.
        Example: 2, 5, 10.
    w : int
        The window size for capturing temporal dependencies.
        Example: 5 (short-term), 20 (long-term).
    lambda_value : float
        The forgetting factor controlling how quickly past data is "forgotten".
        Example: 0.8 (fast adaptation), 0.95 (stable systems).
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
        >>> recov_data = spirit(incomp_data=incomp_data, k=2, w=5, lambda_value=0.8, logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    # for loop for uni-dimentional data ===start========================================================================
    #nan_counts_per_col = np.sum(np.isnan(incomp_data), axis=0)
    #cols_with_nans = np.where(nan_counts_per_col > 0)[0].shape[0]
    #series_imputations = []
    #for series in range(0, incomp_data.shape[1]):
    #    series_to_impute = incomp_data[:, series]
    #    has_nans = np.isnan(series_to_impute).any()
    #    series_to_impute = series_to_impute.reshape(-1, 1)
    #    if has_nans or cols_with_nans < 1:
    #    imputed_matrix = native_spirit(series_to_impute, k, w, lambda_value, verbose)
    #    imputed_matrix = np.asarray(imputed_matrix).reshape(-1)
    #    series_imputations.append(imputed_matrix)
    #recov_data = np.column_stack(series_imputations)
    # for loop for uni-dimentional data ===end==========================================================================

    if utils.check_contamination_series(incomp_data, algo="spirit", verbose=verbose):
        return incomp_data

    recov_data = native_spirit(incomp_data, k, w, lambda_value, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation SPIRIT - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
