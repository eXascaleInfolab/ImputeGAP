import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils

def native_dynammo(__py_matrix, __py_h, __py_maxIter, __py_fast, __verbose=True):
    """
    Perform matrix imputation using the DynaMMo algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        A 2D input matrix (time-series data) containing missing values (NaNs).
    __py_h : int
        The time window (H) parameter for modeling temporal dynamics.
    __py_maxIter : int
        The maximum number of iterations for the imputation process.
    __py_fast : bool
        If True, enables faster approximate processing.
    __verbose : bool, optional
        Whether to display the contamination information (default is False).

     Returns
    -------
    numpy.ndarray
        A completed matrix with missing values imputed using the DynaMMo algorithm.


    References
    ----------
    L. Li, J. McCann, N. S. Pollard, and C. Faloutsos. Dynammo: mining and summarization of coevolving sequences with missing values. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Paris, France, June 28 - July 1, 2009, pages 507–516, 2009.
    """

    shared_lib = utils.load_share_lib("lib_dynammo", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_h >= 0);
    assert (__py_h < __py_m);
    assert (__py_maxIter > 0);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __py_h = __native_c_types_import.c_ulonglong(__py_h);
    __py_maxIter = __native_c_types_import.c_ulonglong(__py_maxIter);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoveryDynaMMo(__ctype_matrix, __ctype_size_n, __ctype_size_m, __py_h, __py_maxIter, __py_fast);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def dynammo(incomp_data, h, max_iteration, approximation, logs=True, verbose=True, lib_path=None):
    """
    DynaMMo algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    h : int
        The time window (H) parameter for modeling temporal dynamics.
    max_iteration : int
        The maximum number of iterations for the imputation process.
    approximation : bool
        If True, enables faster approximate processing.
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).

    Returns
    -------
    numpy.ndarray
        A completed matrix with missing values imputed using the DynaMMo algorithm.

    Example
    -------
        >>> recov_data = dynammo(incomp_data=incomp_data, h=5, max_iteration=100, approximation=True, logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    if logs and verbose:
        print(f"\n> logs: imputation DynaMMo - Data requires transposition to be effective. New shape: {incomp_data.T.shape}\n")

    # Call the C++ function to perform recovery
    recov_data = native_dynammo(incomp_data.T, h, max_iteration, approximation, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation DynaMMo - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data.T
