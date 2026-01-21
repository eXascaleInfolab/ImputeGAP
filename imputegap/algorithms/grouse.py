import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils

def native_grouse(__py_matrix, __py_rank, __verbose=True):
    """
    Perform matrix imputation using the GROUSE algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_rank : int
        The truncation rank for matrix decomposition (must be greater than 0 and less than the number of columns).
    __verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    References
    ----------
    D. Zhang and L. Balzano. Global convergence of a grassmannian gradient descent algorithm for subspace estimation. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, AISTATS 2016, Cadiz, Spain, May 9-11, 2016, pages 1460–1468, 2016.
    """

    shared_lib = utils.load_share_lib("lib_grouse", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoveryGROUSE(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def grouse(incomp_data, max_rank, logs=True, verbose=True, lib_path=None):
    """
    GROUSE algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    max_rank : int
        The max rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
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
        >>> recov_data = grouse(incomp_data=incomp_data, max_rank=5, logs=True)
        >>> print(recov_data)
    """
    start_time = time.time()  # Record start time

    if logs and verbose:
        print(f"\n> logs: imputation GROUSE - Data requires transposition to be effective. New shape: {incomp_data.T.shape}\n")

    # Call the C++ function to perform recovery
    recov_data = native_grouse(incomp_data.T, max_rank, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation GROUSE - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data.T
