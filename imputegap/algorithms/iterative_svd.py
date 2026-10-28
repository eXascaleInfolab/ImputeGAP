import ctypes as __native_c_types_import
import time

from imputegap.tools import utils


def native_iterative_svd(__py_matrix, __py_rank, __verbose=True):  # noqa: PYI063
    """
    Perform matrix imputation using the Iterative SVD algorithm with native C++ support.

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
    Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman, Missing value estimation methods for DNA microarrays , Bioinformatics, Volume 17, Issue 6, June 2001, Pages 520–525, https://doi.org/10.1093/bioinformatics/17.6.520
    """

    shared_lib = utils.load_share_lib("lib_iterative_svd", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoveryIterativeSVD(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def iterative_svd(incomp_data, rank=3, logs=True, verbose=True, lib_path=None):
    """
    Iterative SVD algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    rank : int
        The truncation rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
    logs : bool, optional
        Whether to log the execution time (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
        >>> recov_data = iterative_svd(incomp_data=incomp_data, truncation_rank=1, logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_iterative_svd(incomp_data, rank, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation iterative svd - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
