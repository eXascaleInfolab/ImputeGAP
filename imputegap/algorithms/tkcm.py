import time
import ctypes as __native_c_types_import;

import numpy as np

from imputegap.tools import utils

def native_tkcm(__py_matrix, __py_rank, __verbose=True):
    """
    Perform matrix imputation using the TKCM algorithm with native C++ support.

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
    K. Wellenzohn, M. H. Böhlen, A. Dignös, J. Gamper, and H. Mitterer. Continuous imputation of missing values in streams of pattern-determining time series. In Proceedings of the 20th International Conference on Extending Database Technology, EDBT 2017, Venice, Italy, March 21-24, 2017., pages 330–341, 2017.
    """

    shared_lib = utils.load_share_lib("lib_tkcm", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoveryTKCM(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def tkcm(incomp_data, rank, logs=True, verbose=True, lib_path=None):
    """
    TKM algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    rank : int
        The rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
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
        >>> recov_data = tkcm(incomp_data=incomp_data, rank=5, logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if utils.check_contamination_series(incomp_data):
        print(f"(IMPUTATION-ERROR) tkcm is a uni-dimensional algorithm; it requires that only series 0 be contaminated.\n")
        return incomp_data

    recov_data = native_tkcm(ts_m, rank, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation TKCM - Execution Time: {(end_time - start_time):.4f} seconds\n")

    recov[m_mask] = recov_data[m_mask]

    return recov
