import ctypes
import time
import ctypes as __native_c_types_import;
import numpy as __numpy_import;
import numpy as np

from imputegap.tools import utils


def native_rosl(__py_matrix, __py_rank, __py_regularization, __verbose=True):
    """
    Perform matrix imputation using the ROSL algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input 2D matrix with missing values (NaNs) to be recovered.
    __py_rank : int
        The rank of the low-dimensional subspace for matrix decomposition.
        Must be greater than 0 and less than or equal to the number of columns in the matrix.
    __py_regularization : float
        The regularization parameter to control the trade-off between reconstruction accuracy and robustness.
        Higher values enforce sparsity or robustness against noise in the data.
    __verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation, with missing values filled in and noise/outliers handled.


    References
    ----------
    X. Shu, F. Porikli, and N. Ahuja. Robust orthonormal subspace learning: Efficient recovery of corrupted low-rank matrices. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2014, Columbus, OH, USA, June 23-28, 2014, pages 3874–3881, 2014.
    """
    shared_lib = utils.load_share_lib("lib_rosl", verbose=__verbose)
    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);

    assert (__py_regularization >= 0);


    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
    __ctype_regularization = __native_c_types_import.c_double(__py_regularization);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoveryROSL(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank, __ctype_regularization);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def rosl(incomp_data, rank, regularization, logs=True, verbose=True, lib_path=None):
    """
    ROSL algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    rank : int
        The rank of the low-dimensional subspace for matrix decomposition.
        Must be greater than 0 and less than or equal to the number of columns in the matrix.
    regularization : float
        The regularization parameter to control the trade-off between reconstruction accuracy and robustness.
        Higher values enforce sparsity or robustness against noise in the data.
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).
    lib_path : str, optional
        Custom path to the shared library file (default is None).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation, with missing values filled in and noise/outliers handled.

    Example
    -------
        >>> recov_data = rosl(incomp_data=incomp_data, rank=5, regularization=10 logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_rosl(incomp_data, rank, regularization, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation ROSL - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
