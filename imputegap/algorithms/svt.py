import ctypes as __native_c_types_import
import time

from imputegap.tools import utils

def native_svt(__py_matrix, __py_tau, __verbose=True):  # noqa: PYI063
    """
    Perform matrix imputation using the SVT algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).

    __py_tau : float
        The thresholding parameter for singular values. Controls how singular values are shrunk during the decomposition process.
        Larger values encourage a sparser, lower-rank solution, while smaller values retain more detail.

    __verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    References
    ----------
    J. Cai, E. J. Candès, and Z. Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on Optimization, 20(4):1956–1982, 2010. [8] J. Cambronero, J. K. Feser, M. J. Smith, and
    """

    shared_lib = utils.load_share_lib("lib_svt", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __py_tau = __native_c_types_import.c_double(__py_tau);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.recoverySVT(__ctype_matrix, __ctype_size_n, __ctype_size_m, __py_tau);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def svt(incomp_data, tau=0.2, logs=True, verbose=True, lib_path=None):
    """
    SVT algorithm for matrix imputation.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    tau : float
        The thresholding parameter for singular values. Controls how singular values are shrunk during the decomposition process.
        Larger values encourage a sparser, lower-rank solution, while smaller values retain more detail.
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
        >>> recov_data = svt(incomp_data=incomp_data, tau=0.5, logs=True)
        >>> print(recov_data)

    """
    start_time = time.time()  # Record start time

    print(f"OPTI svt: {tau=}")
    # Call the C++ function to perform recovery
    recov_data = native_svt(incomp_data, tau, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation SVT - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
