import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils


def native_cdrec(__py_matrix, __py_rank, __py_epsilon, __py_iterations, __verbose=True):
    """
    Perform matrix imputation using the CDRec algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_rank : int
        The truncation rank for matrix decomposition (must be greater than 0 and less than the number of columns).
    __py_epsilon : float
        The epsilon value, used as the threshold for stopping iterations based on difference.
    __py_iterations : int
        The maximum number of allowed iterations for the algorithm.
    __verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    References
    ----------
    Khayati, M., Cudré-Mauroux, P. & Böhlen, M.H. Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl Inf Syst 62, 2257–2280 (2020). https://doi.org/10.1007/s10115-019-01421-7
    """

    shared_lib = utils.load_share_lib("lib_cdrec", verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);

    assert (__py_rank >= 0);
    assert (__py_rank < __py_m);
    assert (__py_epsilon > 0);
    assert (__py_iterations > 0);

    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
    __ctype_epsilon = __native_c_types_import.c_double(__py_epsilon);
    __ctype_iterations = __native_c_types_import.c_ulonglong(__py_iterations);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.cdrec_imputation_parametrized(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank, __ctype_epsilon, __ctype_iterations);

    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def cdrec(incomp_data, truncation_rank, iterations, epsilon, logs=True, verbose=True, lib_path=None):
    """
    CDRec algorithm for matrix imputation of missing values using Centroid Decomposition.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    truncation_rank : int
        The truncation rank for matrix decomposition (must be greater than 1 and smaller than the number of series).
    epsilon : float
        The learning rate (stopping criterion threshold).
    iterations : int
        The maximum number of iterations allowed for the algorithm.
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
        >>> recov_data = cdrec(incomp_data=incomp_data, truncation_rank=1, iterations=100, epsilon=0.000001, logs=True)
        >>> print(recov_data)

    """

    if verbose:
        print(f"\n(IMPUTATION) CDRec\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\ttruncation rank: {truncation_rank}\n\tepsilon: {epsilon}\n\titerations: {iterations}\n")
        print(f"call: cdrec.impute(params={{'rank': {truncation_rank}, 'epsilon': {epsilon}, 'iterations': {iterations}}})\n")

    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_cdrec(incomp_data, truncation_rank, epsilon, iterations, False)

    end_time = time.time()

    if logs and verbose:
        print(f"> logs: imputation cdrec - Execution Time: {(end_time - start_time):.4f} seconds.")

    return recov_data
