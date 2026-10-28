import ctypes as __native_c_types_import
import time

from imputegap.tools import utils

def native_stmvl(__py_matrix, __py_window, __py_gamma, __py_alpha, __verbose=True):  # noqa: PYI063
    """
    Perform matrix imputation using the STMVL algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).

    __py_window : int
        The window size for the temporal component in the STMVL algorithm.

    __py_gamma : float
        The smoothing parameter for temporal weight (0 < gamma < 1).

    __py_alpha : float
        The power for the spatial weight.

    __verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    Notes
    -----
    The STMVL algorithm leverages temporal and spatial relationships to recover missing values in a matrix.
    The native C++ implementation is invoked for better performance.

    Example
    -------
        >>> recov_data = stmvl(incomp_data=incomp_data, window_size=2, gamma=0.85, alpha=7)
        >>> print(recov_data)

    References
    ----------
    Yi, X., Zheng, Y., Zhang, J., & Li, T. ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data.
    School of Information Science and Technology, Southwest Jiaotong University; Microsoft Research; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences.
    """

    shared_lib = utils.load_share_lib("lib_stmvl", verbose=__verbose)

    __py_sizen = len(__py_matrix);
    __py_sizem = len(__py_matrix[0]);

    assert (__py_window >= 2);
    assert (__py_gamma > 0.0);
    assert (__py_gamma < 1.0);
    assert (__py_alpha > 0.0);

    __ctype_sizen = __native_c_types_import.c_ulonglong(__py_sizen);
    __ctype_sizem = __native_c_types_import.c_ulonglong(__py_sizem);

    __ctype_window = __native_c_types_import.c_ulonglong(__py_window);
    __ctype_gamma = __native_c_types_import.c_double(__py_gamma);
    __ctype_alpha = __native_c_types_import.c_double(__py_alpha);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_input_matrix = utils.__marshal_as_native_column(__py_matrix);

    shared_lib.stmvl_imputation_parametrized(
        __ctype_input_matrix, __ctype_sizen, __ctype_sizem,
        __ctype_window, __ctype_gamma, __ctype_alpha
    );

    __py_recovered = utils.__marshal_as_numpy_column(__ctype_input_matrix, __py_sizen, __py_sizem);

    return __py_recovered;


def stmvl(incomp_data, window_size=7, gamma=0.85, alpha=2, logs=True, verbose=True):
    """
    ST-MVL algorithm for imputation of missing data

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    window_size : int
        window size for temporal component
    gamma : float
        smoothing parameter for temporal weight
    alpha : float
        power for spatial weight
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is False).
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
        >>> recov_data = stmvl(incomp_data=incomp_data, window_size=7, gamma=0.85, alpha=7, logs=True)
        >>> print(recov_data)

    References
    ----------
    Yi, X., Zheng, Y., Zhang, J., & Li, T. ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data.
    School of Information Science and Technology, Southwest Jiaotong University; Microsoft Research; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences.

    """
    window_size, gamma, alpha = utils.patch_classifiers_stmvl(incomp_data, window_size, gamma, alpha)

    if verbose:
        print(f"(IMPUTATION) ST-MVL\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\twindow_size: {window_size}\n\tgamma: {gamma}\n\talpha: {alpha}")

    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_stmvl(incomp_data, window_size, gamma, alpha, verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation stvml - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
