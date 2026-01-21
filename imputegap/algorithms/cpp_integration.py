import time
import ctypes as __native_c_types_import;

from imputegap.tools import utils

# =========================================================== #
# IN CASE OF NEED, YOU CAN ADAPT AND TAKE CDREC.PY AS A MODEL #
# =========================================================== #

def native_algo(__py_matrix, __py_param, __verbose=True):
    """
    Perform matrix imputation using the CDRec algorithm with native C++ support.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input matrix with missing values (NaNs).
    __py_param : int
        parameters to adapt
    __verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    numpy.ndarray
        The recovered matrix after imputation.

    """

    lib_file = "lib_soft_impute" # to adapt
    shared_lib = utils.load_share_lib(lib_file, verbose=__verbose)

    __py_n = len(__py_matrix);
    __py_m = len(__py_matrix[0]);


    __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
    __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

    # depends on your needs
    __py_param = __native_c_types_import.c_ulonglong(__py_param);

    # Native code uses linear matrix layout, and also it's easier to pass it in like this
    __ctype_matrix = utils.__marshal_as_native_column(__py_matrix);

    # call your algorithm > replace recoverySoftImpute by your_algo_fct
    shared_lib.recoverySoftImpute(__ctype_matrix, __ctype_size_n, __ctype_size_m, __py_param);

    # convert back to numpy
    __py_imputed_matrix = utils.__marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

    return __py_imputed_matrix;


def your_algo(contamination, param, logs=True, verbose=True):
    """
    CDRec algorithm for matrix imputation of missing values using Centroid Decomposition.

    Parameters
    ----------
    contamination : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    param : to adapt
        to adapt
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
    """
    start_time = time.time()  # Record start time

    # Call the C++ function to perform recovery
    recov_data = native_algo(contamination, param, verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation algo - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
