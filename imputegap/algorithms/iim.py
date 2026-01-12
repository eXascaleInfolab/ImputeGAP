import time
from imputegap.wrapper.AlgoPython.IIM.runnerIIM import impute_with_algorithm


def iim(incomp_data, number_neighbor, algo_code, logs=True, verbose=True):
    """
    Perform imputation using the Iterative Imputation Method (IIM) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    number_neighbor : int
        The number of neighbors to use for the K-Nearest Neighbors (KNN) classifier (default is 10).
    algo_code : str
        The specific action code for the IIM output. This determines the behavior of the algorithm.
    logs : bool, optional
        Whether to log the execution time (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The IIM algorithm works by utilizing K-Nearest Neighbors (KNN) to estimate missing values in time series data.
    Depending on the provided `algo_code`, different versions of the algorithm may be executed.

    The function logs the total execution time if `logs` is set to True.

    Example
    -------
        >>> recov_data = iim(incomp_data, number_neighbor=10, algo_code="iim 2")
        >>> print(recov_data)

    References
    ----------
    A. Zhang, S. Song, Y. Sun and J. Wang, "Learning Individual Models for Imputation," 2019 IEEE 35th International Conference on Data Engineering (ICDE), Macao, China, 2019, pp. 160-171, doi: 10.1109/ICDE.2019.00023.
    keywords: {Data models;Adaptation models;Computational modeling;Predictive models;Numerical models;Aggregates;Regression tree analysis;Missing values;Data imputation}
    """
    start_time = time.time()  # Record start time

    recov_data = impute_with_algorithm(algo_code, incomp_data.copy(), number_neighbor, verbose=verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation iim - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
