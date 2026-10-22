import time
from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import IterativeImputer  # Now import IterativeImputer

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def mice(incomp_data, max_iter=1, tol=0.001, initial_strategy='mean', sample_posterior=False, seed=42, logs=True, verbose=True):
    """
    Perform imputation using the Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    max_iter : int, optional
        Maximum number of imputation rounds to perform before returning the imputations computed during the final round. (default is 3).

    tol : float, optional
        Tolerance of the stopping condition. (default is 0.001).

    initial_strategy : str, optional
        Which strategy to use to initialize the missing values. {‘mean’, ‘median’, ‘most_frequent’, ‘constant’} (default is "means").

    sample_posterior : bool, optional
        Whether to sample from the (Gaussian) predictive posterior of the fitted estimator for each imputation. Estimator must support return_std in its predict method if set to True. Set to True if using IterativeImputer for multiple imputations.

    seed : int, optional
        The seed of the pseudo random number generator to use. Randomizes selection of estimator features (default is 42).

    logs : bool, optional
        Whether to log the execution time (default is True).

    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Example
    -------
        >>> recov_data = mice(incomp_data, max_iter=3, tol=0.001, initial_strategy='mean', sample_posterior=False, seed=42)
        >>> print(recov_data)

    References
    ----------
    P. Royston and I. R. White. Multiple Imputation by Chained Equations (MICE): Implementation in Stata. Journal of Statistical Software, 45(4):1–20, 2011. Available: https://www.jstatsoft.org/index.php/jss/article/view/v045i04.
    Stef van Buuren, Karin Groothuis-Oudshoorn (2011). “mice: Multivariate Imputation by Chained Equations in R”. Journal of Statistical Software 45: 1-67.
    S. F. Buck, (1960). “A Method of Estimation of Missing Values in Multivariate Data Suitable for use with an Electronic Computer”. Journal of the Royal Statistical Society 22(2): 302-306.
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
    """

    if verbose:
        print(f"(IMPUTATION) MICE\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tmax_iter: {max_iter}\n\ttol: {tol}\n\tinitial_strategy: {initial_strategy}\n\tsample_posterior:{sample_posterior}\n\tseed: {seed}")

    start_time = time.time()  # Record start time

    mice_imputer = IterativeImputer(max_iter=max_iter, tol=tol, initial_strategy=initial_strategy, random_state=seed, sample_posterior=sample_posterior)
    recov_data = mice_imputer.fit_transform(incomp_data)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation MICE - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
