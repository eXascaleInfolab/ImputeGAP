import time

import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def miss_forest(incomp_data, n_estimators=10, max_iter=3, max_features='sqrt', seed=42, logs=True, verbose=True):
    """
    Perform imputation using the MissForest algorithm (Random-Forest-based imputation for mixed data)

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    n_estimators : int, optional
        The number of trees in the Random Forest model used for imputation (default is 10).
    max_iter : int, optional
        Maximum number of imputation rounds to perform before returning the final imputed matrix (default is 3).
    max_features : {'auto', 'sqrt', 'log2', float, int}, optional
        The number of features to consider when looking for the best split during imputation.
        - 'sqrt' (default): Uses the square root of the total number of features.
        - 'auto': Uses all features.
        - 'log2': Uses log2 of the total features.
        - float or int: Can specify a fraction or fixed number of features.
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
        >>> recov_data = mice(incomp_data, n_estimators=10, max_iter=3, max_features='sqrt', seed=42)
        >>> print(recov_data)

    References
    ----------
    Daniel J. Stekhoven, Peter Bühlmann, MissForest—non-parametric missing value imputation for mixed-type data, Bioinformatics, Volume 28, Issue 1, January 2012, Pages 112–118, https://doi.org/10.1093/bioinformatics/btr597
    https://github.com/yuenshingyan/MissForest
    https://pypi.org/project/MissForest/
    """
    incomp_data = incomp_data

    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)
    verbose_int = 2 if verbose else 0  # use 2 for full output, 0 for none

    if verbose:
        print(f"(IMPUTATION) MISS FOREST\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tn_estimators: {n_estimators}\n\tmax_iter: {max_iter}\n\tmax_features: {max_features}\n\tseed: {seed}\n")

    # Convert numpy array to pandas DataFrame if needed
    if isinstance(incomp_data, np.ndarray):
        incomp_data = pd.DataFrame(incomp_data)

    start_time = time.time()  # Record start time

    # Define custom estimators with desired parameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=seed)
    rgr = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=seed)

    # Initialize MissForest with custom estimators
    mf_imputer = MissForest(clf=clf, rgr=rgr, max_iter=max_iter, verbose=verbose_int)
    recov_data = mf_imputer.fit_transform(incomp_data)
    recov_data = np.array(recov_data)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation MISS FOREST - Execution Time: {(end_time - start_time):.4f} seconds\n")

    recov[m_mask] = recov_data[m_mask]

    return recov
