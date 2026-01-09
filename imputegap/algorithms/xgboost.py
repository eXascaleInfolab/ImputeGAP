import time

import numpy as np
import pandas as pd
from imputegap.tools import utils

from xgboost import XGBRegressor


def xgboost(incomp_data, n_estimators=10, seed=42, logs=True, verbose=True):
    """
    Perform imputation using the Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    n_estimators : int, optional
        The number of trees in the Random Forest model used for imputation (default is 10).
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
        >>> recov_data = xgboost(incomp_data, n_estimators=10, seed=42)
        >>> print(recov_data)

    References
    ----------
    Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 785–794. https://doi.org/10.1145/2939672.2939785
    https://dl.acm.org/doi/10.1145/2939672.2939785
    https://medium.com/@tzhaonj/imputing-missing-data-using-xgboost-802757cace6d
    """

    if verbose:
        print(f"(IMPUTATION) XGBOOST\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tn_estimators: {n_estimators}\n\tseed: {seed}\n")

    if isinstance(incomp_data, np.ndarray):
        incomp_data = pd.DataFrame(incomp_data)

    recov_data = incomp_data.copy()

    start_time = time.time()  # Record start time

    for column in recov_data.columns:
        model = XGBRegressor(n_estimators=n_estimators, random_state=seed)

        non_missing = recov_data.loc[incomp_data[column].notna()]
        missing = recov_data.loc[incomp_data[column].isna()]

        X_train = non_missing.drop(columns=[column])
        y_train = non_missing[column]
        X_missing = missing.drop(columns=[column])

        # Fit the model
        model.fit(X_train, y_train)

        # Predict missing values
        predictions = model.predict(X_missing)

        # Assign the predicted values
        recov_data.loc[recov_data[column].isna(), column] = predictions

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation XGBOOST - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return np.array(recov_data)
