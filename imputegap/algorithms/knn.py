import time

import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

def knn(incomp_data, k=5, weights="uniform", logs=True, verbose=True):
    """
    Perform imputation using the K-Nearest Neighbor (KNN) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    k : int, optional
        Number of nearest neighbor (default is 5).
    weights : str, optional
        "uniform" for mean, "distance" for inverse-distance weighting.
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
        >>> recov_data = knn(incomp_data, k=5)
        >>> print(recov_data)

    """

    if verbose:
        print(f"(IMPUTATION) KNNImpute\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tk: {k}\n\tweights: {weights}\n")

    start_time = time.time()  # Record start time

    recov_data = np.copy(incomp_data)  # Copy data to avoid modifying original
    _, num_cols = recov_data.shape  # Get dataset dimensions


    # Standardize data (important for correct neighbor selection)
    norm_data = (incomp_data - np.nanmean(incomp_data, axis=0)) / (np.nanstd(incomp_data, axis=0) + 1e-8)

    # Compute nan-euclidean distance matrix
    dist_matrix = nan_euclidean_distances(norm_data)

    for j in range(num_cols):  # Column-wise processing
        missing_indices = np.where(np.isnan(recov_data[:, j]))[0]  # Indices of missing values
        available_indices = np.where(~np.isnan(incomp_data[:, j]))[0]  # Indices of available values

        if len(available_indices) == 0:
            continue  # Skip column if all values are NaN

        for i in missing_indices:  # Process each missing value
            distances = dist_matrix[i, available_indices]  # Distances to known values
            values = incomp_data[available_indices, j]  # Corresponding known values

            # Select k-nearest neighbors
            k_neighbors = np.argsort(distances)[:min(k, len(distances))]
            nearest_values = values[k_neighbors]
            nearest_distances = distances[k_neighbors]

            # Fallback to column mean if no valid neighbors
            if len(nearest_values) == 0:
                recov_data[i, j] = np.nanmean(incomp_data[:, j])
                continue

            # Compute imputed value
            if weights == "uniform":
                recov_data[i, j] = np.mean(nearest_values)
            elif weights == "distance":
                weight_factors = 1 / (nearest_distances + 1e-5)
                recov_data[i, j] = np.dot(weight_factors, nearest_values) / np.sum(weight_factors)


    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation knn_impute - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
