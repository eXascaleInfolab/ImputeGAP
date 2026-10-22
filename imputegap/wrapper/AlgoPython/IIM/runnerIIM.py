import re

import numpy
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

global rmse;


def iim_recovery(matrix_nan: np.ndarray, adaptive_flag: bool = False, learning_neighbors: int = 10):
    """Implementation of the IIM algorithm
    Via the adaptive flag, the algorithm can be run in two modes:
    - Adaptive: The algorithm will run the adaptive version of the algorithm, as described in the paper
        - Essentially, the algorithm will determine the best number of learning neighbors
    - Non-adaptive: The algorithm will run the non-adaptive version of the algorithm, as described in the paper

    Parameters
    ----------
    matrix_nan : np.ndarray
        The complete matrix of values with missing values in the form of NaN.
    adaptive_flag : bool, optional
        Whether to use the adaptive version of the algorithm, by default False.
    learning_neighbors : int, optional
        The number of neighbors to use for the KNN classifier, by default 10.

    Returns
    -------
    matrix_nan : np.ndarray
        The complete matrix of values with missing values imputed.
    """
    tuples_with_nan = np.isnan(matrix_nan).any(axis=1)
    if np.any(tuples_with_nan):  # if there are any tuples with missing values as NaN
        incomplete_tuples_indices = np.array(np.where(tuples_with_nan))
        incomplete_tuples = matrix_nan[tuples_with_nan]
        complete_tuples = matrix_nan[~tuples_with_nan]  # Rows that do not contain a NaN value

        if not isinstance(learning_neighbors, (int, numpy.int64)):
            learning_neighbors = learning_neighbors[0]

        if learning_neighbors > len(complete_tuples):
            learning_neighbors = min(len(complete_tuples), learning_neighbors)
        if len(complete_tuples) == 0:
            #print("No complete tuples found, unable to proceed, returning original matrix")
            return matrix_nan
        if adaptive_flag:
            lr_models = adaptive(complete_tuples, incomplete_tuples, learning_neighbors, max_learning_neighbors=min(len(complete_tuples), 10))
            imputation_result = imputation(incomplete_tuples, lr_models)

        else:
            lr_models = learning(complete_tuples, incomplete_tuples, learning_neighbors)
            imputation_result = imputation(incomplete_tuples, lr_models)

        for result in imputation_result:
            matrix_nan[np.array(incomplete_tuples_indices)[:, result[0]], result[1]] = result[2]
        return matrix_nan
    else:
        print("No missing values as NaN, returning original matrix")
        return matrix_nan


#  Algorithm 1: Learning
def learning(complete_tuples: np.ndarray, incomplete_tuples: np.ndarray, learn: int = 10):
    """Learns individual regression models for each learning neighbor and each attribute,
       by fitting on the other attributes and the missing attribute

    Parameters
    ----------
    complete_tuples : np.ndarray
        The complete matrix of values without missing values.
        Should already be normalized.
    incomplete_tuples : np.ndarray
        The complete matrix of values with missing values in the form of NaN.
        Should already be normalized.
    learn : int, optional
        The number of neighbors to use for the KNN classifier, by default 10.

    Returns
    -------
    model_params: np.ndarray[Ridge]
        The learned regression models. The structure of this array is as follows:
        - First dimension: which tuple we are looking at
        - Second dimension: which attribute we are looking at
        - Third dimension: which learning neighbor we are looking at
    """

    knn_euc = NearestNeighbors(n_neighbors=learn, metric='euclidean').fit(complete_tuples)
    number_of_attributes = incomplete_tuples.shape[1]  # Number of attributes, should be 12
    model_params = np.empty((len(incomplete_tuples), number_of_attributes, learn), dtype=object)

    # Replace NaN values with 0
    incomplete_tuples_no_nan = np.nan_to_num(incomplete_tuples)

    # Find the k nearest neighbors for all incomplete tuples at once
    learning_neighbors = knn_euc.kneighbors(incomplete_tuples_no_nan, return_distance=False)

    for tuple_index, incomplete_tuple in enumerate(incomplete_tuples):
        nan_indicator = np.isnan(incomplete_tuple)

        # Check if there is more than one NaN value in the tuple
        if np.count_nonzero(nan_indicator) == 1:
            nan_index = np.where(nan_indicator)[0][0]  # Get the index of the NaN value

            # Learn the relevant value/column
            X = complete_tuples[learning_neighbors[tuple_index]][:, ~nan_indicator]
            y = complete_tuples[learning_neighbors[tuple_index]][:, nan_indicator]
            models = [Ridge(tol=1e-20).fit(X_i.reshape(1, -1), y_i) for X_i, y_i in zip(X, y)]
            model_params[tuple_index, nan_index] = [(model.coef_, model.intercept_) for model in models]
        else:
            # If there are multiple NaNs in the tuple
            for missing_value_index in np.where(nan_indicator)[0]:
                # Create a new indicator for this specific NaN
                current_nan_indicator = np.zeros_like(nan_indicator)
                current_nan_indicator[missing_value_index] = True

                # Learn the relevant value/column
                X = complete_tuples[learning_neighbors[tuple_index]][:, ~current_nan_indicator]
                y = complete_tuples[learning_neighbors[tuple_index]][:, current_nan_indicator]
                models = [Ridge(tol=1e-20).fit(X_i.reshape(1, -1), y_i) for X_i, y_i in zip(X, y)]
                model_params[tuple_index, missing_value_index] = [(model.coef_, model.intercept_) for model in models]

    return model_params


# Algorithm 2: Imputation
def imputation(incomplete_tuples: np.ndarray, lr_coef_and_threshold: np.ndarray):
    """ Imputes the missing values of the incomplete tuples using the learned linear regression models.

    Parameters
    ----------
    incomplete_tuples : np.ndarray
        The complete matrix of values with missing values in the form of NaN.
        Should already be normalized.
    lr_coef_and_threshold : np.ndarray[Ridge]
        The learned regression models containing the coefficients and intercepts.

    Returns
    -------
    imputed_values : list[list[int, int, float]]
        Returns the imputed values in the form of a list of lists,
        where each list contains the tuple index, the attribute index and the imputed value.
    """
    imputed_values = []

    # For each missing tuple
    for i, incomplete_tuple in enumerate(incomplete_tuples):  # for t_i in r
        nan_indicator = np.isnan(incomplete_tuple)  # show which attribute is missing as NaN

        # Check if there is more than one NaN value in the tuple
        if np.count_nonzero(nan_indicator) == 1:
            missing_attributes_indices = np.where(nan_indicator)[0]  # indices of missing attributes

            # Prepare the input array for multiple samples
            incomplete_tuple_no_nan = incomplete_tuple[~nan_indicator]

            # For each missing attribute
            for missing_attribute_index in missing_attributes_indices:
                # Predict the missing values using the learned Ridge models
                candidate_suggestions = np.array([coef @ incomplete_tuple_no_nan + intercept for coef, intercept in
                                                  lr_coef_and_threshold[i, missing_attribute_index]])

                distances = compute_distances(candidate_suggestions)
                weights = compute_weights(distances)

                impute_result = np.sum(candidate_suggestions * weights)
                # Create tuple with index (in missing tuples), attribute, imputed value
                imputed_values.append([i, missing_attribute_index, impute_result])
        else:

            missing_attributes_indices = np.where(nan_indicator)[0]  # indices of missing attributes
            # If there are multiple NaNs in the tuple
            for missing_value_index in np.where(nan_indicator)[0]:
                # Create a new indicator for this specific NaN
                current_nan_indicator = np.zeros_like(nan_indicator)
                current_nan_indicator[missing_value_index] = True

                # Prepare the input array for multiple samples
                incomplete_tuple_no_nan = np.nan_to_num(incomplete_tuple[~current_nan_indicator])

                # For each missing attribute
                for missing_attribute_index in missing_attributes_indices:
                    # Predict the missing values using the learned Ridge models
                    candidate_suggestions = np.array([coef @ incomplete_tuple_no_nan + intercept for coef, intercept in
                                                      lr_coef_and_threshold[i, missing_attribute_index]])

                    distances = compute_distances(candidate_suggestions)
                    weights = compute_weights(distances)

                    impute_result = np.sum(candidate_suggestions * weights)
                    # Create tuple with index (in missing tuples), attribute, imputed value
                    imputed_values.append([i, missing_attribute_index, impute_result])

    return imputed_values


# Algorithm 3: Adaptive
def adaptive(complete_tuples: np.ndarray, incomplete_tuples: np.ndarray, k: int, max_learning_neighbors: int = 100, step_size: int = 4):
    """Adaptive learning of regression parameters

    Parameters
    ----------
    complete_tuples : np.ndarray
        The complete matrix of values without missing values.
        Should already be normalized.
    incomplete_tuples : np.ndarray
        The complete matrix of values with missing values in the form of NaN.
        Should already be normalized.
    k : int
        The number of neighbors to use for the k nearest neighbors classifier.
    max_learning_neighbors : int, optional
        The maximum number of neighbors to use for the learning phase, by default 100.
    step_size : int, optional
        The step size for the learning phase, by default 3.

    Returns
    -------
    phi: np.ndarray[Ridge]
        The learned regression parameters for all tuples in r.
    """
    # print("Starting Algorithm 3 'adaptive'")
    all_entries = min(int(complete_tuples.shape[0]), max_learning_neighbors)
    phi_list = [learning(complete_tuples, incomplete_tuples, l_learning)  # for l in 1..n
                for l_learning in
                range(1, all_entries + 1, step_size)]
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(complete_tuples)
    number_of_models = max(len(phi_list) - 1, 1)
    number_of_incomplete_tuples = len(incomplete_tuples)
    costs = np.zeros((number_of_incomplete_tuples, number_of_models))
    # print("Finished learning; Starting main loop of Algorithm 3 'adaptive'")
    for log, complete_tuple in enumerate(complete_tuples, 1):  # for t_i in r
        # if (log % 50) == 0: print("Algorithm 3 'adaptive', processing tuple {}".format(str(log)))
        neighbors = nn.kneighbors(complete_tuple.reshape(1, -1), return_distance=False)[0]
        for incomplete_tuple_idx, incomplete_tuple in enumerate(incomplete_tuples):
            nan_indicator = np.isnan(incomplete_tuple)  # Show which attribute is missing as NaN
            neighbors_filtered = np.delete(complete_tuples[neighbors], nan_indicator, axis=1)
            for my_model in range(0, number_of_models):  # Line 6, for l in 1..n
                model_params_for_tuple = phi_list[my_model][incomplete_tuple_idx]
                for attribute_index, model_params in enumerate(model_params_for_tuple):

                    if model_params is not None and not np.any(model_params is None):  # Only compute cost for NaN attributes
                        coefs, intercepts = zip(*model_params)
                        expanded_coef = np.array(coefs)
                        # set NaN attributes in neighbors_filtered to 0 (Nan to Num should also work)
                        neighbors_filtered_copy = np.nan_to_num(neighbors_filtered)

                        # print(expanded_coef.shape, neighbors_filtered_copy.shape)
                        phi_models = (expanded_coef @ neighbors_filtered_copy[:, :, None]).squeeze() + np.array(
                            intercepts)
                        errors = np.abs(complete_tuple[attribute_index] - phi_models)
                        costs[incomplete_tuple_idx, my_model] += np.sum(np.power(errors, 2)) / len(phi_models)

    # Line 8-10 Select best model for each tuple
    best_models_indices = np.argmin(costs, axis=1)

    number_of_attributes = incomplete_tuples.shape[1]
    lr_models = np.empty((number_of_incomplete_tuples, number_of_attributes, my_model), dtype=object)

    for i in range(number_of_incomplete_tuples):
        best_models_indices_for_tuple = best_models_indices[i]
        lr_models[i] = phi_list[best_models_indices_for_tuple][i]

    return lr_models



def compute_distances(candidate_suggestions: np.ndarray):
    """ Calculate the sum of distances to all other candidates (Manhattan) for each candidate

    Parameters
    ----------
    candidate_suggestions : np.ndarray
        All other candidates to compare the values to.

    Returns
    -------
    distances
        The sum of distances to all other candidates.
    """
    distances = []
    for i in range(len(candidate_suggestions)):
        temp_distances = np.abs(candidate_suggestions[i] - np.delete(candidate_suggestions, i))
        distances.append(np.sum(temp_distances))
    return distances


def compute_weights(distances: List[float]):
    """ A candidate's weight is determined by normalizing by all other candidates' distances.
    All weights together sum up to 1.

    Parameters
    ----------
    distances : list[float]
        A list of all distances.

    Returns
    -------
    weight
        The weight of the candidate.
    """

    distances_n = np.array(distances)
    weights = np.zeros(distances_n.shape)

    nonzero_indices = distances_n != 0
    weights[nonzero_indices] = 1 / distances_n[nonzero_indices] / np.sum(1 / distances_n[nonzero_indices])

    # Handle the case where all distances are zero
    if np.sum(weights) == 0:
        weights = np.ones(distances_n.shape) / len(distances_n)

    return weights


def impute_with_algorithm(alg_code: str, matrix: np.ndarray, neighbors=None, verbose=True):
    """
    Imputes the input matrix with a specified algorithm.

    Parameters
    ----------
    alg_code : str
        The algorithm and its parameters.
        The first parameter is the name, the second the number of neighbors and the third whether to use adaptive or not.
    matrix : np.ndarray
        The input matrix to be imputed.

    Returns
    -------
    np.ndarray
        The imputed matrix.
    """

    if verbose:
        print(f"(IMPUTATION) IIM\n\tMatrix: {matrix.shape[0]}, {matrix.shape[1]}\n\talg_code: {alg_code}\n\tneighbors: {neighbors}\n\tverbose: {verbose}")

    global rmse
    # Imputation
    alg_code_spl = alg_code.split()

    if len(alg_code_spl) > 1:
        match = re.match(r"(\d+)([a-zA-Z]+)", alg_code_spl[1], re.I)
        if match:
            if neighbors is None:
                neighbors, adaptive_flag = match.groups()
            else:
                _, adaptive_flag = match.groups()

            print(f"if: {neighbors}")
            matrix_imputed = iim_recovery(matrix, adaptive_flag=adaptive_flag.startswith("a"), learning_neighbors=int(neighbors))
        else:
            if neighbors is None:
                neighbors = int(alg_code_spl[1])
            matrix_imputed = iim_recovery(matrix, adaptive_flag=False, learning_neighbors=neighbors)

    # verification to check for NaN. If found, assign absurdly high value to them.
    nan_mask = np.isnan(matrix_imputed)
    matrix_imputed[nan_mask] = np.sqrt(np.finfo('d').max / 100000.0)

    return matrix_imputed
