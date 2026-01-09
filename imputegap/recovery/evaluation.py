import numpy as np

class Evaluation:
    """
    A class to evaluate the performance of imputation algorithms by comparing imputed time series with the ground truth.

    Methods
    -------
    compute_all_metrics():
        Compute various evaluation metrics (RMSE, MAE, MI, CORRELATION) for the imputation.
    compute_rmse():
        Compute the Root Mean Squared Error (RMSE) between the ground truth and the imputed values.
    compute_mae():
        Compute the Mean Absolute Error (MAE) between the ground truth and the imputed values.
    compute_mi():
        Compute the Mutual Information (MI) between the ground truth and the imputed values.
    compute_correlation():
        Compute the Pearson correlation coefficient between the ground truth and the imputed values.

    """


    def __init__(self, input_data, recov_data, incomp_data, algorithm="", mask=None, security=True, verbose=True):
        """
        Initialize the Evaluation class with ground truth, imputation, and incomp_data time series.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series without contamination.

        recov_data : numpy.ndarray
            The imputed time series.

        incomp_data : numpy.ndarray
            The time series with contamination (NaN values).

        algorithm : str, optional
            Name of the algorithm to evaluate.

        mask: numpy.ndarray, optional
            Mask describing the missing values (default is None).
            If None, take the missing values in the ts_m matrix (contaminated matrix).

        security: bool, optional
            Check if the no imputed values are the same in the imputed matrix and the raw data

        verbose : bool, optional
            Display of not the anomaly (default: True).

        Returns
        -------
        None
        """
        self.input_data = input_data
        self.recov_data = recov_data
        self.incomp_data = incomp_data
        self.large_error = 100
        self.algorithm = algorithm
        self.verbose = verbose
        self.security = security
        self.mask = mask

    def compute_all_metrics(self):
        """
        Compute a set of evaluation metrics for the imputation based on the ground truth and contamination data.

        The metrics include RMSE, MAE, Mutual Information (MI), and Pearson Correlation.

        Returns
        -------
        dict
            A dictionary containing the computed metrics:
            - "RMSE": Root Mean Squared Error
            - "MAE": Mean Absolute Error
            - "MI": Mutual Information
            - "CORRELATION": Pearson Correlation Coefficient
        """

        if self.mask is None:
            nan_locations = np.isnan(self.incomp_data)
        else:
            nan_locations = self.mask

        recov_vals = self.recov_data[nan_locations]

        if np.isnan(recov_vals).all():
            print(f"(EVAL) {self.algorithm} ended with errors, the imputed time series contains NaN values.\n\tPlease, check your configuration, the algorithm might not work with the percentage of contamination or the pattern chosen.")

        rmse = self.compute_rmse()
        mae = self.compute_mae()
        mi_d = self.compute_mi()
        correlation = self.compute_correlation()

        metrics = {"RMSE": rmse, "MAE": mae, "MI": mi_d, "CORRELATION": correlation}

        return metrics

    def compute_rmse(self):
        """
        Compute the Root Mean Squared Error (RMSE) between the ground truth and imputed values for NaN positions in contamination.

        The RMSE measures the average magnitude of the error between the imputed values and the ground truth,
        giving higher weight to large errors.

        Returns
        -------
        float
            The RMSE value for NaN positions in the contamination dataset.
        """
        if self.mask is None:
            nan_locations = np.isnan(self.incomp_data)
        else:
            nan_locations = self.mask

        mse = np.mean((self.input_data[nan_locations] - self.recov_data[nan_locations]) ** 2)
        rmse = np.sqrt(mse)

        if rmse > self.large_error:
            if self.verbose:
                print("Extreme error detected, limited to ", self.large_error)
            rmse = self.large_error

        return float(rmse)

    def compute_mae(self):
        """
        Compute the Mean Absolute Error (MAE) between the ground truth and imputed values for NaN positions in contamination.

        The MAE measures the average magnitude of the error in absolute terms, making it more robust to outliers than RMSE.

        Returns
        -------
        float
            The MAE value for NaN positions in the contamination dataset.
        """
        if self.mask is None:
            nan_locations = np.isnan(self.incomp_data)
        else:
            nan_locations = self.mask

        absolute_error = np.abs(self.input_data[nan_locations] - self.recov_data[nan_locations])
        mean_absolute_error = np.mean(absolute_error)

        if mean_absolute_error > self.large_error:
            if self.verbose:
                print("Extreme error detected, limited to ", self.large_error)
            mean_absolute_error = self.large_error

        return float(mean_absolute_error)

    def compute_mi(self):
        """
        Compute the Mutual Information (MI) between the ground truth and imputed values for NaN positions in contamination.

        MI measures the amount of shared information between the ground truth and the imputed values,
        indicating how well the imputation preserves the underlying patterns of the data.

        Returns
        -------
        float
            The mutual information (MI) score for NaN positions in the contamination dataset.
        """
        from sklearn.metrics import mutual_info_score

        if self.mask is None:
            nan_locations = np.isnan(self.incomp_data)
        else:
            nan_locations = self.mask

        input_vals = self.input_data[nan_locations]
        recov_vals = self.recov_data[nan_locations]

        if (np.isnan(recov_vals).all() or np.isnan(input_vals).all()) and self.security:
            return np.nan

        # Discretize the continuous data into bins
        input_data_binned = np.digitize(input_vals, bins=np.histogram_bin_edges(input_vals, bins=10))
        imputation_binned = np.digitize(recov_vals, bins=np.histogram_bin_edges(recov_vals, bins=10))

        mi_discrete = mutual_info_score(input_data_binned, imputation_binned)
        # mi_continuous = mutual_info_score(self.input_data[nan_locations], self.input_data[nan_locations])

        return mi_discrete

    def compute_correlation(self):
        """
        Compute the Pearson Correlation Coefficient between the ground truth and imputed values for NaN positions in contamination.

        Pearson Correlation measures the linear relationship between the ground truth and imputed values,
        with 1 being a perfect positive correlation and -1 a perfect negative correlation.

        Returns
        -------
        float
            The Pearson correlation coefficient for NaN positions in the contamination dataset.
        """
        from scipy.stats import pearsonr

        if self.mask is None:
            nan_locations = np.isnan(self.incomp_data)
        else:
            nan_locations = self.mask

        input_data_values = self.input_data[nan_locations]
        imputed_values = self.recov_data[nan_locations]

        # Check if input data is constant (i.e., no variance)
        if np.all(input_data_values == input_data_values[0]) or np.all(imputed_values == imputed_values[0]) and self.security:
            if self.verbose:
                print("\t\t\t\nAn input array is constant; the correlation coefficient is not defined, set to 0")
            return np.nan

        correlation, _ = pearsonr(input_data_values, imputed_values)

        if np.isnan(correlation):
            correlation = np.nan

        return correlation

