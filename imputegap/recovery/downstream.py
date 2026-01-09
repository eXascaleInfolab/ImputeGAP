import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from imputegap.tools import utils



class Downstream:
    """
    A class to evaluate the performance of imputation algorithms using downstream analysis.

    This class provides tools to assess the quality of imputed time series data by analyzing
    the performance of downstream forecasting models. It computes metrics such as Mean Absolute
    Error (MAE) and Mean Squared Error (MSE) and visualizes the results for better interpretability.

    ImputeGAP downstream models for forcasting : ['arima', 'bats', 'croston', 'deepar', 'ets', 'exp-smoothing',
    'hw-add', 'lightgbm', 'lstm', 'naive', 'nbeats', 'prophet', 'sf-arima', 'theta',
    'transformer', 'unobs', 'xgboost']

    Attributes
    ----------
    input_data : numpy.ndarray
        The original time series without contamination (ground truth).

    recov_data : numpy.ndarray
        The imputed time series to evaluate.

    incomp_data : numpy.ndarray
        The time series with contamination (NaN values).

    downstream : dict
        Configuration for the downstream analysis, including the evaluator, model, and parameters.

    Methods
    -------
    __init__(input_data, recov_data, incomp_data, downstream)
        Initializes the Downstream class with the provided data and configuration.
    downstream_analysis()
        Performs downstream analysis, computes metrics, and optionally visualizes results.
    _plot_downstream(y_train, y_test, y_pred, incomp_data, title="Ground Truth vs Predictions", max_series=4)
        Static method to plot ground truth vs. predictions for contaminated series.
    """



    def __init__(self, input_data, recov_data, incomp_data, algorithm, downstream, verbose=True):
        """
        Initialize the Downstream class

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series without contamination.

        recov_data : numpy.ndarray
            The imputed time series.

        incomp_data : numpy.ndarray
            The time series with contamination (NaN values).

        algorithm : str
            Name of the algorithm to analyse.

        downstream : dict
            Information about the model to launch with its parameters

        verbose : bool
            Display or not information.
        """
        self.input_data = input_data
        self.recov_data = recov_data
        self.incomp_data = incomp_data
        self.downstream = downstream
        self.algorithm = algorithm
        self.sktime_models = utils.list_of_downstreams_sktime()
        self.verbose = verbose

    def downstream_analysis(self):
        """
        Compute a set of evaluation metrics with a downstream analysis

        ImputeGAP downstream models for forcasting : ['arima', 'bats', 'croston', 'deepar', 'ets', 'exp-smoothing',
        'hw-add', 'lightgbm', 'lstm', 'naive', 'nbeats', 'prophet', 'sf-arima', 'theta',
        'transformer', 'unobs', 'xgboost']

        Returns
        -------
        dict or None
            Metrics from the downstream analysis or None if no valid evaluator is provided.
        """
        evaluator = self.downstream.get("task", "forecast")
        model = self.downstream.get("model", "naive")
        params = self.downstream.get("params", None)
        plots = self.downstream.get("plots", True)
        baseline = self.downstream.get("baseline", None)
        split = self.downstream.get("split", 0.8)
        selected_series = self.downstream.get("selected_series", None)

        if baseline is None:
            baseline = self.downstream.get("comparator", None)

        plt = None

        model = model.lower()
        evaluator = evaluator.lower()

        if selected_series is None:
            nan_cols = np.isnan(self.incomp_data).any(axis=0)
            series_indices = int(np.argmax(nan_cols)) if nan_cols.any() else -1
            selected_series = [series_indices]
        elif selected_series == -1:
            selected_series = range(self.input_data.shape[1])

        if not params:
            print("\n\n(DOWNSTREAM) Default parameters of the downstream model loaded.")
            loader = "forecaster-" + str(model)
            params = utils.load_parameters(query="default", algorithm=loader)

        print(f"\n(DOWNSTREAM) Analysis launched !\n\ttask: {evaluator}\n\tmodel: {model}\n\tparams: {params}\n\tbase algorithm: {str(self.algorithm).lower()}\n\treference algorithm: {str(baseline).lower()}\n\tselected series: {selected_series}\n")

        if evaluator in ["forecast", "forecaster", "forecasting"]:
            y_train_all, y_test_all, y_pred_all = [], [], []
            mae, mse, smape = [], [], []

            for x in range(3):  # Iterate over recov_data, input_data, and mean_impute
                if x == 0:
                    data = self.input_data
                elif x == 1:
                    data = self.recov_data
                elif x == 2:
                    from imputegap.recovery.imputation import Imputation

                    if baseline is not None:
                        impt = utils.config_impute_algorithm(self.incomp_data, algorithm=baseline)
                        impt.impute()
                        data = impt.recov_data
                    else:
                        baseline = "zero-impute"
                        zero_impute = Imputation.Statistics.ZeroImpute(self.incomp_data).impute()
                        data = zero_impute.recov_data

                time_len  = data.shape[0]
                train_len = int(time_len  * split)

                y_train = data[:train_len, :]
                y_test = data[train_len:, :]

                forecaster = utils.config_forecaster(model, params)
                y_pred = np.zeros_like(y_test, dtype=float)

                if model in self.sktime_models:
                    # --- SKTIME APPROACH ---
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    from sktime.forecasting.base import ForecastingHorizon
                    from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

                    mae_list, mse_list, smape_list = [], [], []
                    fh = np.arange(1, y_test.shape[0] + 1)  # Forecast horizon
                    for series_idx in selected_series:
                        series_train = y_train[:, series_idx]

                        if model == "ltsf" or model == "rnn":
                            forecaster.fit(series_train, fh=ForecastingHorizon(fh))
                            series_pred = forecaster.predict()
                        else:
                            forecaster.fit(series_train)
                            series_pred = forecaster.predict(fh=fh)

                        y_pred[:, series_idx] = np.asarray(series_pred).ravel()

                        # Compute metrics using sktime
                        mae_list.append(mean_absolute_error(y_test, y_pred))
                        mse_list.append(mean_squared_error(y_test, y_pred))
                        scoring_m = MeanAbsolutePercentageError(symmetric=True)
                        smape_list.append(scoring_m.evaluate(y_test, y_pred)*100)  # Compute SMAPE

                else:
                    # --- DARTS APPROACH ---
                    from darts import TimeSeries
                    from darts.metrics import mae as darts_mae, mse as darts_mse
                    from darts.metrics import smape as darts_smape

                    mae_list, mse_list, smape_list = [], [], []
                    fh = y_test.shape[0]
                    for series_idx in selected_series:
                        train_ts = TimeSeries.from_values(y_train[:, series_idx])
                        test_ts = TimeSeries.from_values(y_test[:, series_idx])

                        forecaster.fit(train_ts)
                        pred_ts = forecaster.predict(n=fh)

                        y_pred[:, series_idx] = pred_ts.values().ravel()

                        # Ensure pred_ts has the same components as y_test_ts
                        pred_ts = pred_ts.with_columns_renamed(pred_ts.components, test_ts.components)

                        # Shift time index to match
                        if pred_ts.start_time() != test_ts.start_time():
                            pred_ts = pred_ts.shift(test_ts.start_time() - pred_ts.start_time())

                        mae_list.append(darts_mae(test_ts, pred_ts))
                        mse_list.append(darts_mse(test_ts, pred_ts))
                        smape_list.append(darts_smape(test_ts, pred_ts))

                mae.append(float(np.mean(mae_list)))
                mse.append(float(np.mean(mse_list)))
                smape.append(float(np.mean(smape_list)))

                # Store for plotting
                y_train_all.append(y_train)
                y_test_all.append(y_test)
                y_pred_all.append(y_pred)

            # Save metrics in a dictionary
            al_name = "MSE_" + self.algorithm.lower()
            al_name_s = "sMAPE_" + self.algorithm.lower()
            al_name_c = "MSE_" + baseline.lower()
            al_name_cs = "sMAPE_" + baseline.lower()

            metrics = {"MSE_original": mse[0], al_name: mse[1], al_name_c: mse[2],
                       "sMAPE_original": smape[0], al_name_s: smape[1], al_name_cs: smape[2] }

            if plots:
                # Global plot with all rows and columns
                here = os.path.dirname(os.path.dirname(__file__))
                save_path = os.path.join(here, "imputegap_assets/downstream")

                plt = self._plot_downstream(y_train=y_train_all, y_test=y_test_all, y_pred=y_pred_all,
                                            incomp_data=self.incomp_data, algorithm=self.algorithm, comparison=baseline,
                                            model=model, type=evaluator, save_path=save_path)

            return metrics, plt

        else:
            print("\tNo evaluator found... list possible : 'forecaster'" + "*" * 30 + "\n")

            return None

    @staticmethod
    def _plot_downstream(y_train, y_test, y_pred, incomp_data, algorithm, comparison, model=None, type=None, title="", max_series=1, save_path="./imputegap_assets/downstream"):
        """
        Plot ground truth vs. predictions for contaminated series (series with NaN values).

        Parameters
        ----------
        y_train : np.ndarray
            Training data array of shape (n_series, train_len).
        y_test : np.ndarray
            Testing data array of shape (n_series, test_len).
        y_pred : np.ndarray
            Forecasted data array of shape (n_series, test_len).
        incomp_data : np.ndarray
            Incomplete data array of shape (n_series, total_len), used to identify contaminated series.
        model : str
            Name of the current model used
        algorithm : str
            Name of the current algorithm used
        comparison : str
            Name of the current algorithm used as comparison
        type : str
            Name of the current type used
        title : str
            Title of the plot.
        max_series : int
            Maximum number of series to plot (default is 9).

        Returns
        -------
        plt
            Return the plots object.
        """
        # Create a 3x3 subplot grid (3 rows for data types, 3 columns for valid series)

        x_size = max_series * 5

        if max_series == 1:
            x_size = 24

        fig, axs = plt.subplots(3, max_series, figsize=(x_size, 15))
        fig.canvas.manager.set_window_title("downstream evaluation")
        fig.suptitle(title, fontsize=16)

        # Find indices of the first 4 valid (non-NaN) series
        valid_indices = [i for i in range(incomp_data.shape[1]) if np.isnan(incomp_data[:, i]).any()][:max_series]

        # Iterate over the three data types (recov_data, input_data, mean_impute)
        for row_idx in range(len(y_train)):
            for col_idx, series_idx in enumerate(valid_indices):
                ax = axs[row_idx, col_idx] if max_series > 1 else axs[row_idx]

                # Extract the corresponding data for this data type and series
                s_y_train = y_train[row_idx]
                s_y_test = y_test[row_idx]
                s_y_pred = y_pred[row_idx]

                train_series = s_y_train[:, series_idx]
                test_series = s_y_test[:, series_idx]
                pred_series = s_y_pred[:, series_idx]

                # Combine training and testing data for visualization
                full_series = np.concatenate([train_series, test_series])

                # Plot training data
                ax.plot(range(len(train_series)), train_series, color="green")

                # Plot ground truth (testing data)
                ax.plot(
                    range(len(train_series), len(full_series)),
                    test_series,
                    label="ground truth",
                    color="green"
                )

                label = type + " " + model
                # Plot forecasted data
                ax.plot(
                    range(len(train_series), len(full_series)),
                    pred_series,
                    label=label,
                    linestyle="--",
                    marker=None,
                    color="red"
                )

                # Add a vertical line at the split point
                ax.axvline(x=len(train_series), color="orange", linestyle="--")

                # Add labels, title, and grid
                if row_idx == 0:
                    ax.set_title(f"original data, series_{series_idx+1}")
                elif row_idx == 1:
                    ax.set_title(f"{algorithm.lower()} imputation, series_{series_idx+1}")
                else:
                    ax.set_title(f"{comparison.lower()} imputation, series_{series_idx+1}")

                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Value")
                ax.legend(loc='upper left', fontsize=7, frameon=True, fancybox=True, framealpha=0.8)
                ax.grid()

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(top=0.92, hspace=0.4)

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            file_path = os.path.join(save_path + "/" + current_time + "_" + type + "_" + model + "_downstream.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            print("plots saved in: ", save_path)

        plt.show()

        return plt
