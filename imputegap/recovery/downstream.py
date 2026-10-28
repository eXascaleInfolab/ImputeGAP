# Standard library
import os
import datetime
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# ImputeGAP imports
from imputegap.tools import utils

warnings.filterwarnings("ignore", message="GPU available but not used.*" )

class Forecaster:
    def __init__(self, task="forecast", model="exp-smoothing", y_train=None, season=24, params=None, baseline="mean-impute", selected_series=None, single_pipeline=False, plots=True, save_path="./imputegap_assets/downstream", to_cache=True, use_cache=False, artifact=None, referential=True, bypass_error=False, s_exception=False, verbose=True):
        """
        Container for downstream forecasting configuration and inputs.

        This class stores the settings needed to run a forecasting evaluation in the downstream pipeline
        (e.g., choice of model, seasonality, hyperparameters, baseline imputation method, and plotting options).

        Parameters
        ----------
        task : str, optional
            Downstream task identifier. Default is "forecast".

        model : str, optional
            Forecasting model identifier (e.g., "exp-smoothing"). Default is "exp-smoothing".

        y_train : np.ndarray or None, optional
            Training matrix/array used for forecasting ([timestamps, series]). Default is None.

        season : int, optional
            Seasonality length used by seasonal forecasters (e.g., daily seasonality in hourly data). Default is 24.

        params : dict or None, optional
            Model hyperparameters passed to the forecaster configuration helper. Default is None.

        baseline : str, optional
            Baseline imputation method name used for comparisons (e.g., "mean-impute"). Default is "mean-impute".

        selected_series : list[int] or None, optional
            Indices of the series (columns) to include in forecasting and evaluation. Default is None.

        single_pipeline : bool, optional
            Analyze a single downstream pipeline or a complete one. Default is False.

        plots : bool, optional
            Whether to generate downstream plots. Default is True.

        save_path : str, optional
            Directory where downstream results/plots are saved. Default is "./imputegap_assets/downstream".

        to_cache : bool, optional
            Cache the results of the downstream to prevent to run them again (default: True).

        use_cache: bool, optional
            If a cache exists for a specific use-case, use the cache and do not run the model again (default: False).

        verbose : bool, optional
            Verbosity flag for downstream logging. Default is True.

        Attributes
        ----------
        task, model, y_train, season, params, baseline, selected_series, plots, save_path : see Parameters
        """

        self.task = task
        self.model = model
        self.y_train = y_train
        self.season = season
        self.params = params
        self.single_pipeline = single_pipeline
        self.plots = plots
        self.save_path = save_path
        self.baseline = baseline
        self.selected_series = selected_series
        self.to_cache = to_cache
        self.use_cache = use_cache
        self.artifact = artifact
        self.referential = referential
        self.bypass_error = bypass_error
        self.s_exception = s_exception


class Artifact:
    def __init__(self, dataset, model, algorithm, pattern, x=0.2, fixed_x=0.2, offset=0.05, contamination_by_class=True, imputation_by_class=True, horizon=0.2, verbose=True):
        """
        Container for experiment metadata / configuration.
        """
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        self.pattern = pattern
        self.x = x
        self.fixed_x = fixed_x
        self.offset = offset
        self.contamination_by_class = contamination_by_class
        self.imputation_by_class = imputation_by_class
        self.horizon = horizon
        self.verbose = verbose


class Classifier:
    def __init__(self, task="classifier", model="arsenal", params=None, baseline="mean-impute", X_test=None, y_train=None, y_test=None, imputer_test=None, missing_test=None, do_testset_imputation=False, algorithm=None, dataset=None, selected_series=None, plots=True, save_path="./imputegap_assets/downstream", to_cache=True, use_cache=False, artifact=None, referential=True, bypass_error=False, verbose=True, v2=False):
        """
        Container for downstream time-series classification configuration and inputs.

        This class stores the settings and datasets used to run a classification evaluation in the downstream
        pipeline. It supports optional test-set imputation, where the test data is imputed either by the main
        algorithm or by a baseline imputer before classification.

        Parameters
        ----------
        task : str, optional
            Downstream task identifier. Default is "classifier"

        model : str, optional
            Classifier model identifier (e.g., "arsenal"). Default is "arsenal".

        params : dict or None, optional
            Model hyperparameters passed to the classifier configuration helper. Default is None.

        baseline : str, optional
            Baseline imputation method name used for comparisons (e.g., "mean-impute"). Default is "mean-impute".

        X_test : np.ndarray or None, optional
            Test feature matrix for classification. Default is None.

        y_train : np.ndarray or None, optional
            Training labels. Default is None.

        y_test : np.ndarray or None, optional
            Test labels. Default is None.

        imputer_test : np.ndarray or None, optional
            Test-set data imputed by the main algorithm (used when `do_testset_imputation=True`). Default is None.

        missing_test : np.ndarray or None, optional
            Test-set data with missing values (used for baseline test-set imputation). Default is None.

        do_testset_imputation : bool, optional
            If True, replace `X_test` with an imputed version (either `imputer_test` for the main algorithm,
            or a baseline-imputed version of `missing_test`). Default is False.

        algorithm : str or None, optional
            Name of the main imputation/recovery algorithm used upstream (for labeling/reporting). Default is None.

        dataset : str or None, optional
            Dataset name (for labeling/reporting and file naming). Default is None.

        selected_series : list[int] or None, optional
            Indices of the series (columns) to include in evaluation (if applicable). Default is None.

        plots : bool, optional
            Whether to generate downstream plots. Default is True.

        save_path : str, optional
            Directory where downstream results/plots are saved. Default is "./imputegap_assets/downstream".

        to_cache : bool, optional
            Cache the results of the downstream to prevent to run them again (default: True).

        use_cache: bool, optional
            If a cache exists for a specific use-case, use the cache and do not run the model again (default: False).

        verbose : bool, optional
            Verbosity flag for downstream logging. Default is True.

        Attributes
        ----------
        task, model, params, baseline, X_test, y_train, y_test, imputer_test, missing_test,
        do_testset_imputation, algorithm, dataset, selected_series, plots, save_path, verbose : see Parameters
        """

        self.task = task
        self.model = model
        self.params = params
        self.plots = plots
        self.save_path = save_path
        self.baseline = baseline
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.imputer_test = imputer_test
        self.missing_test = missing_test
        self.do_testset_imputation = do_testset_imputation
        self.algorithm = algorithm
        self.dataset = dataset
        self.selected_series = selected_series
        self.to_cache = to_cache
        self.use_cache = use_cache
        self.artifact = artifact
        self.referential = referential
        self.bypass_error = bypass_error
        self.v2 = v2


class Downstream:
    """
    A class to evaluate the performance of imputation algorithms using downstream analysis.

    This class provides tools to assess the quality of imputed time series data by analyzing
    the performance of downstream forecasting models. It computes metrics such as Mean Absolute
    Error (MAE) and Mean Squared Error (MSE) and visualizes the results for better interpretability.

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

    def _smape(self, reference, forecast):
        # Make them 1D lists of scalars
        reference = np.asarray(reference).ravel()
        forecast = np.asarray(forecast).ravel()
        n = len(forecast)
        if n <= 0:
            raise ValueError("No elements to compare (empty inputs or horizon <= 0).")

        mape_sum = 0.0

        for i in range(n):
            diff = abs(reference[i] - forecast[i])
            s = abs(reference[i]) + abs(forecast[i])

            if s < 1e-5 and diff < 1e-5:
                diff = 0.0
                s = 1.0
            elif s < 1e-5:
                diff = 2.0
                s = 1.0

            mape_sum += diff / (s / 2.0)

        return mape_sum / n

    def _format_metrics_table(self, metrics):
        """
        Format a flat metrics dictionary as a readable, space-separated text table (no borders).

        This helper converts the provided `metrics` dict into a simple aligned table with two columns:
        "Metric" and "Value". Column widths are computed automatically from the longest key/value strings.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric names to metric values (e.g., {"mse_groundtruth": 0.97, "smape_cdrec": 1.84}).
            Keys and values are converted to strings for display.

        Returns
        -------
        str
            A newline-joined string representing the formatted table, suitable for writing to a text file
            or printing to stdout.
        """
        # stringify + compute column widths
        rows = [(str(k), str(v)) for k, v in metrics.items()]
        k_w = max([len("Metric")] + [len(k) for k, _ in rows])
        v_w = max([len("Value")] + [len(v) for _, v in rows])

        gap = "   "  # spaces between columns
        total_w = k_w + len(gap) + v_w
        line = "-" * total_w

        out = []
        out.append(line)
        out.append(f"{'Metric'.ljust(k_w)}{gap}{'Value'.ljust(v_w)}")
        out.append(line)
        for k, v in rows:
            out.append(f"{k.ljust(k_w)}{gap}{v.ljust(v_w)}")
        out.append(line)
        return "\n".join(out)

    def classify(self, model, params, baseline, plots=True, save_path="./imputegap_assets/downstream", verbose=True):
        """
        Run downstream time-series classification on multiple data variants (ground-truth/input, recovered, and a
        baseline-imputed version) and report aggregated classification metrics.

        The method evaluates classification performance for three conditions:

        1) Ground-truth / raw data (self.input_data)

        2) Recovered data produced by the chosen recovery algorithm (self.recov_data)

        3) Baseline imputation applied to the incomplete data (self.incomp_data), e.g., mean-impute or another baseline

        A classifier is configured via `utils.config_classifier(model, params)`, trained on the (transposed) training
        matrix, and evaluated on the (transposed) test matrix.

        The following metrics are computed for each condition:

        - Accuracy: sklearn.metrics.accuracy_score
        - Recall  : sklearn.metrics.recall_score (macro-average by default)
        - F1      : sklearn.metrics.f1_score (macro-average by default)

        Metrics are stored in a dictionary using a consistent naming scheme:

        - "accuracy_groundtruth", "recall_groundtruth", "f1_groundtruth"
        - "accuracy_<algorithm>", "recall_<algorithm>", "f1_<algorithm>" where <algorithm> is `self.algorithm.lower()`
        - "accuracy_<baseline>",  "recall_<baseline>",  "f1_<baseline>"  where <baseline> is `baseline.lower()`

        The results are appended to a text file in `save_path` and (optionally) a plot is generated.

        Parameters
        ----------
        model : str
            Name/identifier of the classifier to use (passed to `utils.config_classifier`).

        params : dict
            Hyperparameters passed to the classifier configuration helper (`utils.config_classifier`).

        baseline : str or None
            Baseline imputation algorithm name.

        plots : bool, optional
            If True, generate and return an accuracy comparison plot.

        save_path : str, optional
            Directory where downstream results and plots are saved. Default is "./imputegap_assets/downstream".

        verbose : bool, optional
            If True, prints per-sample predictions and metric summaries for each condition. Default is True.

        Returns
        -------
        tuple[dict, object]
            metrics : dict
                Dictionary containing accuracy/recall/F1 for groundtruth, algorithm-imputed, and baseline-imputed data.
                Values are floats

            plt : object or None
                Plot handle/figure returned by `self._plot_classify_accuracy(...)` if `plots=True`, otherwise None.


        """
        from sklearn.metrics import accuracy_score, recall_score, f1_score

        acc_list, rec_list, f1_list = [], [], []
        plt, tag, loaded_cache = None, None, False
        average, zero_division = "macro", 0

        if not self.downstream.referential:
            cl_range = [1]
        else:
            cl_range = range(3)


        for x in cl_range:  # Iterate over recov_data, input_data, and mean_impute
            if x == 0:
                data = self.input_data  # raw data
                tag = "raw data"
                t = "raw"
            elif x == 1:
                data = self.recov_data  # algorithm imputed matrix on training set
                if self.downstream.do_testset_imputation:
                    self.downstream.X_test = self.downstream.imputer_test  # algorithm imputed matrix on testing set
                tag = "imputed data " + self.downstream.algorithm.upper()
                t = "impute"
            elif x == 2:
                from imputegap.recovery.imputation import Imputation
                if baseline is not None:
                    tr_base_imp = utils.config_impute_algorithm(incomp_data=self.incomp_data, algorithm=baseline)
                else:
                    baseline = "mean-impute"
                    tr_base_imp = Imputation.Statistics.MeanImpute(self.downstream.incomp_data)
                tr_base_imp.impute()
                data = tr_base_imp.recov_data
                if self.downstream.do_testset_imputation:
                    if baseline is not None:
                        ts_base_imp = utils.config_impute_algorithm(incomp_data=self.downstream.missing_test, algorithm=baseline)
                    else:
                        baseline = "mean-impute"
                        ts_base_imp = Imputation.Statistics.MeanImpute(self.downstream.missing_test)
                    ts_base_imp.impute()
                    self.downstream.X_test = ts_base_imp.recov_data
                tag = "baseline :" + str(baseline)
                t = "baseline"

            classifier = utils.config_classifier(model, params)

            data = data.T
            X_test = self.downstream.X_test.T

            if self.downstream.use_cache:
                name, dir_cache = utils.prepare_caching(dataset=self.downstream.artifact.dataset.lower(),
                                                        algorithm=self.downstream.artifact.algorithm.lower(),
                                                        pattern=self.downstream.artifact.pattern.lower(),
                                                        x=self.downstream.artifact.x,
                                                        contamination_by_class=self.downstream.artifact.contamination_by_class,
                                                        imputation_by_class=self.downstream.artifact.imputation_by_class,
                                                        fixed_rate=self.downstream.artifact.fixed_x)

                if x == 0: # raw_data
                    name = self.downstream.artifact.dataset.lower()
                print(model.lower())

                if model.lower() == "cif":
                    smodel = "cif"
                elif model.lower() == "patchtst":
                    smodel = "patchtst4"
                elif model.lower() == "chronos":
                    smodel = "chronos7d"
                elif model.lower() == "exp-smoothing":
                    smodel = "exp-smoothing_t2"
                else:
                    smodel = model

                opti = "_"
                if self.downstream.artifact.algorithm.lower() in ["svt"]:
                    opti = "_opti_"

                sname= f"_c_{smodel}_{t}_{name}{opti}PRED.txt"

                if self.downstream.v2:
                    sname = f"_c_{smodel}_{t}_{name}_v3_PRED.txt"

                cache_path_pred = os.path.join(dir_cache, "classifiers/", sname)

                if os.path.exists(cache_path_pred):
                    y_pred = utils.pred_caching_load(cache_path_pred)
                    loaded_cache = True
                    print(f"\t\tthe predictions have been loaded from the cache {y_pred.shape}: {sname}")
                else:
                    if self.downstream.bypass_error:
                        print(f"\t\t\t(CLASSIFICATION): by-pass the error... {self.downstream.artifact.dataset.lower()} - {self.downstream.artifact.algorithm.lower()} - {self.downstream.artifact.pattern.lower()} - {self.downstream.artifact.x} - {model=}")
                        al_name_acc = "accuracy_" + "imputer"  # self.algorithm.lower()
                        al_name_rec = "recall_" + "imputer"  # self.algorithm.lower()
                        al_name_f1 = "f1_" + "imputer"  # self.algorithm.lower()
                        base_name_acc = "accuracy_" + "meanimpute"  # baseline.lower()
                        base_name_rec = "recall_" + "meanimpute"  # baseline.lower()
                        base_name_f1 = "f1_" + "meanimpute"  # baseline.lower()
                        name_acc = "accuracy_groundtruth"
                        name_rec = "recall_groundtruth"
                        name_f1 = "f1_groundtruth"
                        return {name_acc: np.nan, al_name_acc: np.nan,base_name_acc: np.nan, name_rec: np.nan,al_name_rec: np.nan,base_name_rec: np.nan,name_f1: np.nan,al_name_f1: np.nan,base_name_f1: np.nan,}, None
                    else:
                        loaded_cache = False

            if not loaded_cache:
                if "lstm" in model:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU
                try:
                    classifier.fit(data, self.downstream.y_train)

                except ValueError as e:
                    msg = str(e)

                    if model == "signature":
                        data = np.asarray(data)
                        X_test = np.asarray(X_test)
                        if data.ndim == 2:
                            data = data[:, :, np.newaxis]
                        if X_test.ndim == 2:
                            X_test = X_test[:, :, np.newaxis]
                        classifier.fit(data, self.downstream.y_train)

                    # patch ts-fresh
                    elif model == "tsfresh" and "Too many bins for data range" in msg:
                        print("\t\t\ttsfresh failed on binned_entropy; retrying without binned_entropy...")
                        from sktime.classification.feature_based import TSFreshClassifier
                        from tsfresh.feature_extraction import EfficientFCParameters
                        fc_parameters = EfficientFCParameters()
                        fc_parameters.pop("binned_entropy", None)
                        retry_params = dict(params) if params is not None else {}
                        retry_params["default_fc_parameters"] = fc_parameters
                        classifier = TSFreshClassifier(**retry_params)
                        classifier.fit(data, self.downstream.y_train)
                    else:
                        print(f"\t\t\t(CLASS) FIT: {data.shape = } - {self.downstream.y_train.shape = } - {X_test.shape = }")
                        print(f"\t\t\t(CLASS) FIT: \t\t\t{msg}")

                try:
                    y_pred = classifier.predict(X_test)
                except ValueError as e:
                    msg = str(e)
                    print(f"\t\t\t(CLASS) PRED: {data.shape = } - {self.downstream.y_train.shape = } - {X_test.shape = }")
                    print(f"\t\t\t(CLASS) PRED: \t\t\t{msg}")

            acc = accuracy_score(self.downstream.y_test, y_pred)
            rec = recall_score(self.downstream.y_test, y_pred, average=average, zero_division=zero_division)
            f1 = f1_score(self.downstream.y_test, y_pred, average=average, zero_division=zero_division)

            acc_list.append(round(float(acc), 5))
            rec_list.append(round(float(rec), 5))
            f1_list.append(round(float(f1), 5))

            if verbose:
                print(f"\n\nresults of the {tag} ***************************************************")
                for i in range(0, len(y_pred)):
                    match = y_pred[i]==self.downstream.y_test[i]
                    print(f"series {i} > pred : {y_pred[i]} / actual : {self.downstream.y_test[i]}\t| {match=}\t|")
                print(f"\t accuracy : {acc}\t recall: {rec}\tf1: {f1}")



            if self.downstream.to_cache and not loaded_cache:
                name, dir_cache = utils.prepare_caching(dataset=self.downstream.artifact.dataset.lower(),
                                                        algorithm=self.downstream.artifact.algorithm.lower(),
                                                        pattern=self.downstream.artifact.pattern.lower(),
                                                        x=self.downstream.artifact.x,
                                                        contamination_by_class=self.downstream.artifact.contamination_by_class,
                                                        imputation_by_class=self.downstream.artifact.imputation_by_class,
                                                        fixed_rate=self.downstream.artifact.fixed_x)
                if x == 0:
                    name = self.downstream.artifact.dataset.lower()

                opti = "_"
                if self.downstream.artifact.algorithm.lower() in ["svt"]:
                    opti = "_opti_"

                if self.downstream.v2:
                    ext =  opti +"v3_PRED.txt"
                else:
                    ext =  opti +"PRED.txt"

                _, s_name = utils.classifiers_caching(y_pred=y_pred, model=model.lower(), t=t, name=name, family="_c", dir_cache=dir_cache, x=x, ext=ext)
                print(f"\t\tcaching stored for prediction: {s_name}")


        al_name_acc = "accuracy_" + "imputer" #self.algorithm.lower()
        al_name_rec = "recall_" + "imputer" #self.algorithm.lower()
        al_name_f1 = "f1_" + "imputer" #self.algorithm.lower()

        base_name_acc = "accuracy_" + "meanimpute" # baseline.lower()
        base_name_rec = "recall_" + "meanimpute" # baseline.lower()
        base_name_f1 = "f1_" + "meanimpute" # baseline.lower()

        name_acc = "accuracy_groundtruth"
        name_rec = "recall_groundtruth"
        name_f1 = "f1_groundtruth"

        if not self.downstream.referential :
            metrics = {name_acc: np.nan, al_name_acc: acc_list[0], base_name_acc: np.nan,
                       name_rec: np.nan, al_name_rec: rec_list[0], base_name_rec: np.nan,
                       name_f1: np.nan, al_name_f1: f1_list[0], base_name_f1: np.nan, }
        else:
            metrics = {name_acc: acc_list[0], al_name_acc: acc_list[1], base_name_acc: acc_list[2],
                       name_rec: rec_list[0], al_name_rec: rec_list[1], base_name_rec: rec_list[2],
                       name_f1: f1_list[0], al_name_f1: f1_list[1], base_name_f1: f1_list[2], }

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path + "/_classifier_" +self.downstream.dataset.lower() + "_results_downstream.txt")

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 50 + "\n")
            f.write(f"Results | dataset={self.downstream.dataset} | model={model} | alg={self.downstream.algorithm}\n")
            f.write("=" * 50 + "\n\n")

            # Nice table
            f.write(self._format_metrics_table(metrics))
            f.write("\n\n")


        if plots:
            plt = self._plot_classify_accuracy(metrics, model=model, title=f"Accuracy – {self.algorithm} vs {baseline}", save_path=save_path, display=self.downstream.plots, verbose=verbose)

        return metrics, plt


    def forecasting(self, model, params, baseline, selected_series, evaluator, plots, save_path="./imputegap_assets/downstream"):
        """
       Run downstream forecasting on multiple data variants (ground-truth/input, recovered, and a baseline-imputed version)
       and report aggregated forecasting metrics.

       The method evaluates forecasting performance for three conditions:

       1) Ground-truth / input data (self.input_data)
       2) Recovered data from the chosen recovery algorithm (self.recov_data)
       3) Baseline imputation applied to the incomplete data (self.incomp_data), e.g., mean-impute or another baseline

       Depending on the selected model, forecasting is performed using either:

       - sktime forecasters (when `model` is in `self.sktime_models`), or
       - Darts models otherwise.

       For each condition, predictions are produced for the selected series indices and the following metrics are computed:

       - MAE  : Mean Absolute Error
       - MSE  : Mean Squared Error
       - sMAPE: Symmetric Mean Absolute Percentage Error (in sktime computed via mean_absolute_percentage_error(..., symmetric=True))

       Metrics are aggregated across the selected series by averaging the per-series values.

       Parameters
       ----------
       model : str
           Name/identifier of the forecasting model to use. If `model` is contained in `self.sktime_models`,
           the sktime pipeline is used; otherwise the Darts pipeline is used.

       params : dict
           Hyperparameters passed to the forecaster configuration helper (`utils.config_forecaster`).

       baseline : str or None
           Baseline imputation algorithm name. If not None, the baseline is configured via
           `utils.config_impute_algorithm(self.incomp_data, algorithm=baseline)`. If None, defaults to "mean-impute"
           using `Imputation.Statistics.MeanImpute`.

       selected_series : list[int] or np.ndarray
           Indices of the time series (columns) to include in forecasting and evaluation.

       evaluator : str
           Label describing the evaluation type (passed through to plotting as `type=evaluator`).

       plots : bool
           If True, generate and return downstream plots via `self._plot_downstream(...)`.

       save_path : str, optional
           Directory used to save downstream assets/plots. Default is "./imputegap_assets/downstream".

       Returns
       -------
       tuple[dict, object]
           metrics : dict
               Dictionary of downstream metrics with keys:
               - "mse_groundtruth", "mae_groundtruth", "smape_groundtruth"
               - "mse_<algorithm>", "mae_<algorithm>", "smape_<algorithm>" where <algorithm> is `self.algorithm.lower()`
               - "mse_<baseline>",  "mae_<baseline>",  "smape_<baseline>"  where <baseline> is `baseline.lower()`
               Values are floats (rounded in this implementation).

           plt : object or None
               Plot handle/figure returned by `self._plot_downstream(...)` if `plots=True`, otherwise None.
       """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

        y_train_all, y_test_all, y_pred_all = [], [], []
        mae, mse, smape = [], [], []
        plt, tag, loaded_cache = None, None, False
        raw_data = self.input_data
        recovery = self.recov_data

        save_sele = selected_series[0]

        if not self.downstream.referential:
            cl_range = [1]
        else:
            cl_range = range(3)

        for x in cl_range:  # Iterate over recov_data, input_data, and mean_impute
            selected_series[0] = save_sele
            if x == 0:
                data = self.input_data
                tag = "raw data"
                t = "raw"
                selected_series[0] = 0
            elif x == 1:
                data = self.recov_data
                tag = "imputed data " + self.downstream.artifact.dataset.lower()
                t = "impute"
            elif x == 2:
                from imputegap.recovery.imputation import Imputation

                if baseline is not None:
                    impt = utils.config_impute_algorithm(self.incomp_data, algorithm=baseline)
                    impt.impute()
                    data = impt.recov_data
                else:
                    baseline = "mean-impute"
                    mean_impute = Imputation.Statistics.MeanImpute(self.incomp_data).impute()
                    data = mean_impute.recov_data
                tag = "baseline :" + str(baseline)
                t = "baseline"

            if self.downstream.y_train is None:
                if self.downstream.artifact.horizon > 0:
                    train_len = data.shape[0] - self.downstream.artifact.horizon
                else:
                    train_len = data.shape[0] - int(data.shape[0] * self.downstream.artifact.horizon)

                print(f"\t\t\tconverting the dataset into forecasting task... horizon:{self.downstream.artifact.horizon}, train_len:{train_len}")

                y_train = data[:train_len, :]
                y_test = data[train_len:, :]
            else:
                y_train = data
                y_test = self.downstream.y_train

            if self.downstream.use_cache:
                name, dir_cache = utils.prepare_caching(dataset=self.downstream.artifact.dataset.lower(),
                                                        algorithm=self.downstream.artifact.algorithm.lower(),
                                                        pattern=self.downstream.artifact.pattern.lower(),
                                                        x=self.downstream.artifact.x,
                                                        contamination_by_class=None,
                                                        imputation_by_class=self.downstream.artifact.horizon,
                                                        fixed_rate=self.downstream.artifact.fixed_x)

                if x == 0: # raw_data
                    name = self.downstream.artifact.dataset.lower() + "_" + str(self.downstream.artifact.horizon)

                if model.lower() == "cif":
                    smodel = "cif2"
                elif model.lower() == "patchtst":
                    smodel = "patchtst4"
                elif model.lower() == "chronos":
                    smodel = "chronos7d"
                elif model.lower() == "exp-smoothing":
                    smodel = "exp-smoothing_t"
                else:
                    smodel = model

                opti = "_"
                if self.downstream.artifact.algorithm.lower() in ["dynammo", "saits"]:
                    opti = "_opti_"

                s_selection_idx = str(selected_series[0])
                sname= f"_f_{smodel}_{t}_{name}_s{s_selection_idx}{opti}PRED.txt"
                cache_path_pred = os.path.join(dir_cache, "forecasters/", sname)

                if os.path.exists(cache_path_pred):
                    y_pred = utils.pred_caching_load(cache_path_pred)
                    y_pred = np.asarray(y_pred, dtype=float)
                    loaded_cache = True
                    print(f"\t\tthe predictions have been loaded from the cache {y_pred.shape}: {sname}")
                else:
                    if self.downstream.bypass_error:
                        al_name = "mse_" + "imputer"  # self.algorithm.lower()
                        al_name_s = "smape_" + "imputer"  # self.algorithm.lower()
                        al_name_m = "mae_" + "imputer"  # self.algorithm.lower()
                        al_name_c = "mse_" + "baseline"  # baseline.lower()
                        al_name_cs = "smape_" + "baseline"  # baseline.lower()
                        al_name_mcs = "mae_" + "baseline"  # baseline.lower()
                        return {"mse_groundtruth": np.nan, al_name: np.nan, al_name_c: np.nan, "mae_groundtruth": np.nan, al_name_m: np.nan, al_name_mcs: np.nan, "smape_groundtruth": np.nan, al_name_s: np.nan, al_name_cs: np.nan}, None
                    loaded_cache = False

            forecaster = utils.config_forecaster(model, params, y_test.shape[0], s_exception=self.downstream.s_exception)

            if self.verbose:
                print(f"\n\t\t{tag}: call of the forecaster:{type(forecaster)}\n")

            if not loaded_cache:
                y_pred = np.zeros_like(y_test, dtype=float)

            if model in self.sktime_models:
                # --- SKTIME APPROACH ---
                mae_list, mse_list, smape_list = [], [], []
                fh = np.arange(1, y_test.shape[0] + 1)  # Forecast horizon

                for series_idx in selected_series:
                    if len(selected_series) == 1:
                        pred_to_call = 0
                    else:
                        pred_to_call = series_idx

                    series_train = y_train[:, series_idx]

                    if not loaded_cache:
                        if model == "ltsf" or model == "rnn" or model == "moment":
                            forecaster.fit(series_train, fh=ForecastingHorizon(fh))
                            series_pred = forecaster.predict()
                        else:
                            forecaster.fit(series_train)
                            series_pred = forecaster.predict(fh=fh)

                        result_y_test = y_test.copy()
                        result_y_pred = series_pred.copy()

                        y_pred = series_pred[:, pred_to_call]
                        y_test = y_test[:, series_idx]

                    else:
                        result_y_test = y_test.copy()
                        result_y_pred = y_test.copy()
                        result_y_pred[:, series_idx] = y_pred
                        y_test = y_test[:, series_idx]

                    # Compute metrics using sktime
                    mae_list.append(mean_absolute_error(y_test, y_pred))
                    mse_list.append(mean_squared_error(y_test, y_pred))
                    smape_list.append(mean_absolute_percentage_error(y_test, y_pred, symmetric=True))  # Compute SMAPE

                    to_store = y_pred

            else:
                # --- DARTS APPROACH ---
                from darts import TimeSeries
                mae_list, mse_list, smape_list = [], [], []
                fh = y_test.shape[0]

                for series_idx in selected_series:
                    if len(selected_series) == 1:
                        pred_to_call = 0
                    else:
                        pred_to_call = series_idx

                    train_ts = TimeSeries.from_values(y_train[:, series_idx])
                    test_ts = TimeSeries.from_values(y_test[:, series_idx])

                    if not loaded_cache:
                        forecaster.fit(train_ts)
                        pred_ts = forecaster.predict(n=fh)

                        result_y_test = y_test.copy()
                        result_y_pred = pred_ts.values().copy()

                        tm1 = test_ts.values()
                        tm2 = pred_ts.values()

                        test_ts = tm1[:, pred_to_call]
                        pred_ts = tm2[:, pred_to_call]

                    else:
                        pred_ts = y_pred
                        result_y_test = y_test.copy()
                        result_y_pred = y_test.copy()
                        result_y_pred[:, pred_to_call] = pred_ts
                        tm1 = test_ts.values()
                        test_ts = tm1[:, pred_to_call]


                    mae_list.append(mean_absolute_error(test_ts, pred_ts))
                    mse_list.append(mean_squared_error(test_ts, pred_ts))
                    smape_list.append(mean_absolute_percentage_error(test_ts, pred_ts, symmetric=True))  # Compute SMAPE
                    to_store = pred_ts

            mae.append(round(float(np.mean(mae_list)), 5))
            mse.append(round(float(np.mean(mse_list)), 5))
            smape.append(round(float(np.mean(smape_list)), 5))

            # Store for plotting
            y_train_all.append(y_train)
            y_test_all.append(result_y_test)
            y_pred_all.append(result_y_pred)

            if self.downstream.to_cache and not loaded_cache:
                name, dir_cache = utils.prepare_caching(dataset=self.downstream.artifact.dataset.lower(),
                                                        algorithm=self.downstream.artifact.algorithm.lower(),
                                                        pattern=self.downstream.artifact.pattern.lower(),
                                                        x=self.downstream.artifact.x,
                                                        contamination_by_class=None,
                                                        imputation_by_class=self.downstream.artifact.horizon,
                                                        fixed_rate=self.downstream.artifact.fixed_x)

                if x == 0:
                    name = self.downstream.artifact.dataset.lower() + "_" + str(self.downstream.artifact.horizon)

                opti = "_"
                #if self.downstream.artifact.algorithm.lower() in ["dynammo", "saits"]:
                #    opti = "_opti_"

                ext = "_s"+str(selected_series[0]) + opti + "PRED.txt"

                _, s_name = utils.classifiers_caching(y_pred=to_store, model=model.lower(), t=t, name=name, dir_cache=dir_cache, family="_f", ext=ext)
                print(f"\t\tcaching stored for prediction: {s_name}")

        # Save metrics in a dictionary
        al_name = "mse_" + "imputer" # self.algorithm.lower()
        al_name_s = "smape_" + "imputer" # self.algorithm.lower()
        al_name_m = "mae_" + "imputer" # self.algorithm.lower()
        al_name_c = "mse_" + "baseline" # baseline.lower()
        al_name_cs = "smape_" + "baseline" # baseline.lower()
        al_name_mcs = "mae_" + "baseline" # baseline.lower()

        if not self.downstream.referential :
            metrics = {"mse_groundtruth": np.nan, al_name: mse[0], al_name_c: np.nan, "mae_groundtruth": np.nan, al_name_m: mae[0], al_name_mcs: np.nan, "smape_groundtruth": np.nan, al_name_s: smape[0], al_name_cs: np.nan}

            #mse.append(-1)
            #mae.append(-1)
            #smape.append(-1)
        else:
            metrics = {"mse_groundtruth": mse[0], al_name: mse[1], al_name_c: mse[2], "mae_groundtruth": mae[0], al_name_m: mae[1], al_name_mcs: mae[2], "smape_groundtruth": smape[0], al_name_s: smape[1], al_name_cs: smape[2]}

        print(f"\n\n>>>>>>>>>>>>>> {selected_series= } - {pred_to_call=}\n\n")

        if plots:
            if not self.downstream.single_pipeline:
                plt = self._plot_downstream(y_train=y_train_all, y_test=y_test_all, y_pred=y_pred_all, incomp_data=self.incomp_data, algorithm=self.algorithm, comparison=baseline, model=model, type=evaluator, save_path=save_path, display=self.downstream.plots, verbose=self.verbose, pred_to_call=pred_to_call)
            else:
                plt = self._plot_single_downstream(y_train=y_train_all[1] if self.downstream.referential else y_train_all[0], y_test=y_test_all[1] if self.downstream.referential else y_test_all[0], y_pred=y_pred_all[1] if self.downstream.referential else y_pred_all[0], incomp_data=self.incomp_data, ground_truth=raw_data, recov_data=recovery, series_index=selected_series[0], algorithm=self.algorithm, model=model, type=evaluator, save_path=save_path, display=self.downstream.plots, verbose=self.verbose, pred_to_call=pred_to_call)

        return metrics, plt


    def downstream_analysis(self):
        """
        Compute a set of evaluation metrics with a downstream analysis

        ImputeGAP downstream models for forecasting : ['arima', 'bats', 'croston', 'deepar', 'ets', 'exp-smoothing',
        'hw-add', 'lightgbm', 'lstm', 'naive', 'nbeats', 'prophet', 'sf-arima', 'theta',
        'transformer', 'unobs', 'xgboost']

        Returns
        -------
        dict or None
            Metrics from the downstream analysis or None if no valid evaluator is provided.
        """

        if self.downstream.baseline is None:
            baseline = "mean-impute"

        model = self.downstream.model.lower()
        evaluator = self.downstream.task.lower()


        if evaluator in ["forecast", "forecaster", "forecasting"]:
            evaluator = "forecaster"
        elif evaluator in ["classification", "classifier", "classify"]:
            evaluator = "classifier"
            self.downstream.selected_series = -1
        else:
            print("\tEvaluator found... list possible : 'forecaster', 'classifier'" + "*" * 20 + "\n")
            return None

        if self.downstream.params is None:
            if self.verbose:
                print("\n\n(DOWNSTREAM) Default parameters of the downstream model loaded.")
            loader = str(evaluator) + "-" + str(model)
            params = utils.load_parameters(query="default", task=evaluator, algorithm=loader)
        else:
            params = self.downstream.params

        if evaluator == "forecaster" and self.downstream.season is not None:
            for k in ("sp", "lags"):
                if k in params:
                    params[k] = self.downstream.season

        if self.downstream.selected_series is None:
            nan_cols = np.isnan(self.incomp_data).any(axis=0)
            series_indices = int(np.argmax(nan_cols)) if nan_cols.any() else -1  # take the 1st contaminated series
            selected_series = [series_indices]
        elif self.downstream.selected_series == -1:
            selected_series = range(self.input_data.shape[1])

        if self.verbose and evaluator == "forecaster":
            print(f"\n(DOWNSTREAM) Analysis launched !\n\ttask: {evaluator}\n\tmodel: {model}\n\tparams: {params}\n\tbase algorithm: {str(self.algorithm).lower()}\n\treference algorithm: {str(self.downstream.baseline).lower()}\n\tselected series: {selected_series}\n\thorizon: {self.downstream.artifact.horizon}\n")
        elif self.verbose and evaluator == "classifier":
            print(f"\n(DOWNSTREAM) Analysis launched !\n\ttask: {evaluator}\n\tmodel: {model}\n\tparams: {params}\n\tbase algorithm: {str(self.algorithm).lower()}\n\treference algorithm: {str(self.downstream.baseline).lower()}\n\tselected series: {selected_series}\n")

        if evaluator == "forecaster":
            return self.forecasting(model=model, params=params, baseline=self.downstream.baseline, selected_series=selected_series, evaluator=evaluator, plots=self.downstream.plots, save_path=self.downstream.save_path)
        elif evaluator == "classifier":
            return self.classify(model=model, params=params, baseline=self.downstream.baseline, plots=self.downstream.plots, save_path=self.downstream.save_path, verbose=self.verbose)


    @staticmethod
    def _plot_downstream(y_train, y_test, y_pred, incomp_data, algorithm, comparison, model=None, type=None, title="", max_series=1, save_path="./imputegap_assets/downstream", display=True, verbose=True, pred_to_call=0):
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
                pred_series = s_y_pred[:, pred_to_call]

                # Combine training and testing data for visualization
                full_series = np.concatenate([train_series, test_series])

                #print(f"{row_idx=}, {full_series=}")

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
            if verbose:
                print("\nplots saved in: ", file_path)
            if display:
                plt.show()

        return plt

    @staticmethod
    def _plot_single_downstream(y_train, y_test, y_pred, incomp_data,ground_truth, recov_data, algorithm, series_index, model=None, type=None, title="", save_path="./imputegap_assets/downstream", display=True, verbose=True, pred_to_call=0):
        """
        Plot ground truth, imputed values, and predictions for the first contaminated series.

        Parameters
        ----------
        y_train : np.ndarray
            Training data array of shape (train_len, n_series).
        y_test : np.ndarray
            Testing data array of shape (test_len, n_series).
        y_pred : np.ndarray
            Forecasted data array of shape (test_len, n_predictions).
        incomp_data : np.ndarray
            Incomplete data array used to identify contaminated series.
        ground_truth : np.ndarray
            Ground truth data array used to identify contaminated series.
        recov_data : np.ndarray
            Imputed data array containing recovered values.
        algorithm : str
            Name of the imputation algorithm.
        series_index: int
            Index of the series to plot.
        model : str
            Name of the forecasting model.
        type : str
            Name of the current type.
        title : str
            Title of the plot.
        save_path : str
            Directory where plots are saved.
        display : bool
            Whether to display the plots.
        verbose : bool
            Whether to print the saved file path.
        pred_to_call : int
            Index of the prediction column to display.

        Returns
        -------
        plt
            Return the plots object.
        """

        fig, ax = plt.subplots(figsize=(24, 5))
        fig.canvas.manager.set_window_title("downstream evaluation")
        fig.suptitle(title, fontsize=16)

        # Select the first contaminated series
        valid_indices = [i for i in range(incomp_data.shape[1]) if np.isnan(incomp_data[:, i]).any()]

        if not valid_indices:
            plt.close(fig)
            raise ValueError("No contaminated series found.")

        series_idx = series_index

        # Extract recovered training data and predictions
        train_series = y_train[:, series_idx]
        test_series = y_test[:, series_idx]
        pred_series = y_pred[:, pred_to_call]
        recov_series = recov_data[:, series_idx]

        # Extract original ground truth, independently of recovered training data
        train_len = len(train_series)
        test_len = len(test_series)

        ground_train = ground_truth[:train_len, series_idx]

        # Use original testing values when available; otherwise use y_test
        if ground_truth.shape[0] >= train_len + test_len:
            ground_test = ground_truth[train_len:train_len + test_len, series_idx]
        else:
            ground_test = test_series

        # Identify missing values in the training portion
        missing_mask = np.isnan(incomp_data[:train_len, series_idx])

        # Full timeline
        full_series = np.concatenate([ground_train, ground_test])

        # Plot original ground truth over the entire timeline
        ground_normal = ground_train.copy()
        ground_normal[missing_mask] = np.nan

        ax.plot(range(train_len), ground_normal, label="ground truth", color="gray", linewidth=1.5, alpha=0.9, zorder=2)

        # Ground truth inside contaminated regions (more transparent)
        ground_missing = ground_train.copy()
        ground_missing[~missing_mask] = np.nan

        ax.plot(range(train_len), ground_missing, label="_nolegend_", color="gray", linewidth=1.5, alpha=0.5, zorder=2)
        ax.plot(range(train_len, train_len + test_len), ground_test, label="_nolegend_", color="gray", linewidth=0.8, alpha=0.5, zorder=1)

        # Highlight recovered values only where contamination occurred
        imputed_series = ground_train.copy()
        imputed_series[missing_mask] = recov_series[:train_len][missing_mask]

        plot_mask = missing_mask.copy()
        plot_mask[1:] |= missing_mask[:-1]
        plot_mask[:-1] |= missing_mask[1:]
        imputed_series[~plot_mask] = np.nan

        # Light red background only over contaminated timestamp zones
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0:
            groups = np.split(missing_indices, np.where(np.diff(missing_indices) > 1)[0] + 1)
            for group in groups:
                ax.axvspan(group[0] - 0.5, group[-1] + 0.5, color="red", alpha=0.08, zorder=0)

        ax.plot(range(len(train_series)), imputed_series, label=f"{algorithm.lower()} imputation", color="red", linestyle="-", linewidth=1.8, alpha=0.9, zorder=5)

        # Plot forecasted data on top of testing data
        label = f"{type or ''} {model or ''}".strip()
        ax.plot(range(len(train_series), len(full_series)), pred_series, label=label, linestyle="-", marker=None, color="green", zorder=4)

        # Add a vertical line at the split point
        ax.axvline(x=len(train_series), color="orange", linestyle="--")

        # Add labels, title, and grid
        ax.set_title(f"{algorithm.lower()} imputation, series_{series_idx + 1}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left", fontsize=7, frameon=True, fancybox=True, framealpha=0.8)
        ax.grid()

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(top=0.88)

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            file_path = os.path.join(save_path, current_time + "_" + str(type) + "_" + str(model) + "_downstream.jpg")

            plt.savefig(file_path, bbox_inches="tight")

            if verbose:
                print("\nplots saved in: ", file_path)

            if display:
                plt.show()

        return plt

    def _plot_classify_accuracy(self, metrics: dict, model="classifier", title="Classification accuracy", save_path="./imputegap_assets/downstream", display=True, verbose=True):
        """
        Plot a bar chart of classification accuracies from a metrics dictionary.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics to plot. Keys are used as x-axis labels and values are plotted as bar heights.
            In the typical use-case, this contains accuracy values for groundtruth, algorithm-imputed, and baseline-imputed
            settings (but any flat dict of numeric values is accepted).

        model : str, optional
            Model identifier used only for file naming when saving the plot. Default is "classifier".

        title : str, optional
            Title displayed at the top of the plot. Default is "Classification accuracy".

        save_path : str or None, optional
            Directory where the plot image is saved. If falsy/None, the plot is not saved. Default is
            "./imputegap_assets/downstream".

        display : bool, optional
            If True, display the plot window (plt.show()) after saving. Default is True.

        verbose : bool, optional
            If True, prints the path where the plot was saved. Default is True.

        Returns
        -------
        matplotlib.pyplot
            The matplotlib pyplot module/handle used to build the plot (compatible with your current pipeline).
        """
        keys = metrics.keys()
        vals = [metrics[k] for k in keys]

        scale_factor = 0.7
        x_size_screen = (1920 / 100) * scale_factor
        y_size_screen = (1080 / 100) * scale_factor

        plt.figure(figsize=(x_size_screen, y_size_screen))
        plt.bar(keys, vals)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.xticks(rotation=15, ha="right")

        # value labels
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")
            file_path = os.path.join(save_path + "/" + current_time + "_classifier_" + model + "_downstream.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            if verbose:
                print("\nplots saved in: ", file_path)
            if display:
                plt.show()
        plt.tight_layout()

        return plt
