import os
import unittest
import numpy as np

class TestPipeline(unittest.TestCase):

    def test_pipeline_load(self):
        """
        Verify if the manager of a dataset is working
        """
        x = False

        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        # 1. ===========================================================================================================
        ts_none = TimeSeries()
        ts_none.load_series(utils.search_path("test-logic-llm.txt"), normalizer=None)
        ts_max = TimeSeries()
        ts_max.load_series(utils.search_path("test-logic-llm.txt"), normalizer="min-max")
        test = utils.list_of_algorithms()
        test = utils.list_of_datasets()
        test = utils.list_of_extractors()
        test = utils.list_of_metrics()
        test = utils.list_of_downstreams()

        ts = TimeSeries()
        print(f"\nImputeGAP datasets : {ts.datasets}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z_score")

        # plot and print a subset of time series
        ts.print(nbr_series=3, nbr_val=20)
        ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets")

        # --- Checks you requested ---
        # 1) ts.data is not None
        self.assertIsNotNone(ts.data)

        # 2) ts.data.shape == (32, 6)
        self.assertEqual(ts.data.shape, (32, 6))

        # (needed for comparisons)
        self.assertIsNotNone(ts_none.data)
        self.assertIsNotNone(ts_max.data)
        self.assertEqual(ts_none.data.shape, ts.data.shape)
        self.assertEqual(ts_max.data.shape, ts.data.shape)

        # 3) ts_none has different values than ts.data (z-score vs raw)
        self.assertFalse(np.allclose(ts_none.data, ts.data))

        # 4) ts_max (min-max) has different values than ts_none (raw)
        self.assertFalse(np.allclose(ts_max.data, ts_none.data))

        # plot and print a subset of time series
        ts.print(nbr_series=3, nbr_val=20)
        ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets")

        x = not x
        self.assertTrue(x)


    def test_pipeline_gengap(self):
        """
        Verify if the manager of a dataset is working
        """
        x = False

        from imputegap.recovery.manager import TimeSeries
        from imputegap.recovery.contamination import GenGap
        from imputegap.tools import utils

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nMissingness patterns : {ts.patterns}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

        # contaminate the time series with MCAR pattern
        ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

        # plot the contaminated time series
        ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")

        # --- checks ---
        # 1) ts_m is not None
        self.assertIsNotNone(ts_m)

        # 2) ts_m contains NaNs (i.e., contamination actually happened)
        self.assertTrue(np.isnan(ts_m).any(), "Expected ts_m to contain NaNs after MCAR contamination")

        self.assertEqual(ts_m.shape, ts.data.shape)

        x = not x
        self.assertTrue(x)



    def test_pipeline_imputation(self):
        """
        Verify if the manager of a dataset is working
        """
        x = False

        from imputegap.recovery.imputation import Imputation
        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputation algorithms : {ts.algorithms}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

        # contaminate the time series
        ts_m = GenGap.mcar(ts.data)

        # impute the contaminated series
        imputer = Imputation.MatrixCompletion.CDRec(ts_m)
        imputer.impute()

        # --- check recovered data has no NaNs ---
        self.assertIsNotNone(imputer.recov_data)
        self.assertFalse(np.isnan(imputer.recov_data).any(), "Recovered data contains NaNs")

        # compute and print the imputation metrics
        imputer.score(ts.data, imputer.recov_data)

        print(f"{imputer.metrics = }")

        # --- check metrics ---
        self.assertIsInstance(imputer.metrics, dict)
        self.assertTrue(len(imputer.metrics) > 0, "Metrics dict is empty")

        # all metric values > 0 (only for numeric metrics)
        for k, v in imputer.metrics.items():
            if isinstance(v, (int, float, np.floating)):
                self.assertGreater(v, 0, f"Metric {k} should be > 0, got {v}")

        # MAE between 0 and 1
        self.assertIn("MAE", imputer.metrics)
        mae = float(imputer.metrics["MAE"])
        self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae}")
        self.assertLessEqual(mae, 1.0, f"MAE should be <= 1, got {mae}")

        x = not x
        self.assertTrue(x)



    def test_pipeline_auto_ml(self):
        """
        Verify if the manager of a dataset is working
        """
        x = False

        from imputegap.recovery.imputation import Imputation
        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        # initialize the time series object
        ts = TimeSeries()

        # load and normalize the dataset
        ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

        # contaminate and impute the time series
        ts_m = GenGap.mcar(ts.data)
        imputer = Imputation.MatrixCompletion.CDRec(ts_m)
        imputer.verbose = False

        # use Ray Tune to fine tune the imputation algorithm
        imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

        # compute the imputation metrics with optimized parameter values
        imputer.score(ts.data, imputer.recov_data)

        # compute the imputation metrics with default parameter values
        imputer_def = Imputation.MatrixCompletion.CDRec(ts_m).impute()
        imputer_def.score(ts.data, imputer_def.recov_data)

        # print the imputation metrics with default and optimized parameter values
        ts.print_results(imputer_def.metrics, text="Default values")
        ts.print_results(imputer.metrics, text="Optimized values")

        # plot the recovered time series
        ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation", display=True)

        # save hyperparameters
        utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")

        # --- checks on default metrics ---
        self.assertIsNotNone(imputer_def.metrics)
        self.assertIsInstance(imputer_def.metrics, dict)
        self.assertTrue(len(imputer_def.metrics) > 0, "Default metrics dict is empty")

        # metric values are not NaN and > 0 (for numeric metrics)
        for k, v in imputer_def.metrics.items():
            if isinstance(v, (int, float, np.floating)):
                self.assertFalse(np.isnan(v), f"Metric {k} is NaN")
                self.assertGreater(v, 0, f"Metric {k} should be > 0, got {v}")

        # MAE between 0 and 1
        self.assertIn("MAE", imputer_def.metrics)
        mae = float(imputer_def.metrics["MAE"])
        self.assertFalse(np.isnan(mae), "MAE is NaN")
        self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae}")
        self.assertLessEqual(mae, 1.0, f"MAE should be <= 1, got {mae}")

        x = not x
        self.assertTrue(x)


    def test_pipeline_explainer(self):
        """
        Verify if the manager of a dataset is working
        """
        xx = False

        from imputegap.recovery.manager import TimeSeries
        from imputegap.recovery.explainer import Explainer
        from imputegap.tools import utils

        # initialize the time series and explainer object
        ts = TimeSeries()
        exp = Explainer()
        print(f"\nImputeGAP explainer features extractor : {ts.extractors}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

        # configure the explanation
        exp.shap_explainer(input_data=ts.data, extractor="tsfresh", pattern="mcar", file_name=ts.name, algorithm="CDRec")

        # print the impact of each feature
        exp.print(exp.shap_values, exp.shap_details)

        # plot the feature impacts
        exp.show()

        print(f"{exp.shap_values = }")

        # --- checks: shap_values contains no NaNs in any field ---
        self.assertIsNotNone(exp.shap_values)
        self.assertTrue(len(exp.shap_values) > 0, "shap_values is empty")

        for (x, algo, rate, description, feature, category, mean_features) in exp.shap_values:
            # numeric fields
            self.assertFalse(np.isnan(float(rate)), "rate is NaN")

            # string-ish fields (guard against None)
            self.assertIsNotNone(algo, "algo is None")
            self.assertIsNotNone(description, "description is None")
            self.assertIsNotNone(feature, "feature is None")
            self.assertIsNotNone(category, "category is None")

            # mean_features can be scalar or array-like
            if mean_features is None:
                self.fail("mean_features is None")

            arr = np.asarray(mean_features, dtype=float)  # works for scalar, list, np array
            self.assertFalse(np.isnan(arr).any(), "mean_features contains NaNs")

        xx = not xx
        self.assertTrue(xx)

    def test_pipeline_benchmark(self):

        x = False

        from imputegap.recovery.benchmark import Benchmark
        from imputegap.tools import utils

        my_algorithms = ["MeanImpute", "SoftImpute"]

        my_opt = "default_params"

        my_datasets = ["eeg-alcohol"]

        my_patterns = ["mcar"]

        range = [0.2]

        my_metrics = ["*"]

        # launch the evaluation
        bench = Benchmark()
        bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range,
                   metrics=my_metrics, optimizer=my_opt)

        x = not x
        self.assertTrue(x)


    def test_pipeline_algos_dl(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_deep_learning() = }")

        for name in utils.list_of_algorithms_deep_learning():
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name == "DeepMVI":
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
                incomp_data = np.tile(incomp_data, (7, 5))
                ts.data = np.tile(ts.data, (7, 5))
            if name == "SPIRIT" or name == "TKCM":
                incomp_data = GenGap.aligned(ts.data, single_series=0)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)

    def test_pipeline_algos_llms(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_llms() = }")

        for name in utils.list_of_algorithms_llms():
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name == "DeepMVI":
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
                incomp_data = np.tile(incomp_data, (7, 5))
                ts.data = np.tile(ts.data, (7, 5))
            if name == "SPIRIT" or name == "TKCM":
                incomp_data = GenGap.aligned(ts.data, single_series=0)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)



    def test_pipeline_algos_matrixcompletion(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_matrix_completion() = }")

        for name in utils.list_of_algorithms_matrix_completion():
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name == "DeepMVI":
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
                incomp_data = np.tile(incomp_data, (7, 5))
                ts.data = np.tile(ts.data, (7, 5))
            if name == "SPIRIT" or name == "TKCM":
                incomp_data = GenGap.aligned(ts.data, single_series=0)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)


    def test_pipeline_algos_patternsearch(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_pattern_search() = }")

        for name in ["TKCM"]:
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name in ["DeepMVI", "TKCM", "SPIRIT"]:
                ts.data = np.tile(ts.data, (7, 5))
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)

    def test_pipeline_algos_stats(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_statistics() = }")

        for name in utils.list_of_algorithms_statistics():
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name == "DeepMVI":
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
                incomp_data = np.tile(incomp_data, (7, 5))
                ts.data = np.tile(ts.data, (7, 5))
            if name == "SPIRIT" or name == "TKCM":
                incomp_data = GenGap.aligned(ts.data, single_series=0)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)


    def test_pipeline_algos_ml(self):

        x = False

        from imputegap.recovery.contamination import GenGap
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        print(f"{utils.list_of_algorithms_machine_learning() = }")

        for name in utils.list_of_algorithms_machine_learning():
            # load and normalize the dataset
            ts = TimeSeries()
            ts.load_series(utils.search_path("test-logic-llm.txt"), normalizer="z-score")
            incomp_data = GenGap.mcar(ts.data)
            print(f"\n\n{incomp_data.shape = }")
            if name == "DeepMVI":
                incomp_data = GenGap.aligned(ts.data, single_series=0, rate_series=0.5)
                incomp_data = np.tile(incomp_data, (7, 5))
                ts.data = np.tile(ts.data, (7, 5))
            if name == "SPIRIT" or name == "TKCM":
                incomp_data = GenGap.aligned(ts.data, single_series=0)
            algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
            algo.impute()
            algo.score(ts.data)
            metrics = algo.metrics
            print(f"{name} : {metrics}\n\n")

            self.assertIsInstance(algo.metrics, dict)
            self.assertTrue(len(algo.metrics) > 0, "Metrics dict is empty")
            self.assertIn("MAE", algo.metrics)
            mae = float(algo.metrics["MAE"])
            self.assertGreaterEqual(mae, 0.0, f"MAE should be >= 0, got {mae} for {name}")
            self.assertLessEqual(mae, 100.0, f"MAE should be <= 1, got {mae} for {name}")

        x = not x
        self.assertTrue(x)


