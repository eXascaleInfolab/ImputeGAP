import math
import unittest
from imputegap.recovery.benchmark import Benchmark

class TestBenchmarking(unittest.TestCase):

    def test_benchmarking(self):
        """
        the goal is to test if only the simple imputation benchmarking has the expected outcome
        """
        bench = Benchmark()
        expected_datasets = ["test-logic-llm.txt"]

        opti_bayesian = {"optimizer": "bayesian", "options": {"n_calls": 2, "n_random_starts": 50, "acq_func": "gp_hedge", "metrics": "RMSE"}}
        optimizer = "default_params"

        algorithms_full = ["knn", "cdrec", "stmvl", "iim", "mrnn"]
        patterns_small = ["mcar"]

        x_axis = [0.1, 0.2, 0.8]

        bench.eval(datasets=expected_datasets, optimizer=optimizer, algorithms=algorithms_full, patterns=patterns_small, x_axis=x_axis, runs=-1)
        results_benchmarking = bench.list_results
        results_benchmarking = results_benchmarking[0]
        expected_datasets = ["testlogicllm.txt"]

        print(f"{bench.list_results = }")

        # Check that all datasets exist
        actual_datasets = list(results_benchmarking.keys())


        self.assertCountEqual(actual_datasets, expected_datasets, f"Missing datasets. Expected: {expected_datasets}, Found: {actual_datasets}" )

        # For each dataset, validate structure and values
        for dataset, dataset_data in results_benchmarking.items():

            # Check that patterns exist (e.g., 'mcar')
            self.assertIn(
                "mcar", dataset_data,
                f"Dataset '{dataset}' is missing 'mcar' scenario."
            )
            scenario_data = dataset_data["mcar"]

            # Check that all algorithms exist
            expected_algorithms = {"stmvl", "cdrec", "iim", "knn", "mrnn"}
            actual_algorithms = set(scenario_data.keys())
            self.assertTrue(
                expected_algorithms.issubset(actual_algorithms),
                f"Missing algorithms in dataset '{dataset}'. Expected: {expected_algorithms}, Found: {actual_algorithms}"
            )

            # Check that each algorithm contains the expected keys
            expected_keys = {"0.1", "0.2", "0.8"}
            for algorithm, algorithm_data in scenario_data.items():
                for key in expected_keys:
                    self.assertIn(key, algorithm_data.get("default_params", {}), f"Algorithm '{algorithm}' in dataset '{dataset}' is missing key '{key}'." )

                    sub_data = algorithm_data["default_params"].get(key, {})
                    for score_key, score_value in sub_data.get("scores", {}).items():
                        self.assertIsInstance( score_value, (float, int), f"Score '{score_key}' in dataset '{dataset}', algorithm '{algorithm}', key '{key}' is not a float or int." )
                        self.assertTrue( math.isfinite(float(score_value)), f"Score '{score_key}' in dataset '{dataset}', algorithm '{algorithm}', key '{key}' is NaN/inf: {score_value}" )

    def test_benchmarking_deuxpointzero(self):

        from imputegap.recovery.benchmark import Benchmark
        # launch the evaluation
        bench = Benchmark()
        bench.eval_imputation(algorithms=["MeanImpute", "SoftImpute"],
                              datasets=["eeg-alcohol"],
                              patterns=["mcar"],
                              x_axis=[0.1, 0.2, 0.4, 0.6, 0.8])

        self.assertTrue(True)



    def test_benchmarking_matrix(self):
        """
        the goal is to test if only the simple imputation with ST-MVL has the expected outcome
        """
        alpha_1 = {'eegalcohol': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': 0, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': -10, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': -40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': -30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': 40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        alpha_2 = {'eegalcohol': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': -1, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': 50, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': 20, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': 30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': -100, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        beta_1 = {'eegreading': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': 0, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': -10, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': -40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': -30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': 40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        beta_2 = {'eegreading': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': -1, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': 50, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': 20, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': 30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': -100, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        delta_1 = {'fmristoptask': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': 0, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': -10, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': -40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': -30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': 40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        delta_2 = {'fmristoptask': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': -1, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': 50, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': 20, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': 30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': -100, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        epsilon_1 = {'chlorine': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': 0, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': -10, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': -40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': -30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': 40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        epsilon_2 = {'chlorine': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': -1, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': 50, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': 20, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': 30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': -100, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        gamma_1 = {'drift': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': 0, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': -10, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': -40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': -30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': 40, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}
        gamma_2 = {'drift': {'mcar': {'MeanImpute': {
            'default_params': {'0.05': {'scores': {'RMSE': -1, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'CDRec': {
            'default_params': {'0.05': {'scores': {'RMSE': 50, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'STMVL': {
            'default_params': {'0.05': {'scores': {'RMSE': 20, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'MRNN': {
            'default_params': {'0.05': {'scores': {'RMSE': 30, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}, 'IIM': {
            'default_params': {'0.05': {'scores': {'RMSE': -100, 'MAE': 1, 'MI': 2, 'CORRELATION': 3}},
                               '0.2': {'scores': {'RMSE': 1, 'MAE': 2, 'MI': 3, 'CORRELATION': 4}},
                               '0.4': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.6': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}},
                               '0.8': {'scores': {'RMSE': 0.5, 'MAE': 1.5, 'MI': 2.5, 'CORRELATION': 3.5}}}}}}}

        results = [alpha_1, alpha_2, beta_1, beta_2, delta_1, delta_2, epsilon_1, epsilon_2, gamma_1, gamma_2]

        # Replace the original 10 fixture assignments with the lines above.
        # Update the expectations in test_benchmarking_matrix as follows:
        expected_algos = ['STMVL', 'CDRec', 'IIM', 'MeanImpute', 'MRNN']
        expected_sets = ['eegalcohol', 'eegreading', 'fmristoptask', 'chlorine', 'drift']
        algo_order = ['CDRec', 'IIM', 'MeanImpute', 'MRNN', 'STMVL']
        expected_rmse = {'CDRec': 4.5, 'IIM': -5.5, 'MeanImpute': 0.4, 'MRNN': 0.5, 'STMVL': -1.5}

        print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n=====================================")
        bench = Benchmark()
        print(f"{alpha_1=}")
        scores_list, algos, sets = bench.avg_results_task(alpha_1, alpha_2, beta_1, beta_2, delta_1, delta_2, epsilon_1, epsilon_2, gamma_1, gamma_2)

        print(f"OUT: {scores_list=}")
        print(f"OUT: {algos=}")
        print(f"OUT: {sets=}")

        # Check that algos contains all expected algorithms
        #expected_algos = ["stmvl", "cdrec", "iim", "meanimpute", "mrnn"]
        self.assertCountEqual(
            algos, expected_algos,
            f"Missing algorithms in Benchmarking. Expected: {expected_algos}, Found: {algos}"
        )

        # Check that sets contains all expected datasets
        #expected_sets = ["eeg-alcohol", "eeg-reading", "fmri-stoptask", "chlorine", "drift"]
        self.assertCountEqual(
            sets, expected_sets,
            f"Missing datasets in Benchmarking. Expected: {expected_sets}, Found: {sets}"
        )

        # Check average values for each dataset and algorithm
        #algo_order = ['cdrec', 'iim', 'meanimpute', 'mrnn', 'stmvl']

        print(f"{scores_list=}")

        # Iterate through the matrix
        for i, dataset_scores in enumerate(scores_list):
            for j, rmse_value in enumerate(dataset_scores):
                algo = algos[j]  # Match the algorithm based on column index
                expected_value = expected_rmse[algo]

                print(f"{rmse_value=}")
                print(f"{expected_value=}")

                self.assertEqual(
                    rmse_value,
                    expected_value,
                    f"Unexpected RMSE for algorithm '{algo}' at dataset index {i}."
                )


        validation = bench.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric="RMSE", save_dir="./reports", display=False)
        validation = bench.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric="RMSE", save_dir="./reports", display=True)

        self.assertTrue(validation)
