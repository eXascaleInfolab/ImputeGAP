import os
import unittest
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap


class TestSPIRIT(unittest.TestCase):

    def test_imputation_spirit(self, name="spirit", limit=0.05):
        """
        the goal is to test if only the simple imputation with the technique has the expected outcome
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "toml/imputegap_results.toml")

        dataset, rmse, mae = utils.get_resuts_unit_tests(algo_name=name, loader=path)

        ts = TimeSeries()
        ts.load_series(utils.search_path("chlorine"), normalizer="z_score")

        incomp_data = GenGap.aligned(ts.data, single_series=0)
        algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
        algo.impute()
        algo.score(ts.data)
        metrics = algo.metrics

        print(f"{name}:{metrics = }\n")

        ts.print_results(algo.metrics, algo.algorithm)

        expected_metrics = {"RMSE": rmse, "MAE": mae}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < limit, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < limit, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")

        # ==============================================================================================================

        algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
        algo.impute(params={'k': 4, 'w': 6, 'lambda_value': 1.0})
        algo.score(ts.data)
        metrics = algo.metrics

        print(f"{name}:{metrics = }\n")

        ts.print_results(algo.metrics, algo.algorithm)

        expected_metrics = {"RMSE": rmse, "MAE": mae}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < limit, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < limit, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")