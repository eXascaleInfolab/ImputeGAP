import os
import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestCSDI(unittest.TestCase):

    def test_imputation_csdi_dft(self, name="csdi", limit=0.05):
        """
        the goal is to test if only the simple imputation with GPT4TS has the expected outcome
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "toml/imputegap_results.toml")
        
        dataset, rmse, mae = utils.get_resuts_unit_tests(algo_name=name, loader=path)
        
        ts = TimeSeries()
        ts.load_series(utils.search_path(dataset), normalizer="z_score", nbr_val=250, nbr_series=40)

        incomp_data = GenGap.mcar(ts.data)

        algo = Imputation.DeepLearning.CSDI(incomp_data).impute()
        algo.score(ts.data)
        metrics = algo.metrics

        print(f"{name}:{metrics = }\n")

        ts.print_results(algo.metrics, algo.algorithm)

        expected_metrics = {"RMSE": rmse, "MAE": mae}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < limit, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < limit, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")