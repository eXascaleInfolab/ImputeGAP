import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap


class TestOptiSTMVL(unittest.TestCase):

    def test_optimization_bayesian_stmvl(self):
        """
        the goal is to test if only the simple optimization with stmvl has the expected outcome
        """

        algorithm = "stmvl"
        dataset = "chlorine"

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(dataset), nbr_val=200)


        incomp_data = GenGap.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=2, offset=0.1, seed=True)

        params = utils.load_parameters(query="default", algorithm=algorithm)
        params_optimal_load = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="b")

        algo_opti = Imputation.PatternSearch.STMVL(incomp_data)
        algo_opti.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})
        algo_opti.score(input_data=ts_1.data)
        metrics_optimal = algo_opti.metrics

        algo_default = Imputation.PatternSearch.STMVL(incomp_data)
        algo_default.impute(params=params)
        algo_default.score(input_data=ts_1.data)
        metrics_default = algo_default.metrics

        print(f"{metrics_optimal = }")
        print(f"{metrics_default = }")

        self.assertTrue(abs(metrics_optimal["RMSE"] - metrics_default["RMSE"]) < 0.025, f"Expected {metrics_optimal['RMSE']} - {metrics_default['RMSE']} < 0.025")
        self.assertTrue(metrics_optimal["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal['RMSE']} < {metrics_default['RMSE']}")