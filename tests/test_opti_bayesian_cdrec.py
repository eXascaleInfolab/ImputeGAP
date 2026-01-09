import unittest

from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestOptiCDREC(unittest.TestCase):

    def test_optimization_bayesian_cdrec(self):
        """
        the goal is to test if only the simple optimization with CDRec has the expected outcome
        """
        algorithm = "cdrec"
        dataset = "chlorine"

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(dataset))

        incomp_data = GenGap.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=2, offset=0.1, seed=True)
        params_optimal_load = utils.load_parameters(query="optimal", algorithm=algorithm, dataset=dataset, optimizer="b")

        algo_opti = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo_opti.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 5}})
        algo_opti.score(input_data=ts_1.data)
        metrics_optimal = algo_opti.metrics

        algo_default = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo_default.impute(params={"rank":2, "epsilon":0.000001, "iteration":100})
        algo_default.score(input_data=ts_1.data)
        metrics_default = algo_default.metrics

        algo_load = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo_load.impute(params=params_optimal_load)
        algo_load.score(input_data=ts_1.data)
        metrics_optimal_load = algo_load.metrics

        self.assertTrue(metrics_optimal["RMSE"] <= metrics_default["RMSE"], f"Imputation Expected {metrics_optimal['RMSE']} < {metrics_default['RMSE']} ")
        self.assertTrue(metrics_optimal_load["RMSE"] <= metrics_default["RMSE"], f"Expected {metrics_optimal_load['RMSE']} < {metrics_default['RMSE']}")