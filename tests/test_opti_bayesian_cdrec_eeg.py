import unittest

from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestOptiCDRECEEG(unittest.TestCase):

    def test_optimization_bayesian_cdrec_eeg(self):
        """
        the goal is to test if only the simple optimization with CDRec has the expected outcome
        """
        algorithm = "cdrec"
        dataset = "eeg-alcohol"

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(dataset), header=False)

        incomp_data = GenGap.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.36, block_size=2, offset=0.1, seed=True)

        params = utils.load_parameters(query="default", algorithm=algorithm)

        algo_opti = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo_opti.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 5}})
        algo_opti.score(input_data=ts_1.data)
        metrics_optimal = algo_opti.metrics

        print("\t\t\t\toptimal params", algo_opti.parameters)
        print("\t\t\t\tdefault params", params, "\n")

        algo_default = Imputation.MatrixCompletion.CDRec(incomp_data)
        algo_default.impute(params=params)
        algo_default.score(input_data=ts_1.data)
        metrics_default = algo_default.metrics

        self.assertTrue(metrics_optimal["RMSE"] < metrics_default["RMSE"], f"Expected {metrics_optimal['RMSE']} < {metrics_default['RMSE']}")
