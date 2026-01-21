import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestOptiMRNN(unittest.TestCase):

    def test_optimization_bayesian_mrnn(self):
        """
        the goal is to test if only the simple optimization with mrnn has the expected outcome
        """
        dataset, algorithm = "chlorine", "mrnn"

        ts_1 = TimeSeries()
        ts_1.load_series(data=utils.search_path(dataset), nbr_val=200)

        incomp_data = GenGap.mcar(input_data=ts_1.data)

        params = utils.load_parameters(query="default", algorithm=algorithm)

        algo_opti = Imputation.DeepLearning.MRNN(incomp_data)

        expected = (
            r"This algorithm 'mrnn' is not optimized for this optimizer\."
        )

        with self.assertRaisesRegex(ValueError, expected):
            algo_opti.impute(user_def=False, params={"input_data": ts_1.data, "optimizer": "bayesian", "options": {"n_calls": 2}})
