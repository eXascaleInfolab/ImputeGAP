import unittest
from imputegap.recovery.contamination import GenGap


class TestLIB(unittest.TestCase):

    def test_lib(self):
        """
        Verify if the lib is working correctly
        """

        from imputegap.recovery.imputation import Imputation
        from imputegap.recovery.manager import TimeSeries
        from imputegap.tools import utils

        # initialize the time series object
        ts = TimeSeries()
        print(f"Imputation algorithms : {ts.algorithms}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("eeg-alcohol"))
        ts.normalize(normalizer="z_score")

        # contaminate the time series
        ts_m = GenGap.mcar(ts.data)

        # impute the contaminated series
        imputer = Imputation.MatrixCompletion.CDRec(ts_m)
        imputer.impute()

        # compute and print the imputation metrics
        imputer.score(ts.data, imputer.recov_data)
        ts.print_results(imputer.metrics)

        # plot the recovered time series
        ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets", display=False)

        self.assertTrue(imputer.recov_data is not None)
        self.assertTrue(ts_m is not None)
        self.assertTrue(imputer.metrics is not None)
        self.assertTrue(ts.data is not None)

