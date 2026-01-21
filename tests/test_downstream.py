import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.recovery.contamination import GenGap
import pytest

class TestDownstream(unittest.TestCase):


    def test_downstream(self):
        """
        Verify if the downstream process is working
        """
        # Load and normalize the series
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("forecast-economy"))

        print(f"{utils.list_of_downstreams() = }")
        print(f"{utils.list_of_downstreams_darts() = }")
        print(f"{utils.list_of_downstreams_sktime() = }")

        # Create a mask for contamination
        ts_mask = GenGap.aligned(ts_1.data, rate_series=0.7)

        # Perform imputation
        imputer = Imputation.MatrixCompletion.CDRec(ts_mask)
        imputer.impute()

        models = utils.list_of_downstreams()

        # Configure downstream options
        downstream_options= [{"task": "forecast", "model": str(m), "params": None, "plots": None, "baseline": None, "split": 0.8 } for m in models]

        print(f"{downstream_options = }")

        for options in downstream_options:

            model = options.get("model")

            if model == "prophet":
                options = {"task": "forecast", "model": "prophet", "params": None, "plots": True, "baseline": "CDRec", "split": 0.8}

            # Score and evaluate
            imputer.score(ts_1.data, imputer.recov_data)
            imputer.score(ts_1.data, imputer.recov_data, downstream=options)

            # Assert metrics are dictionaries with values
            self.assertIsInstance(imputer.metrics, dict, "imputer.metrics should be a dictionary, for " + model)
            self.assertTrue(imputer.metrics, "imputer.metrics should not be empty, for " + model)

            self.assertIsInstance(imputer.downstream_metrics, dict, "imputer.downstream_metrics should be a dictionary, for " + model)
            self.assertTrue(imputer.downstream_metrics, "imputer.downstream_metrics should not be empty, for " + model)

            # Display the results
            ts_1.print_results(imputer.metrics, algorithm=model)
            ts_1.print_results(imputer.downstream_metrics, algorithm=model)

    def test_downstream_empty(self):
        """
        Verify if the downstream process is working
        """
        # Load and normalize the series
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("forecast-economy"))

        print(f"{utils.list_of_downstreams() = }")
        print(f"{utils.list_of_downstreams_darts() = }")
        print(f"{utils.list_of_downstreams_sktime() = }")

        # Create a mask for contamination
        ts_mask = GenGap.aligned(ts_1.data, rate_series=0.7)

        # Perform imputation
        imputer = Imputation.MatrixCompletion.CDRec(ts_mask)
        imputer.impute()

        models = utils.list_of_downstreams()

        # Configure downstream options
        downstream_options= [{"task": "nope", "model": "naive", "params": None, "plots": None, "baseline": None, "split": 0.8 }]

        print(f"{downstream_options = }")

        for options in downstream_options:

            model = options.get("model")

            # Score and evaluate
            imputer.score(ts_1.data, imputer.recov_data)

            with pytest.raises(TypeError):
                imputer.score(ts_1.data, imputer.recov_data, downstream=options)