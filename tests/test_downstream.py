import unittest
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.recovery.contamination import GenGap
import pytest


from imputegap.recovery.downstream import Classifier, Forecaster, Artifact


class TestDownstream(unittest.TestCase):


    def test_downstream(self):
        """
        Verify if the downstream process is working
        """
        # Load and normalize the series
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("forecast-economy"))

        print(f"{utils.list_of_forecasters() = }")
        print(f"{utils.list_of_downstreams_darts() = }")
        print(f"{utils.list_of_downstreams_sktime() = }")

        # Create a mask for contamination
        ts_mask = GenGap.aligned(ts_1.data, rate_series=0.7)

        # Perform imputation
        imputer = Imputation.MatrixCompletion.CDRec(ts_mask)
        imputer.impute()

        models = utils.list_of_forecasters()
        forecasters = []
        for model in models:
            if model != "cif":
                artifact = Artifact(dataset="test-logic-llm", model=model, algorithm="cdrec", pattern="mcar", horizon=2)
                forecasters.append(Forecaster(task="forecasting", model=artifact.model, y_train=None, season=2, use_cache=False, to_cache=False, artifact=artifact, plots=False, verbose=False))

        print(f"{forecasters = }")

        for forecaster in forecasters:

            model = forecaster.model

            print(f"{model = }")

            imputer.score(ts_1.data, imputer.recov_data)
            print(f"{imputer.metrics=}")

            if (isinstance(forecaster, Forecaster) or isinstance(forecaster, Classifier)) and forecaster is not None:
                assert True, (
                    f"Unexpected options value: {forecaster} "
                    f"{type(forecaster)=} vs {Forecaster=}"
                )
            else:
                assert False, (
                    f"Unexpected options value: {forecaster} "
                    f"{type(forecaster)=} vs {Forecaster=}"
                )

            imputer.score(ts_1.data, imputer.recov_data, downstream=forecaster)
            print(f"{imputer.downstream_metrics=}")


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

        print(f"{utils.list_of_forecasters() = }")
        print(f"{utils.list_of_downstreams_darts() = }")
        print(f"{utils.list_of_downstreams_sktime() = }")

        # Create a mask for contamination
        ts_mask = GenGap.aligned(ts_1.data, rate_series=0.7)

        # Perform imputation
        imputer = Imputation.MatrixCompletion.CDRec(ts_mask)
        imputer.impute()

        models = utils.list_of_forecasters()

        artifact = Artifact(dataset="eeg-alcohol", model="exp-smoothing", algorithm="cdrec", pattern="mcar", horizon=12)
        forecaster = Forecaster(task="nope", model=artifact.model, y_train=None, season=12, use_cache=False, to_cache=True, artifact=artifact, plots=True, verbose=False)

        # Configure downstream options
        downstream_options= [forecaster]

        print(f"{downstream_options = }")

        for options in downstream_options:

            # Score and evaluate
            imputer.score(ts_1.data, imputer.recov_data)

            with pytest.raises(TypeError):
                imputer.score(ts_1.data, imputer.recov_data, downstream=options)