import unittest
import numpy as np
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestExplainer(unittest.TestCase):

    def test_explainer_shap(self):
        """
        Verify if the SHAP explainer is working
        """
        filename = "chlorine"

        exp = Explainer()

        RMSE = [0.508740447256769, 0.5834508294411466, 0.5318564446461029, 0.5162089180931576, 0.46608400269248135,
         0.4531125301877796, 0.4071204339932292, 0.38188027439915706, 0.32530898769725997, 0.3077025334161655,
         0.275820985814237, 0.24169961372557672, 0.18094568173830244, 0.12943484328240668, 0.3299572333029556]

        expected_categories, expected_features, _ = exp.load_configuration()

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path(filename))

        exp.shap_explainer(input_data=ts_1.data, file_name=filename, rate_dataset=0.3, seed=True, verbose=True, extractor="pycatch22")

        exp.print(exp.shap_values, exp.shap_details)

        self.assertTrue(exp.shap_values is not None)
        self.assertTrue(exp.shap_details is not None)

        for i, (_, output) in enumerate(exp.shap_details):
            assert np.isclose(RMSE[i], output, atol=0.75)

        for i, (x, algo, rate, description, feature, category, mean_features) in enumerate(exp.shap_values):
            assert rate >= 0, f"Rate must be >= 0, but got {rate}"

            self.assertTrue(x is not None and not (isinstance(x, (int, float)) and np.isnan(x)))
            self.assertTrue(algo is not None)
            self.assertTrue(rate is not None and not (isinstance(rate, (int, float)) and np.isnan(rate)))
            self.assertTrue(description is not None)
            self.assertTrue(feature is not None)
            self.assertTrue(category is not None)
            self.assertTrue(mean_features is not None and not (isinstance(mean_features, (int, float)) and np.isnan(mean_features)))

            # Check relation feature/category
            feature_found_in_category = False
            for exp_category, exp_features in expected_categories.items():
                if feature in exp_features:
                    assert category == exp_category, f"Feature '{feature}' must in '{exp_category}', but is in '{category}'"
                    feature_found_in_category = True
                    break
            assert feature_found_in_category, f"Feature '{feature}' not found in any category"

            # Check relation description/feature
            if feature in expected_features:
                expected_description = expected_features[feature]
                assert description == expected_description, f"Feature '{feature}' has wrong description. Expected '{expected_description}', got '{description}' "
            else:
                assert False, f"Feature '{feature}'not found in the FEATURES dictionary"
