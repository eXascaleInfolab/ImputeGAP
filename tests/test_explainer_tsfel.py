import unittest
import numpy as np
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestExplainerTSFEL(unittest.TestCase):

    def test_explainer_tsfel(self):
        """
        Verify if the SHAP TSFEL EXTRACTOR is working
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))

        exp = Explainer()
        shap_values, shap_details = exp.extractor_tsfel(data=ts_1.data)

        self.assertTrue(shap_values is not None)
        self.assertTrue(shap_details is not None)

        # Expected number of features per category
        expected_feature_counts = {"spectral": 111, "statistical": 31, "temporal": 14}

        # Initialize counters for actual feature counts
        actual_feature_counts = {category: 0 for category in expected_feature_counts.keys()}

        for feature_name, category, value in shap_details:
            # Increment the count for the feature's category
            self.assertIn(category, actual_feature_counts, f"Unexpected category: {category}")
            actual_feature_counts[category] += 1

            shap_value = shap_values[feature_name]

            self.assertFalse(np.isnan(shap_value), f"Feature {feature_name} in category {category} has NaN value")
            print(f"Feature {feature_name}\t\tin category {category}\t\twithout NaN value {shap_value}")

        print("\n\n\n")

        # Compare actual counts with expected counts
        for category, expected_count in expected_feature_counts.items():
            actual_count = actual_feature_counts[category]
            print(f"Category {category} has {actual_count} features, expected {expected_count}")
            self.assertEqual(actual_count, expected_count,
                f"Category {category} has {actual_count} features, expected {expected_count}")