import unittest
import numpy as np
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries


class TestExplainerPyCatch(unittest.TestCase):

    def test_explainer_pycatch(self):
        """
        Verify if the SHAP TSFEL EXTRACTOR is working
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))

        exp = Explainer()

        categories, features, _ = exp.load_configuration()
        shap_values, shap_details = exp.extractor_pycatch(data=ts_1.data, features_categories=categories, features_list=features, do_catch24=False)

        self.assertTrue(shap_values is not None)
        self.assertTrue(shap_details is not None)

        # Expected number of features per category
        expected_feature_counts = {"Geometry": 11, "Correlation": 6, "Transformation": 2, "Trend": 3}

        # Initialize counters for actual feature counts
        actual_feature_counts = {category: 0 for category in expected_feature_counts.keys()}

        print("\t\ttotal feature counts:", actual_feature_counts)

        for feature_name, category, value in shap_details:
            # Increment the count for the feature's category
            self.assertIn(category, actual_feature_counts, f"Unexpected category: {category}")
            actual_feature_counts[category] += 1

            if feature_name in shap_values:
                shap_value = shap_values[feature_name]

                # Ensure the SHAP value is numeric and not NaN
                if isinstance(shap_value, (int, float)):
                    self.assertFalse(np.isnan(shap_value), f"Feature {feature_name} in category {category} has NaN value")
                    print(f"Feature {feature_name}\t\tin category {category}\t\twithout NaN value {shap_value}")
                else:
                    self.fail(f"Feature {feature_name} in category {category} has a non-numeric value: {shap_value}")
            else:
                self.fail(f"Feature {feature_name} in category {category} is missing in shap_values")

        print("\n\n\n")

        total = 0
        # Compare actual counts with expected counts
        for category, expected_count in expected_feature_counts.items():
            actual_count = actual_feature_counts[category]
            total = total + actual_count
            print(f"Category {category} has {actual_count} features, expected {expected_count}")
            self.assertEqual(actual_count, expected_count,
                             f"Category {category} has {actual_count} features, expected {expected_count}")

        self.assertEqual(total, 22, f"Number of feature is not equal to 738 > {total}")