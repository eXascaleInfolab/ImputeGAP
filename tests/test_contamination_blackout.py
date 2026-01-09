import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap


class TestContaminationBlackout(unittest.TestCase):

    def test_blackout_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test.txt"))

        missing_rates = [0.4, 0.9]
        offset = 0.1
        N, M = ts_1.data.shape

        for missing_rate in missing_rates:
            ts_contaminate = GenGap.blackout(input_data=ts_1.data, rate_series=missing_rate, offset=offset)

            n_nan = np.isnan(ts_contaminate).sum()
            expected_nan_series = M
            expected_nan_values = int(N * missing_rate)
            expected = expected_nan_series * expected_nan_values

            self.assertEqual(n_nan, expected, f"Expected {expected} contaminated series but found {n_nan}")

    def test_blackout_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test-logic-llm.txt"))

        missing_rates = [0.1, 0.4, 0.9]
        ten_percent_index = int(ts_1.data.shape[0] * 0.1)

        for missing_rate in missing_rates:

            ts_contaminate = GenGap.blackout(input_data=ts_1.data, rate_series=missing_rate, offset=0.1)

            print(f"{ts_contaminate = }")

            if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                check_position = False
            else:
                check_position = True

            self.assertTrue(check_position, True)

        for missing_rate in missing_rates:

            ts_contaminate = GenGap.blackout(input_data=ts_1.data, rate_series=missing_rate, offset=4)

            print(f"{ts_contaminate = }")

            if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                check_position = False
            else:
                check_position = True

            self.assertTrue(check_position, True)