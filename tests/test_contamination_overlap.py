import math
import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestContaminationOverlap(unittest.TestCase):

    def test_overlap_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))

        series_impacted = [0.4, 0.8]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:

            ts_contaminate = GenGap.overlap(input_data=ts_1.data, rate_series=series_sel, limit=1, shift=0.1, offset=100)

            if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                check_position = False
            else:
                check_position = True

            self.assertTrue(check_position, True)

    def get_last_nan_series_index(self, matrix):
        last_nan_index = None  # Initialize the variable to store the result
        all_nan = True  # Assume all series have NaN values initially

        for i in range(matrix.shape[1]):  # Iterate in reverse
            d = matrix[:, i]
            if np.isnan(d).any():  # Check if any NaN exists in the series
                last_nan_index = i + 1  # Update the variable with the index + 1
            else:
                all_nan = False  # Found a series without NaN, update the flag

        if all_nan:
            return matrix.shape[0]  # Return the size of shape[0] if all series have NaN

        return last_nan_index  # Otherwise, return the last series index with NaN

    def test_overlap_logic(self):
        """
        The goal is to test if the logic of the overlap contamination is respected.
        Each series is contaminated in an overlap manner, starting from the end of the contamination
        of the previous series and continuing with overlap.
        """

        datasets = ["test-logic-llm.txt", "chlorine", "eeg-alcohol"]
        series_rate = [0.2, 0.5, 0.8]  # Percentage of series impacted
        P = 0.1  # Offset zone
        shift = [0.05, 0.1]


        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))

            for S in series_rate:

                for A in shift:
                    # Generate disjoint contamination
                    ts_miss = GenGap.overlap(input_data=ts.data, rate_series=S, limit=1, shift=A, offset=P)

                    NS, M = ts.data.shape

                    INC = 0  # Incremental counter to track contamination shifts
                    X = math.ceil(ts.data.shape[0] * P)  # 100

                    FINAL_LIMIT = self.get_last_nan_series_index(ts_miss)

                    for series_index in range(ts_miss.shape[1]):
                        series = ts_miss[:, series_index]

                        N = len(series)  # Total number of values in the series
                        O = int(N * P)  # Values to protect at the beginning of the series
                        W = int(N * S)  # Number of data points to remove
                        if INC != 0:
                            X = X - int(N * A)
                        L = X + W  # Ending position for contamination in the current series

                        print(*[f"({indc} {se})" for indc, se in enumerate(series)], sep=" ")

                        # 1. Check the number of NaN values
                        nbr_expected_nan = W

                        if INC==0:
                            nbr_nan = np.isnan(series).sum()
                        else:
                            nbr_nan = int(np.isnan(series).sum() * A)

                        self.assertTrue(nbr_expected_nan - nbr_nan >= 0,
                            f"Series {series_index}: Expected {nbr_expected_nan} NaN values, found {nbr_nan}.")

                        # 2. Check the disjoint logic
                        # Subsequent series: NaN values should range from X to L
                        self.assertTrue(np.all(np.isnan(series[X:L])),
                            f"Series {series_index}: NaN values not properly placed in range X to L. {X}>{L},"
                            f"for P {P}, O {O}, W {W}, N {N}, S {S}, A {A}, INC {INC}, LIMIT, {FINAL_LIMIT}")

                        # Ensure no NaN values outside the expected range
                        self.assertFalse(np.isnan(series[:O]).any(),
                            f"Series {series_index}: Unexpected NaN values in the protected offset region.")

                        self.assertFalse(np.isnan(series[L:]).any(),
                            f"Series {series_index}: Unexpected NaN values after the contamination region.")

                        # Update X and INC
                        X = L
                        INC = INC + 1

                        # Exit the loop if INC exceeds FINAL_LIMIT
                        if INC == FINAL_LIMIT:
                            break

                        # Check if INC matches FINAL_LIMIT
                        self.assertTrue(INC < FINAL_LIMIT,
                                         f"INC < FINAL_LIMIT ({INC}!<{FINAL_LIMIT}).")

                    if M < FINAL_LIMIT:
                        # Check if INC matches FINAL_LIMIT
                        self.assertEqual(INC, M,
                                         f"INC ({INC}) does not match M ({M}).")
                    else:
                        # Check if INC matches FINAL_LIMIT
                        self.assertEqual(INC, FINAL_LIMIT,
                            f"INC ({INC}) does not match FINAL_LIMIT ({FINAL_LIMIT}).")
