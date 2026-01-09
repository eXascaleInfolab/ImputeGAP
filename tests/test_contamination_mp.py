import unittest
import numpy as np
import math
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap


class TestContaminationMP(unittest.TestCase):

    def test_mp_selection(self):
        """
        the goal is to test if the number of NaN values expected are provided in the contamination output
        """

        datasets = ["drift", "chlorine", "eeg-alcohol", "test-logic-llm.txt"]
        series_impacted = [0.1, 0.5, 1]  # percentage of series impacted
        missing_rates = [0.1, 0.5, 0.9]  # percentage of missing values with NaN


        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))
            N, M = ts.data.shape  # series, values
            P = math.ceil(ts.data.shape[0] * 0.1)

            for S in series_impacted:
                for R in missing_rates:
                    incomp_data = GenGap.aligned(input_data=ts.data, rate_dataset=S, rate_series=R, offset=P)

                    n_nan = np.isnan(incomp_data).sum()
                    expected_nan_series = math.ceil(S * M)
                    expected_nan_values = int(N * R)
                    expected_nan = expected_nan_series * expected_nan_values

                    print(f"\n\tExpected {expected_nan} total missing values but found {n_nan}\n\t\t"
                          f"for dataset_rate {S*100}% and series_rate {R*100}% / ({M},{N})\n\t\t"
                          f"expected_nan_series {expected_nan_series}, expected_nan_values {expected_nan_values}\n")

                    self.assertEqual(n_nan, expected_nan, (f"\nExpected {expected_nan} total missing values but found {n_nan}\n\t"
                          f"for dataset_rate {S*100}% and series_rate {R*100}% / ({M},{N})\n\t"
                          f"expected_nan_series {expected_nan_series}, expected_nan_values {expected_nan_values}\n"))


    def test_mp_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test-logic-llm.txt"))

        series_impacted = [0.4, 0.8]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = GenGap.aligned(input_data=ts_1.data, rate_dataset=series_sel, rate_series=missing_rate, offset=0.1)

                if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)

    def test_mp_missing_percentage_total(self):
        """
        Test if the size of the missing percentage in a contaminated time series meets the expected number defined by the user.
        """
        datasets = ["chlorine", "eeg-alcohol", "drift"]
        series_impacted = [0.4, 0.8]
        missing_rates = [0.2, 0.6]
        offset = 0.1

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.data = None
            ts_1.load_series(utils.search_path(dataset))
            N, M = ts_1.data.shape

            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = GenGap.aligned(input_data=ts_1.data, rate_dataset=series_sel, rate_series=missing_rate, offset=offset)

                    nbr_series_contaminated = 0
                    for inx in range(ts_contaminate.shape[1]):
                        current_series = ts_contaminate[:, inx]

                        if np.isnan(current_series).any():
                            nbr_series_contaminated = nbr_series_contaminated+1

                            num_missing_values = np.isnan(current_series).sum()
                            expected_num_missing = int(N * missing_rate)

                            print(f"\t\tNUMBR OF VALUES for series #{inx} : {num_missing_values}")
                            print(f"\t\tEXPECTED VALUES for series #{inx} : {expected_num_missing}\n")

                            self.assertEqual(num_missing_values, expected_num_missing,
                                msg=f"Dataset '{dataset}', Series Index {current_series}: "
                                    f"Expected {expected_num_missing} missing values, but found {num_missing_values}.")

                            percentage = ((expected_num_missing/N)*100)
                            print(f"\t\tPERCENTAGE VALUES for series #{inx} : {percentage}")
                            print(f"\t\tEXPECTED % VALUES for series #{inx} : {missing_rate*100}\n")

                            self.assertAlmostEqual(percentage, missing_rate * 100, delta=1,
                                msg=f"Dataset '{dataset}': Expected {missing_rate * 100}%, but found {percentage}%.")

                            print("\n\n\n=inner_loop=============================================================\n\n")

                    expected_nbr_series = int(np.ceil(M*series_sel))
                    self.assertEqual(
                        nbr_series_contaminated, expected_nbr_series,
                        msg=f"Dataset '{dataset}': Expected {expected_nbr_series} contaminated series, "
                            f"but found {nbr_series_contaminated}."
                    )

                    print("NUMBR OF SERIES : ", nbr_series_contaminated)
                    print("EXPECTED SERIES : ", expected_nbr_series, "\n")

    def test_single_series(self):
        """
        Test if the size of the missing percentage in a contaminated time series meets the expected number defined by the user.
        """
        datasets = ["test-logic-llm.txt"]
        series_pointer = [0, 3]
        series_impacted = [0.4, 0.8]
        missing_rates = [0.2, 0.6, 0.85]
        offset = 5

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.data = None
            ts_1.load_series(utils.search_path(dataset))
            N, M = ts_1.data.shape
            for series_sel in series_impacted:
                for Z in series_pointer :
                    for missing_rate in missing_rates:
                        ts_contaminate = GenGap.aligned(input_data=ts_1.data, single_series=Z, rate_series=missing_rate, offset=offset)

                        self.assertEqual(ts_contaminate.shape, (N, M))

                        for inx in range(ts_contaminate.shape[1]):
                            current_series = ts_contaminate[:, inx]
                            nan_mask = np.isnan(current_series)

                            # 1) check that current series has NaNs only if inx == Z
                            if inx != Z:
                                self.assertFalse(
                                    nan_mask.any(),
                                    msg=f"Series {inx} should not contain NaNs when single_series={Z}",
                                )
                                continue

                            # Now: inx == Z
                            self.assertTrue(
                                nan_mask.any(),
                                msg=f"Selected series {Z} should contain NaNs",
                            )

                            # 2) NaNs only start after the offset
                            self.assertFalse(
                                nan_mask[:offset].any(),
                                msg=f"NaNs found before offset={offset} in series {Z}",
                            )

                            nan_idx = np.where(nan_mask)[0]
                            first_nan = int(nan_idx.min())
                            last_nan = int(nan_idx.max())

                            self.assertGreaterEqual(
                                first_nan, offset,
                                msg=f"First NaN index {first_nan} is before offset={offset} in series {Z}",
                            )

                            # Ensure NaNs form a single contiguous block
                            self.assertTrue(
                                np.all(np.diff(nan_idx) == 1),
                                msg=f"NaNs are not contiguous in series {Z} (indices: {nan_idx[:20]} ...)",
                            )
                            self.assertTrue(
                                nan_mask[first_nan:last_nan + 1].all(),
                                msg=f"NaN block has gaps in series {Z}",
                            )

                            # 3) check block size matches expected
                            expected_nans = int(N * missing_rate)
                            self.assertEqual(
                                len(nan_idx),
                                expected_nans,
                                msg=(
                                    f"Unexpected number of NaNs in series {Z}: "
                                    f"got {len(nan_idx)}, expected {expected_nans} "
                                    f"(N={N}, missing_rate={missing_rate})"
                                ),
                            )