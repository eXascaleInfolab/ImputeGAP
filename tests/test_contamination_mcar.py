import math
import os
import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestContaminationMCAR(unittest.TestCase):

    def test_mcar_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test-logic-llm.txt"))

        series_impacted = [0.4]
        missing_rates = [40]
        series_check = ["1", "2", "6"]
        offset = 4
        block_size = 2


        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = GenGap.mcar(input_data=ts_1.data, rate_dataset=series_sel, rate_series=missing_rate, block_size=block_size, offset=offset, seed=True)

                print(f"{ts_contaminate}")
                check_nan_series = False

                for series in range(ts_contaminate.shape[1]):
                    data = ts_contaminate[:, series]
                    if str(series+1) in series_check:
                        if np.isnan(data).any():
                            check_nan_series = True
                    else:
                        if np.isnan(data).any():
                            check_nan_series = False
                            break
                        else:
                            check_nan_series = True

                self.assertTrue(check_nan_series, True)

    def test_mcar_position(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test.txt"))

        series_impacted = [0.4, 1]
        missing_rates = [0.1, 0.4, 0.6]
        ten_percent_index = int(ts_1.data.shape[1] * 0.1)

        for series_sel in series_impacted:
            for missing_rate in missing_rates:

                ts_contaminate = GenGap.mcar(input_data=ts_1.data,
                                                         rate_dataset=series_sel,
                                                         rate_series=missing_rate,
                                                         block_size=2, offset=0.1,
                                                         seed=True)

                if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                    check_position = False
                else:
                    check_position = True

                self.assertTrue(check_position, True)

    def test_mcar_selection_datasets(self):
        """
        test if only the selected values are contaminated in the right % of series with the right amount of values
        """
        datasets = ["bafu", "chlorine", "drift"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_series(utils.search_path(dataset))

            for S in series_impacted:
                for R in missing_rates:
                    ts_contaminate = GenGap.mcar(input_data=ts_1.data, rate_series=R, rate_dataset=S, block_size=block_size, offset=offset, seed=True)
                    ts_contaminate = ts_contaminate.T

                    # 1) Check if the number of NaN values is correct
                    M, N = ts_contaminate.shape
                    P = math.ceil(N * offset)
                    W = int(N * R)
                    expected_contaminated_series = int(np.ceil(M * S))
                    B = int(W / block_size)
                    total_expected = (B * block_size) * expected_contaminated_series
                    total_nan = np.isnan(ts_contaminate).sum()

                    print(f"\nExpected {total_expected} total missing values but found {total_nan}\n\t"
                          f"for dataset_rate {S * 100}% and series_rate {R * 100}% / ({M},{N})\n\t")

                    self.assertEqual(total_nan, total_expected,
                                     (f"\nExpected {total_expected} total missing values but found {total_nan}\n\t"
                                      f"for dataset_rate {S * 100}% and series_rate {R * 100}% / ({M},{N})\n\t"))

                    # 2) Check if the correct percentage of series are contaminated
                    contaminated_series = np.isnan(ts_contaminate).any(axis=1).sum()

                    self.assertEqual(contaminated_series, expected_contaminated_series, f"Expected {expected_contaminated_series} contaminated series but found {contaminated_series}")


    def test_mcar_position_datasets(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        datasets = ["test-logic-llm.txt", "meteo"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_series(utils.search_path(dataset))
            ten_percent_index = int(ts_1.data.shape[1] * 0.1)

            for series_sel in series_impacted:
                for missing_rate in missing_rates:

                    ts_contaminate = GenGap.mcar(input_data=ts_1.data, rate_dataset=series_sel, rate_series=missing_rate, block_size=block_size, offset=offset, seed=True)

                    if np.isnan(ts_contaminate[:ten_percent_index, :]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)



    def test_contaminate_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))

        ts_2 = TimeSeries()
        ts_2.import_matrix(GenGap.mcar(input_data=ts_1.data, rate_dataset=0.4, rate_series=0.1, block_size=10, offset=0.1, seed=True))

        ts_1.print()
        filepath = ts_1.plot(input_data=ts_1.data, incomp_data=ts_2.data, nbr_series=10, nbr_val=100, save_path="./assets/", display=False)
        self.assertTrue(os.path.exists(filepath))


    def test_mcar_size_of_block(self):
        """
        test if the size of the block is at least the number defined my the user
        """
        datasets = ["chlorine","eeg-reading", "eeg-alcohol"]
        series_impacted = [0.4, 1]
        missing_rates = [0.2, 0.6]
        offset = 0.1
        block_size = 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            if dataset == "eeg-reading":
                ts_1.load_series(utils.search_path(dataset), header=True)
            else:
                ts_1.load_series(utils.search_path(dataset))

            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = GenGap.mcar(input_data=ts_1.data,
                                                             rate_series=missing_rate,
                                                             rate_dataset=series_sel,
                                                             block_size=block_size, offset=offset,
                                                             seed=True)

                    ts_contaminate = ts_contaminate.T

                    for i, series in enumerate(ts_contaminate):
                        nan_blocks = []
                        block_indices = []
                        current_block_size = 0
                        series_size = len(series)
                        lower_bound = int(offset * series_size) + block_size
                        upper_bound = series_size - lower_bound - block_size
                        protected_indices = set(range(0, lower_bound)) | set(range(upper_bound, series_size))

                        # Find NaN blocks and their indices
                        for index, value in enumerate(series):
                            if np.isnan(value):
                                current_block_size += 1
                                block_indices.append(index)
                            else:
                                if current_block_size > 0:
                                    if not any(i in protected_indices for i in block_indices):
                                        nan_blocks.append(current_block_size)
                                    current_block_size = 0
                                    block_indices = []

                        for block in nan_blocks:
                            print(f"\t\tDataset: {dataset}, Series: {i} "
                                  f"\t\tBlock size {block} found, expected at least {block_size}.")

                            assert block >= block_size, (
                                f"Dataset: {dataset}, Series: {i}, "
                                f"Block size {block} found, expected at least {block_size}."
                            )



    def test_mcar_missing_percentage_total(self):
        """
        Test if the size of the missing percentage in a contaminated time series meets the expected number defined by the user.
        """
        datasets = ["eeg-alcohol", "airq", "chlorine"]
        series_impacted = [0.4, 0.8]
        missing_rates = [0.2, 0.65]
        offset, block_size = 0.1, 10

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_series(utils.search_path(dataset))
            N, M = ts_1.data.shape

            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    ts_contaminate = GenGap.mcar(input_data=ts_1.data, rate_series=missing_rate, rate_dataset=series_sel, block_size=block_size, offset=offset, seed=True)

                    #print(*[f"({indc} {se})" for indc, se in enumerate(ts_contaminate)], sep=" ")  # debug

                    nbr_series_contaminated = 0
                    for inx in range(ts_contaminate.shape[1]):
                        current_series = ts_contaminate[:, inx]

                        if np.isnan(current_series).any():
                            nbr_series_contaminated = nbr_series_contaminated+1

                            num_missing_values = np.isnan(current_series).sum()
                            expected_num_missing = int(N * missing_rate)
                            b_compensation = 0

                            print(f"\t\tNUMBR OF VALUES for series #{inx} : {num_missing_values}")
                            print(f"\t\tEXPECTED VALUES for series #{inx} : {expected_num_missing}\n")

                            if expected_num_missing != num_missing_values:
                                b_compensation = expected_num_missing - num_missing_values

                                expected_num_missing = int(N * missing_rate)
                                B = int(expected_num_missing / block_size)
                                expected_num_missing = (B * block_size)
                                print(f"\t\t\tBLOCK SIZE LIMITATION {block_size}: ", expected_num_missing, "\n")

                            self.assertEqual(num_missing_values, expected_num_missing,
                                msg=f"Dataset '{dataset}', Series Index {current_series}: "
                                    f"Expected {expected_num_missing} missing values, but found {num_missing_values}.")

                            percentage = ((expected_num_missing+b_compensation)/N)*100
                            print(f"\t\t{b_compensation = }")
                            print(f"\t\t{expected_num_missing = }")
                            print(f"\t\t{N = }")
                            print(f"\t\tPERCENTAGE VALUES for series #{inx} : {percentage}")
                            print(f"\t\tPERCENTAGE VALUES for series #{inx} : {percentage}")
                            print(f"\t\tEXPECTED % VALUES for series #{inx} : {missing_rate * 100}\n")

                            if b_compensation == 0:
                                self.assertEqual(percentage, missing_rate*100,
                                     msg=f"Dataset '{dataset}': % Expected {missing_rate*100}, "
                                     f"but found {percentage}.")
                            else:
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

