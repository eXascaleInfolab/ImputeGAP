import unittest
import numpy as np
import math

from scipy.stats import norm

from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap



class TestContaminationDistribution(unittest.TestCase):


    def load_probabilities(self, data, std_dev=0.2, transpose=False):
        probabilities_final = []


        for series in range(0, data.shape[1]):
            d = data[:, series]
            S = int(series)
            N = d.shape[0]
            P = math.ceil(N * 0.1)
            W = int(N * 0.2)
            R = np.arange(P, N)

            # probability density function
            mean = np.mean(data[S])
            mean = max(min(mean, 1), -1)

            probabilities = norm.pdf(R, loc=P + mean * (N - P), scale=std_dev * (N - P))

            # normalizes the probabilities so that their sum equals 1
            probabilities /= probabilities.sum()

            probabilities_final.append(probabilities)

        #probabilities_final = np.array(probabilities_final).T

        return probabilities_final



    def test_distribution_selection(self):
        """
        the goal is to test if the number of NaN values expected are provided in the contamination output
        """

        datasets = ["test-logic-llm.txt", "drift", "chlorine", "eeg-alcohol"]
        series_impacted = [0.1, 0.5, 1]  # percentage of series impacted
        missing_rates = [0.1, 0.5, 0.9]  # percentage of missing values with NaN


        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))
            P = math.ceil(ts.data.shape[0] * 0.1)
            print(f"{P=}")
            for S in series_impacted:
                for R in missing_rates:
                    D = self.load_probabilities(ts.data)
                    incomp_data = GenGap.distribution(input_data=ts.data, rate_dataset=S, rate_series=R, probabilities_list=D, offset=P)
                    N, M = incomp_data.data.shape

                    n_nan = np.isnan(incomp_data).sum()
                    expected_nan_series = math.ceil(S * M)
                    expected_nan_values = int(N * R)
                    expected_nan = expected_nan_series * expected_nan_values

                    print(f"\nExpected {expected_nan} total missing values but found {n_nan}\n\t"
                          f"for dataset_rate {S * 100}% and series_rate {R * 100}% / ({M},{N})\n\t")

                    self.assertEqual(expected_nan, n_nan,
                                     (f"\nExpected {expected_nan} total missing values but found {n_nan}\n\t"
                                      f"for dataset_rate {S * 100}% and series_rate {R * 100}% / ({M},{N})\n\t"))


    def test_distribution_position(self):
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

                D = self.load_probabilities(ts_1.data)
                ts_contaminate = GenGap.distribution(input_data=ts_1.data, rate_dataset=series_sel, rate_series=missing_rate, probabilities_list=D, offset=0.1)

                self.assertFalse(np.isnan(ts_contaminate[:ten_percent_index, :]).any(),
                    msg=f"Found NaNs in first {ten_percent_index} rows (series_sel={series_sel}, missing_rate={missing_rate})")


    def test_distribution_logic(self):
        """
        The goal is to test if the logic of the contamination is respected.
        Specifically, contamination with a higher standard deviation should result in
        more sparsely distributed NaN values compared to a lower standard deviation.
        """

        datasets = ["chlorine"]
        nbr_series_impacted = [0.2, 0.5, 0.80]  # Percentage of series impacted
        missing_rates_per_series = [0.4, 0.6]  # Percentage of missing values with NaN
        std_devs = [0.2, 0.5]  # Standard deviations to test
        P = 0.1  # Offset zone

        for dataset in datasets:
            ts = TimeSeries()
            ts.load_series(utils.search_path(dataset))

            for S in nbr_series_impacted:
                for R in missing_rates_per_series:
                    densities = {}

                    for std_dev in std_devs:
                        # Generate contamination with the current standard deviation
                        D = self.load_probabilities(ts.data, std_dev=std_dev)
                        contaminated_data = GenGap.distribution(input_data=ts.data, rate_dataset=S, rate_series=R, probabilities_list=D, offset=P)

                        # Calculate positions of NaN values
                        nan_positions = np.where(np.isnan(contaminated_data))

                        # Center of the time series (considering offset zone)
                        center = int((ts.data.shape[0] + (ts.data.shape[1] * P)) // 2)

                        # Compute average distances of NaN positions from the center
                        density = np.abs(nan_positions[0] - center).mean()
                        densities[std_dev] = density

                    self.assertLess(densities[0.2], densities[0.5],
                        f"Medium deviation density {densities[0.2]} should be more tightly packed than high deviation density {densities[0.5]}, "
                        f"for dataset {dataset}, series impacted {S}, and missing rate {R}. (Center: {center})")


    def test_distribution_missing_percentage_total(self):
        """
        Test if the size of the missing percentage in a contaminated time series meets the expected number defined by the user.
        """
        datasets = ["drift", "chlorine", "eeg-alcohol"]
        series_impacted = [0.4, 0.8]
        missing_rates = [0.2, 0.6]
        offset, std_dev = 0.1, 0.2

        for dataset in datasets:
            ts_1 = TimeSeries()
            ts_1.load_series(utils.search_path(dataset))
            for series_sel in series_impacted:
                for missing_rate in missing_rates:
                    D = self.load_probabilities(ts_1.data, std_dev=std_dev)
                    ts_contaminate = GenGap.distribution(input_data=ts_1.data, rate_series=missing_rate, rate_dataset=series_sel, probabilities_list=D, offset=offset, seed=True)
                    N, M = ts_contaminate.data.shape

                    nbr_series_contaminated = 0
                    for inx in range(ts_contaminate.shape[1]):
                        current_series = ts_contaminate[:, inx]

                        if np.isnan(current_series).any():
                            nbr_series_contaminated = nbr_series_contaminated + 1

                            num_missing_values = np.isnan(current_series).sum()
                            expected_num_missing = int(N * missing_rate)

                            print(f"\t\tNUMBR OF VALUES for series #{inx} : {num_missing_values}")
                            print(f"\t\tEXPECTED VALUES for series #{inx} : {expected_num_missing}\n")

                            self.assertEqual(num_missing_values, expected_num_missing, msg=f"Dataset '{dataset}', Series Index {current_series}: " f"Expected {expected_num_missing} missing values, but found {num_missing_values}.")

                            percentage = (expected_num_missing / N) * 100
                            print(f"\t\tPERCENTAGE VALUES for series #{inx} : {percentage}")
                            print(f"\t\tEXPECTED % VALUES for series #{inx} : {missing_rate * 100}\n")

                            self.assertAlmostEqual(percentage, missing_rate * 100, delta=1, msg=f"Dataset '{dataset}': Expected {missing_rate * 100}%, but found {percentage}%.")

                            print("\n\n\n===============================\n\n")

                    expected_nbr_series = int(np.ceil(M * series_sel))
                    self.assertEqual(nbr_series_contaminated, expected_nbr_series,
                        msg=f"Dataset '{dataset}': Expected {expected_nbr_series} contaminated series, "
                            f"but found {nbr_series_contaminated}."
                    )

                    print("NUMBR OF SERIES : ", nbr_series_contaminated)
                    print("EXPECTED SERIES : ", expected_nbr_series, "\n")
