import os
import unittest

import matplotlib
import numpy as np

from imputegap.manager._manager import TimeSeriesGAP


class TestContamination(unittest.TestCase):

    def test_mcar_selection(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeriesGAP("./dataset/test.txt")
        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=2, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    series_check = impute_gap.format_selection(series_selected)

                    impute_gap.print()

                    check_nan_series = False

                    for series, data in enumerate(impute_gap.contaminated_ts):
                        if str(series) in series_check:
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
        impute_gap = TimeSeriesGAP("./dataset/test.txt")
        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=2, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    impute_gap.print()

                    if np.isnan(impute_gap.contaminated_ts[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)

    def test_mcar_selection_chlorine(self):
        """
        the goal is to test if only the selected values are contaminated
        """
        impute_gap = TimeSeriesGAP("./dataset/chlorine.txt")
        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=10, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    series_check = impute_gap.format_selection(series_selected)

                    impute_gap.print()

                    check_nan_series = False

                    for series, data in enumerate(impute_gap.contaminated_ts):
                        if str(series) in series_check:
                            if np.isnan(data).any():
                                check_nan_series = True
                        else:
                            if np.isnan(data).any():
                                check_nan_series = False
                                break
                            else:
                                check_nan_series = True

                    self.assertTrue(check_nan_series, True)

    def test_mcar_position_chlorine(self):
        """
        the goal is to test if the starting position is always guaranteed
        """
        impute_gap = TimeSeriesGAP("./dataset/chlorine.txt")
        series_selection = [["1", "3", "4"], ["-2"], ["+2"], ["*"]]
        missing_rates = [0.1, 0.2, 0.4, 0.8]
        seeds_start, seeds_end = 42, 43

        ten_percent_index = int(impute_gap.ts.shape[1] * 0.1)

        for seed_value in range(seeds_start, seeds_end):
            for series_selected in series_selection:
                for missing_rate in missing_rates:

                    impute_gap.contamination_mcar(missing_rate=missing_rate, block_size=10, starting_position=0.1,
                                                  series_selected=series_selected, use_seed=True, seed=seed_value)

                    impute_gap.print()

                    if np.isnan(impute_gap.contaminated_ts[:, :ten_percent_index]).any():
                        check_position = False
                    else:
                        check_position = True

                    self.assertTrue(check_position, True)

    def test_loading_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        if not hasattr(matplotlib.get_backend(), 'required_interactive_framework'):
            matplotlib.use('TkAgg')

        impute_gap = TimeSeriesGAP("./dataset/chlorine.txt")
        filename = "./assets"

        impute_gap.plot("contaminate", "test", filename, 5, (16, 8), False)

        self.assertTrue(os.path.exists(filename))