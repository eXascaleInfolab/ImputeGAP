import os
import unittest
import numpy as np
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

class TestLoading(unittest.TestCase):

    def test_loading_set(self):
        """
        Verify if the manager of a dataset is working
        """
        utils.display_title()

        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test.txt"), normalizer=None)

        self.assertEqual(ts_1.data.shape, (25, 10))
        self.assertEqual(ts_1.data[1, 0], 2.5)
        self.assertEqual(ts_1.data[0, 1], 0.5)

    def test_loading_chlorine(self):
        """
        Verify if the manager of a dataset is working
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"), normalizer=None)

        self.assertEqual(ts_1.data.shape, (1000, 50))
        self.assertEqual(ts_1.data[1, 0], 0.0154797)
        self.assertEqual(ts_1.data[0, 1], 0.0236836)

    def test_loading_plot(self):
        """
        Verify if the manager of a dataset is working
        """
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test.txt"))

        to_save = "./assets"
        file_path = ts_1.plot(input_data=ts_1.data, nbr_series=5, nbr_val=100, size=(16, 8), save_path=to_save, display=False)

        self.assertTrue(os.path.exists(file_path))

    def test_loading_normalization_min_max(self):
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("test.txt"))
        ts_1.normalize(normalizer="min_max")

        assert np.isclose(np.min(ts_1.data), 0), f"Min value after Min-Max normalization is not 0: {np.min(ts_1.normalized_ts)}"
        assert np.isclose(np.max(ts_1.data), 1), f"Max value after Min-Max normalization is not 1: {np.max(ts_1.normalized_ts)}"

    def test_loading_normalization_z_score(self):
        normalized = TimeSeries()
        normalized.load_series(utils.search_path("test.txt"))
        normalized.normalize()

        mean = np.mean(normalized.data)
        std_dev = np.std(normalized.data)

        assert np.isclose(mean, 0, atol=1e-7), f"Mean after Z-score normalization is not 0: {mean}"
        assert np.isclose(std_dev, 1, atol=1e-7), f"Standard deviation after Z-score normalization is not 1: {std_dev}"

    def test_loading_normalization_min_max_lib(self):
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))
        ts_1.normalize(normalizer="min_max")

        ts_2 = TimeSeries()
        ts_2.load_series(utils.search_path("chlorine"))
        ts_2.normalize(normalizer="m_lib")

        assert np.allclose(ts_1.data, ts_2.data, atol=1e-7)

    def test_loading_normalization_z_score_lib(self):
        ts_1 = TimeSeries()
        ts_1.load_series(utils.search_path("chlorine"))
        ts_1.normalize()

        ts_2 = TimeSeries()
        ts_2.load_series(utils.search_path("chlorine"))
        ts_2.normalize(normalizer="z_lib")

        assert np.allclose(ts_1.data, ts_2.data)

