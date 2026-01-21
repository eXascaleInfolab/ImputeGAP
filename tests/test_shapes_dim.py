import unittest
import numpy as np
from imputegap.tools import utils


class TestShapesDim(unittest.TestCase):

    def test_dimension_window_sample(self):
        for seq_len in [2,3]:
            M = np.arange(1, 101).reshape(10, 10)

            model_sample = utils.dataset_add_dimensionality(M, seq_len, reshapable=True, adding_nans=True, verbose=True, deep_verbose=True)
            model_window = utils.window_truncation(M, seq_len=seq_len, stride=1, info="brits", verbose=True, deep_verbose=True)

            print(f"\n\n\n\n{model_sample.shape = }")
            print(f"{model_sample = }")

            print(f"\n\n\n\n{model_window.shape = }")
            print(f"{model_window = }")

            expect_windows = M.shape[0] - seq_len + 1
            expect_sample = M.shape[0] // seq_len + (1 if M.shape[0] % seq_len != 0 else 0)

            if seq_len == 2:
                assert expect_windows == 9
                assert expect_sample == 5
            else:
                assert expect_windows == 8
                assert expect_sample == 4

            assert model_sample.shape == (expect_sample, seq_len, 10)
            assert model_window.shape == (expect_windows, seq_len, 10)

            rec_sample = utils.dataset_reverse_dimensionality(model_sample, 10, verbose=True)
            print(f"\n\n\n\n{rec_sample = }")
            print(f"{rec_sample.shape = }\n\n")
            assert rec_sample.shape == (10, 10)
            assert np.array_equal(rec_sample, M)

            rec_window =  utils.reconstruction_window_based(preds=model_window, nbr_timestamps=10, verbose=True, deep_verbose=True)
            print(f"\n\n\n\n{rec_window = }")
            print(f"{rec_window.shape = }\n\n")
            assert rec_window.shape == (10, 10)
            assert np.array_equal(rec_window, M)