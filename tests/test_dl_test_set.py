import unittest

import numpy as np
from imputegap.recovery.contamination import GenGap

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


class TestDLTestSet(unittest.TestCase):

    def test_dl_test(self):
        """
        the goal is to test if only the simple imputation with ST-MVL has the expected outcome
        """

        ts = TimeSeries()
        print(f"Imputation algorithms : {ts.algorithms}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("chlorine"))

        # contaminate the time series
        ts_m = GenGap.mcar(ts.data)

        ts_m = ts_m.T
        ts.data = ts.data.T

        M, N = ts_m.shape

        # ==================================================================================================================
        tr_ratio, artificial_training_drop = 0.8, 0.2
        nan_replacement = -99999
        cont_data_matrix = ts_m.copy()
        miss = np.sum(np.isnan(cont_data_matrix))
        verbose = True
        seed = 42
        offset = 0.05

        # Step 1: Check original missing ratio
        original_missing_ratio = utils.get_missing_ratio(cont_data_matrix)
        print(f"OK :) > original_missing_ratio = {original_missing_ratio}")
        assert abs(
            original_missing_ratio - 0.04) < 0.005, f"Unexpected original missing ratio: {original_missing_ratio}"

        # Step 2: Apply testing set preparation
        cont_data_matrix, new_mask, error = utils.prepare_testing_set(incomp_m=cont_data_matrix, original_missing_ratio=original_missing_ratio, tr_ratio=tr_ratio, verbose=verbose)

        # Step 3: Diagnostics
        new_missing_ratio = utils.get_missing_ratio(cont_data_matrix)
        num_missing = np.sum(np.isnan(cont_data_matrix))
        mask_sum = np.sum(new_mask)

        print(f"OK :) > new_missing_ratio = {new_missing_ratio}")
        print(f"OK :) > np.sum(np.isnan(cont_data_matrix)) = {num_missing}")
        print(f"OK :) > np.sum(new_mask) = {mask_sum}")

        # Step 4: Assertions
        expected_missing_ratio = 1 - tr_ratio
        expected_ts = int(M*N*expected_missing_ratio)  # should match new_mask sum

        assert abs(new_missing_ratio - expected_missing_ratio) < 0.01, f"Unexpected new missing ratio: {new_missing_ratio}"
        assert num_missing == expected_ts, f"Mismatch in NaN count: {num_missing} != {expected_ts}"
        assert mask_sum == expected_ts, f"Mismatch in NaN count: {mask_sum} != {expected_ts}"
        assert mask_sum > 0, "new_mask should contain at least some test-time NaNs"
        assert not error, "Unexpected format error"

        cont_data_matrix = utils.prevent_leakage(cont_data_matrix, new_mask, nan_replacement, verbose)

        num_missing = np.sum(cont_data_matrix == nan_replacement)

        print(f"OK :) > np.sum(cont_data_matrix == -99999)) = {num_missing}")
        assert num_missing == expected_ts, f"Mismatch in NaN count after leakage: {num_missing} != {expected_ts}"

        # ==================================================================================================================
        mask_test, mask_valid, nbr_nans = utils.split_mask_bwt_test_valid(cont_data_matrix, test_rate=1, valid_rate=0, nan_val=nan_replacement, verbose=verbose, seed=seed)

        num_mask_test = np.sum(mask_test)
        print(f"OK :) > num_mask_test = {num_mask_test}")
        assert num_mask_test == expected_ts, f"Mismatch test set count: {mask_sum} != {expected_ts}"

        num_val_test = np.sum(mask_valid)
        print(f"OK :) > num_val_test = {num_val_test}")
        assert num_val_test == 0, f"Mismatch val set count: {num_val_test} != {0}"


        mask_train = utils.generate_random_mask(gt=cont_data_matrix, mask_test=mask_test, mask_valid=mask_valid, droprate=artificial_training_drop, offset=offset, series_like=False, verbose=verbose, seed=seed)

        exp_tr = int(M * N * artificial_training_drop)-miss
        num_tr = np.sum(mask_train)
        print(f"OK :) > num_tr = {num_tr}")
        assert num_tr == exp_tr, f"Mismatch train set count: {num_tr} != {exp_tr}"


        print(f"{mask_train[0] = }")
        print(f"{mask_test[0] = }")
        print(f"{cont_data_matrix[0] = }")

        D = int(offset*N)
        print(f"{D = }")
        assert np.all(mask_train[0] == 0), f"mask_train[0] contains non-zero values: {mask_train[0]}"
        assert np.all(cont_data_matrix[0, D:] == nan_replacement), f"cont_data_matrix[0] contains nan_replacement values: {nan_replacement}"
        assert np.all(mask_test[0, D:] == 1), f"mask_test[0] contains 1 values"


        print(f"{mask_train[33] = }")
        print(f"{mask_test[33] = }")
        print(f"{cont_data_matrix[33] = }")

        assert np.any(mask_train[33] == 1), f"mask_train[33] contains 1: {mask_train[33]}"
        assert np.all(cont_data_matrix[33] != nan_replacement), f"cont_data_matrix[33] contains none nan_replacement values: {nan_replacement}"
        assert np.all(mask_test[33] != 1), f"mask_test[0] contains 0 values"

        # Print diagnostics
        print(f"{mask_train[45] = }")
        print(f"{mask_test[45] = }")
        print(f"{cont_data_matrix[45] = }")

        # --- Check 1: mask_train and mask_test do not have overlapping 1s ---
        # Element-wise multiplication will be > 0 if both have 1 at the same position
        overlap = mask_train[45] * mask_test[45]
        assert np.all(overlap == 0), f"mask_train and mask_test overlap at positions: {np.where(overlap == 1)}"

        # --- Check 2: all -99999 values in cont_data_matrix[45] correspond to mask_test[45] == 1 ---

        # Invert mask_test: 1 → 0 and 0 → 1
        m_2 = 1 - mask_test

        # Check that all test-mask positions contain the nan_replacement value
        assert np.all(cont_data_matrix[mask_test == 1] == nan_replacement), \
            "Not all test mask positions have the nan_replacement value"

        # Check that non-test-mask positions do NOT contain the nan_replacement value
        assert np.all(cont_data_matrix[m_2 == 1] != nan_replacement), \
            "Some non-test positions incorrectly contain the nan_replacement value"

        # Check that non-test-mask positions do NOT contain the nan_replacement value
        assert np.all(cont_data_matrix[mask_train == 1] != nan_replacement), \
            "Some non-test positions incorrectly contain the nan_replacement value"

        print(np.all(cont_data_matrix * mask_train != nan_replacement))
        print(np.all(((cont_data_matrix * mask_test) == nan_replacement) | ((cont_data_matrix * mask_test) == 0)))

        # ==================================================================================================================