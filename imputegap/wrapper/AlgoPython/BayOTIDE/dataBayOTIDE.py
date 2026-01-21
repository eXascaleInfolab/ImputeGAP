import os

import numpy as np
from scipy.io import loadmat


def generate_data_bayotide(ts_m, replace="neg", ts_r=0.2, val_r=0.1, verbose=True, deep_verbose=False, replicat=False):

    sub_tensor = ts_m
    masker = ts_m

    if replicat:
        print("\nTesting mode for results verification...")
        here = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(here, "data/raw/tensor.mat")
        tensor = loadmat(filepath)
        tensor = tensor['tensor']
        print(f"{tensor.shape = }")
        print(f"{np.isnan(tensor).any() = }")
        sub_tensor = tensor.reshape(214, 61 * 144)
        print(f"{sub_tensor.shape = }")
        sub_tensor = sub_tensor[:, -501:-1]
        masker = sub_tensor
        print(sub_tensor.shape)

    if verbose:
        print(f"preprocessing...\n\nLoading and preprocessing of the data for BayOTIDE\tloaded data shape : {sub_tensor.shape}\n")

    if replace == "mean":
        global_mean = np.nanmean(sub_tensor)
        sub_tensor = np.nan_to_num(sub_tensor, nan=global_mean)
    else:
        sub_tensor = np.nan_to_num(sub_tensor, nan=-999999)

    data_save = {}
    data_save['ndims'] = sub_tensor.shape
    data_save['raw_data'] = sub_tensor
    data_save['data'] = []
    data_save['time_uni'] = np.linspace(0, 1, sub_tensor.shape[1])

    def generate_random_mask(data, shape, drop_rate=0.2, valid_rate=0.1):
        """
        train_ratio: 1-valid_rate-drop_rate
        test_ratio: drop_rate
        valid_ratio: valid_rate
        """
        N, T = shape

        mask_train_list = []
        mask_test_list = []
        mask_valid_list = []

        for t in range(T):
            mask = np.random.rand(N)

            nan_pos = np.isnan(data[:, t])

            mask_train = np.where(mask > drop_rate + valid_rate, 1, 0)
            mask_test = np.where(mask < drop_rate, 1, 0)
            mask_valid = np.where((mask > drop_rate) & (mask < drop_rate + valid_rate), 1, 0)

            mask_train[nan_pos] = 0
            mask_valid[nan_pos] = 0
            mask_test[nan_pos] = 0

            mask_train_list.append(mask_train)
            mask_test_list.append(mask_test)
            mask_valid_list.append(mask_valid)

        mask_train = np.stack(mask_train_list, axis=1)
        mask_test = np.stack(mask_test_list, axis=1)
        mask_valid = np.stack(mask_valid_list, axis=1)

        return mask_train, mask_test, mask_valid

    fold = 5
    drop_rate = ts_r
    valid_rate = val_r

    for i in range(fold):
        mask_train, mask_test, mask_valid = generate_random_mask(masker, masker.shape, drop_rate, valid_rate)
        data_save['data'].append({'mask_train': mask_train, 'mask_test': mask_test, 'mask_valid': mask_valid})

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset_saving_dir = os.path.join(here, "imputegap_assets/data")
    dataset = os.path.join(dataset_saving_dir, "bayotide.npy")
    os.makedirs(dataset_saving_dir, exist_ok=True)
    np.save(dataset, data_save)

    return dataset