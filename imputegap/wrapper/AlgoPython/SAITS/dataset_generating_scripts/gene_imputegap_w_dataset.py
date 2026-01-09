# ===============================================================================================================
# SOURCE: https://github.com/WenjieDu/SAITS
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://doi.org/10.1016/j.eswa.2023.119619
# ===============================================================================================================

"""
The script for generating PhysioNet-2012 dataset.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import imputegap.tools.utils as utils_imp

sys.path.append("..")
from imputegap.wrapper.AlgoPython.SAITS.dataset_generating_scripts.data_processing_utils import  (window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)


def process_each_set(X):
    # gene labels, y
    has_nan = np.array([np.isnan(sample).any() for sample in X])
    y = (has_nan).astype(int).reshape(-1, 1)  # 1 = no NaN, 0 = has NaN
    return X, y, None


def keep_only_features_to_normalize(all_feats, to_remove):
    for i in to_remove:
        all_feats.remove(i)
    return all_feats


def sait_loader_w_imputegap(incomp_data, seq_len, features, here, sliding_windows=1, tr_ratio=0.7, norm=False, shuffle=False, artificial_missing_rate=0.1, verbose=True, deep_verbose=False):

    if verbose:
        print(f"\n\tpreparation of the data with truncation window strategy, conversion to h5 format:\n\t{seq_len=}, {features=}, {norm=}, {shuffle=}, {artificial_missing_rate=}...")

    matrix_in = np.copy(incomp_data)

    saving_path = "datasets"
    dataset_name = "saits"

    ts_r, val_r = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=True)
    #ts_m = utils_imp.dataset_add_dimensionality(matrix_in, seq_length=seq_len, three_dim=True, verbose=True)

    dataset_saving_dir = os.path.join(here, saving_path, dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    all_sample_ids = np.arange(matrix_in.shape[0])
    nsamples = len(all_sample_ids)

    n_train = round(nsamples * tr_ratio)
    n_test = round(nsamples * ts_r)
    n_val = nsamples - n_train - n_test

    train_set_ids = all_sample_ids[:n_train]
    val_set_ids = all_sample_ids[n_train:n_train + n_test]
    test_set_ids = all_sample_ids[n_train + n_test:]

    features_list = list(range(features))
    # separate the sets in df based on train_set_ids / test_set_ids / val_set_ids

    ts_m_df = pd.DataFrame(matrix_in)
    train_set = ts_m_df.iloc[train_set_ids, :]  # rows = samples
    val_set = ts_m_df.iloc[val_set_ids, :]
    test_set = ts_m_df.iloc[test_set_ids, :]

    if verbose:
        print(f"\tThere are total {len(train_set_ids)} timestamps in train set, for {train_set.shape[1]} features.")
        print(f"\tThere are total {len(val_set_ids)} timestamps in val set, for {val_set.shape[1]} features.")
        print(f"\tThere are total {len(test_set_ids)} timestamps in test set, for {test_set.shape[1]} features.\n")

    # standardization
    if norm:
        scaler = StandardScaler()
        train_set_X = scaler.fit_transform(train_set.loc[:, features_list])
        val_set_X = scaler.transform(val_set.loc[:, features_list])
        test_set_X = scaler.transform(test_set.loc[:, features_list])
    else:
        train_set_X = train_set.loc[:, features_list]
        val_set_X = val_set.loc[:, features_list]
        test_set_X = test_set.loc[:, features_list]

    if deep_verbose:
        print(f"{train_set_X.shape = }")
        print(f"{val_set_X.shape = }")
        print(f"{test_set_X.shape = }")

    train_set_X = window_truncate(train_set_X, seq_len, sliding_len=sliding_windows, verbose=verbose, deep_verbose=False)
    val_set_X = window_truncate(val_set_X, seq_len, sliding_len=sliding_windows, verbose=verbose, deep_verbose=False)
    test_set_X = window_truncate(test_set_X, seq_len, sliding_len=sliding_windows, verbose=verbose, deep_verbose=False)

    # add missing values in train set manually
    if artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)

    train_set_dict = add_artificial_mask(train_set_X, artificial_missing_rate, "train")
    val_set_dict = add_artificial_mask(val_set_X, artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(test_set_X, artificial_missing_rate, "test")

    processed_data = { "train": train_set_dict, "val": val_set_dict, "test": test_set_dict, }
    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])

    if verbose:
        print(f"\n\t{train_sample_num} windows in train set of length {seq_len}\n\t{val_sample_num} windows in val set of length {seq_len}\n\t{test_sample_num} windows in test set of length {seq_len}\n\n")

    print(f"{dataset_saving_dir = }")
    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)

    if norm:
        with open(os.path.join(dataset_saving_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"\n\tAll done. Saved to: {dataset_saving_dir}\n\n")

    return dataset_saving_dir, (len(train_set_ids), len(val_set_ids), len(test_set_ids), sliding_windows)

