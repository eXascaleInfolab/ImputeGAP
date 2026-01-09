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
from sklearn.preprocessing import StandardScaler
import pickle
import imputegap.tools.utils as utils_imp

sys.path.append("..")
from imputegap.wrapper.AlgoPython.SAITS.dataset_generating_scripts.data_processing_utils import (add_artificial_mask, saving_into_h5, )


def process_each_set(X):
    # gene labels, y
    has_nan = np.array([np.isnan(sample).any() for sample in X])
    y = (has_nan).astype(int).reshape(-1, 1)  # 1 = no NaN, 0 = has NaN
    return X, y, None


def keep_only_features_to_normalize(all_feats, to_remove):
    for i in to_remove:
        all_feats.remove(i)
    return all_feats


def sait_loader_imputegap(incomp_data, seq_len, here, tr_ratio=0.9, norm=False, shuffle=True, artificial_missing_rate=0.1, verbose=True, deep_verbose=False, replicat=False):

    if verbose:
        print(f"\n\tpreparation of the data with sample strategy, conversion to h5 format:\n\t{seq_len=}, {norm=}, {shuffle=}, {artificial_missing_rate=}...")

    matrix_in = np.copy(incomp_data)

    saving_path = "datasets"
    dataset_name = "saits"

    # standardization
    #if norm or replicat:
    #    scaler = StandardScaler()
    #    matrix_in = scaler.fit_transform(matrix_in)

    ts_r, val_r = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=True)
    ts_m = utils_imp.dataset_add_dimensionality(matrix_in, seq_length=seq_len, three_dim=True, verbose=True)

    dataset_saving_dir = os.path.join(here, saving_path, dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    all_sample_ids = np.arange(ts_m.shape[0])
    nsamples = len(all_sample_ids)


    #train_set_ids, test_set_ids = train_test_split(all_sample_ids, train_size=tr_ratio, shuffle=True)
    #train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=val_r, shuffle=False)

    n_train = round(nsamples * tr_ratio)
    n_test = round(nsamples * ts_r)
    train_set_ids = all_sample_ids[:n_train]
    val_set_ids = all_sample_ids[n_train:n_train + n_test]
    test_set_ids = all_sample_ids[n_train + n_test:]

    if verbose:
        print(f"\tThere are total {len(train_set_ids)} samples in train set.")
        print(f"\tThere are total {len(val_set_ids)} samples in val set.")
        print(f"\tThere are total {len(test_set_ids)} samples in test set.\n")

    if deep_verbose:
        print(f"\n{train_set_ids = }")
        print(f"{val_set_ids = }")
        print(f"{test_set_ids = }\n")

    # separate the sets in df based on train_set_ids / test_set_ids / val_set_ids
    train_set = ts_m[train_set_ids]
    val_set = ts_m[val_set_ids]
    test_set = ts_m[test_set_ids]

    if deep_verbose:
        print(f"\n\n{train_set = }")
        print(f"\n{val_set = }")
        print(f"\n{test_set = }\n\n")

    # standardization
    if norm and not replicat:
        scaler = StandardScaler()
        train_set = scaler.fit_transform(train_set)
        val_set = scaler.fit_transform(val_set)
        test_set = scaler.fit_transform(test_set)

    train_set_X, train_set_y, _ = process_each_set(train_set)
    val_set_X, val_set_y, _ = process_each_set(val_set)
    test_set_X, test_set_y, _ = process_each_set(test_set)

    if deep_verbose:
        print(f"\n\n{train_set_y = } {train_set_y.shape = }")
        print(f"\n{val_set_y = }  {val_set_y.shape = }")
        print(f"\n{test_set_y = }  {test_set_y.shape = }\n\n")

    train_set_dict = add_artificial_mask(train_set_X, artificial_missing_rate, "train")
    val_set_dict = add_artificial_mask(val_set_X, artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(test_set_X, artificial_missing_rate, "test")

    if deep_verbose:
        print(f"\n\n{train_set_dict = }")
        print(f"\n{val_set_dict = }")
        print(f"\n{test_set_dict = }\n\n")

    X_val = val_set_dict["X"]
    X_hat_val = val_set_dict["X_hat"]
    missing_mask_val = val_set_dict["missing_mask"]
    indicating_mask_tval = val_set_dict["indicating_mask"]

    X_test = test_set_dict["X"]
    X_hat_test= test_set_dict["X_hat"]
    missing_mask_test = test_set_dict["missing_mask"]
    indicating_mask_test = test_set_dict["indicating_mask"]

    if deep_verbose:
        print(f"\n\n{X_val = }")
        print(f"\n{X_hat_val = }")
        print(f"\n{missing_mask_val = }")
        print(f"\n{indicating_mask_tval = }\n\n")

        print(f"\n\n{X_test = }")
        print(f"\n{X_hat_test = }")
        print(f"\n{missing_mask_test = }")
        print(f"\n{indicating_mask_test = }\n\n")

    if verbose:
        print(f'\n\tIn val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}')
        print(f'\tIn test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}\n')

    train_set_dict["labels"] = train_set_y
    val_set_dict["labels"] = val_set_y
    test_set_dict["labels"] = test_set_y

    processed_data = {"train": train_set_dict, "val": val_set_dict, "test": test_set_dict, }

    total_sample_num = 0
    total_positive_num = 0
    for set_name, rec in zip(["train", "val", "test"], [train_set_dict, val_set_dict, test_set_dict]):
        total_sample_num += len(rec["labels"])
        total_positive_num += rec["labels"].sum()

    missing_part = np.isnan(ts_m)

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)

    if norm:
        with open(os.path.join(dataset_saving_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"\n\tDataset overall missing rate of original feature vectors (without any artificial mask):{(missing_part.sum() / missing_part.shape[0] / missing_part.shape[1]):.3f}")
        print(f"\n\tAll done. Saved to: {dataset_saving_dir}\n\n")

    return dataset_saving_dir, None

