# ===============================================================================================================
# SOURCE: https://github.com/LMZZML/PriSTI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/10184808
# ===============================================================================================================

import argparse
import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
from imputegap.wrapper.AlgoPython.PriSTI.utils import get_randmask, get_block_mask
device = "cuda" if torch.cuda.is_available() else "cpu"

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


class ImputeGAPDataset(Dataset):
    def __init__(self, ts_m, eval_length=24, mode="train", val_len=0.1, test_len=0.2, missing_pattern='block', is_interpolate=False, target_strategy='random', multivariate=False, verbose=True, deep_verbose=False):
        self.ts_m = ts_m
        self.eval_length = eval_length
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.verbose = verbose
        self.deep_verbose = deep_verbose


        #path = "./data/pems_bay/pems_meanstd.pk"
        #with open(path, "rb") as f:
        #    self.train_mean, self.train_std = pickle.load(f)

        # mean and std for each column, ignoring NaNs
        self.train_mean = np.nanmean(ts_m, axis=0)
        self.train_std = np.nanstd(ts_m, axis=0)

        # create data for batch
        self.use_index = []
        self.cut_length = []

        #df = pd.read_hdf("./data/pems_bay/pems_bay.h5")

        ts_m[np.isnan(ts_m)] = 0.
        df = pd.DataFrame(ts_m)

        if self.deep_verbose:
            print(f"{np.array(df).shape = }")

        ob_mask = (df.values != 0.).astype('uint8')  # all NaNs values from the contamination

        if self.deep_verbose:
            print(f"\n{ob_mask = }")
            print(f"{ob_mask.shape = }\n")

        SEED = 9101112
        self.rng = np.random.default_rng(SEED)
        if missing_pattern is None:
            eval_mask = (ts_m == 0.0).astype(int)

        elif missing_pattern == 'block':
            eval_mask = sample_mask(shape=(ts_m.shape[0], ts_m.shape[1]), p=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=self.rng)

        elif missing_pattern == 'point':
            eval_mask = sample_mask(shape=(ts_m.shape[0], ts_m.shape[1]), p=0., p_noise=0.25, max_seq=12, min_seq=12 * 4, rng=self.rng)

        if self.deep_verbose:
            print(f"\n{eval_mask = }")
            print(f"{eval_mask.shape = }\n")

        gt_mask = (1-(eval_mask | (1-ob_mask))).astype('uint8')

        if self.deep_verbose:
            print(f"\n{gt_mask = }")
            print(f"{gt_mask.shape = }\n")

        val_start = int((1 - val_len - test_len) * len(df))
        test_start = int((1 - test_len) * len(df))+1

        if self.deep_verbose:
            print(f"\n{val_start = }")
            print(f"{test_start = }\n")

        #c_data = ((df.fillna(0).values - self.train_mean) / self.train_std) * ob_mask
        c_data = df.fillna(0).values * ob_mask

        if self.deep_verbose:
                print(f"\n{c_data = }")
                print(f"{c_data.shape = }\n")

        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = ob_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
        elif mode == 'test':
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = c_data[test_start:]
        elif mode == 'imputegap':
            self.observed_mask = ob_mask
            self.gt_mask = gt_mask
            self.observed_data = c_data

        if self.verbose: # prints
            print(f"{mode = }: {self.observed_mask.shape = }")
            if self.deep_verbose:
                print(f"\n{mode = }: {self.observed_mask = }\n")
            print(f"{mode = }: {self.gt_mask.shape = }")
            if self.deep_verbose:
                print(f"\n{mode = }: {self.gt_mask = }\n")
            print(f"{mode = }: {self.observed_data.shape = }")
            if self.deep_verbose:
                print(f"\n{mode = }: {self.observed_data = }\n")

        if multivariate:
            current_length = len(self.observed_mask) // eval_length
        else:
            current_length = len(self.observed_mask) - eval_length  + 1

        if self.verbose:
           print(f"{mode = }: {current_length = }\n")

        if mode == "test" or multivariate:
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(0, 0 + eval_length * n_sample, eval_length)

            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [eval_length - len(self.observed_data) % eval_length]

        elif mode != "test":
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

        if self.deep_verbose:
            print(f"\n{self.use_index = }")
            print(f"{self.cut_length = }")

    def __getitem__(self, org_index):

        index = self.use_index[org_index]

        ob_data = self.observed_data[index: index + self.eval_length]

        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy)
            else:
                cond_mask = get_randmask(ob_mask_t)

        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask
        }

        if self.deep_verbose:
            print(f"\n{np.array(ob_data).shape = }")
            print(f"\n{np.array(ob_data) = }")

        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(ts_m, seq_length, batch_size, device, val_len=0.1, test_len=0.2, missing_pattern='block', is_interpolate=False, num_workers=4, target_strategy='random', multivariate=False, verbose=True, deep_verbose=False):

    if verbose:
        print(f"\nget_dataloader: {ts_m.shape = }, {seq_length = }, {batch_size = }, {multivariate =}, {device = }, {val_len = }, {test_len = }, {missing_pattern = }, {is_interpolate = }, {num_workers = }, {target_strategy = }\n")

    dataset = ImputeGAPDataset(ts_m, mode="train", eval_length=seq_length, val_len=val_len, test_len=test_len, missing_pattern=missing_pattern, is_interpolate=is_interpolate, target_strategy=target_strategy, multivariate=multivariate, verbose=verbose, deep_verbose=deep_verbose)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False )

    dataset_test = ImputeGAPDataset(ts_m, mode="test", eval_length=seq_length, val_len=val_len, test_len=test_len, missing_pattern=missing_pattern, is_interpolate=is_interpolate, target_strategy=target_strategy, multivariate=multivariate, verbose=verbose, deep_verbose=deep_verbose)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset_valid = ImputeGAPDataset(ts_m, mode="valid", eval_length=seq_length, val_len=val_len, test_len=test_len, missing_pattern=missing_pattern, is_interpolate=is_interpolate, target_strategy=target_strategy, multivariate=multivariate, verbose=verbose, deep_verbose=deep_verbose)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    dataset = ImputeGAPDataset(ts_m, mode="imputegap", eval_length=seq_length, val_len=0, test_len=0, missing_pattern=None, is_interpolate=is_interpolate, target_strategy=None, multivariate=multivariate, verbose=verbose, deep_verbose=deep_verbose)
    imputegap_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader, test_loader, scaler, mean_scaler, imputegap_loader

