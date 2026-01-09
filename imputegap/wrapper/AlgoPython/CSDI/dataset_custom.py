# ===============================================================================================================
# SOURCE: https://github.com/ermongroup/CSDI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://neurips.cc/virtual/2021/poster/26846
# ===============================================================================================================

from torch.utils.data import DataLoader, Dataset
import numpy as np
import imputegap.tools.utils as utils_imp


class DatasetCustom(Dataset):
    def __init__(self, datatype, mode="train", seq_len=16, tr_ratio=0.7, ts_m=None, use_index_list=None, missing_ratio=0.1, sliding_windows=1, seed=1, verbose=True):

        self.seq_len = seq_len
        self.missing_ratio = missing_ratio

        #self.mean_data = np.nanmean(ts_m, axis=0)  # shape (D,)
        #self.std_data = np.nanstd(ts_m, axis=0, ddof=0)  # shape (D,)

        self.observed_values = np.array(ts_m)
        self.observed_masks = ~np.isnan(self.observed_values)
        self.observed_values = np.nan_to_num(self.observed_values)
            
        #self.main_data = (self.main_data - self.mean_data) / self.std_data
        #self.main_data = np.nan_to_num(self.main_data)
        #self.main_data = self.main_data.astype("float32")

        np.random.seed(seed)
        flat_mask = self.observed_masks.flatten()  # for element wise masking as univariate data
        obs_indices = np.where(flat_mask)[0]
        miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * self.missing_ratio), replace=False)
        flat_mask[miss_indices] = False
        self.gt_masks = flat_mask.reshape(self.observed_masks.shape)

        self.observed_values = utils_imp.window_truncation(self.observed_values, seq_len=self.seq_len, stride=sliding_windows, info=mode+" |", verbose=verbose)
        self.observed_masks = utils_imp.window_truncation(self.observed_masks, seq_len=self.seq_len, stride=sliding_windows, info=mode+" |", verbose=verbose)
        self.gt_masks = utils_imp.window_truncation(self.gt_masks, seq_len=self.seq_len, stride=sliding_windows, info=mode+" |", verbose=verbose)
        if verbose:
            print("\n")

        ts_r, val_r = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=False)
        n_samples = ts_m.shape[0]
        total_length = len(self.observed_values)

        self.test_length= round(total_length*ts_r)
        self.valid_length= round(total_length*val_r)

        if mode == 'train': 
            start = 0
            end = total_length - self.seq_len - self.valid_length - self.test_length
            self.use_index_list = np.arange(start, end)
        elif mode == 'valid': #valid
            start = total_length - self.seq_len - self.valid_length - self.test_length
            end = total_length - self.seq_len - self.test_length
            self.use_index_list = np.arange(start, end)
        elif mode == 'test': #test
            start = total_length - self.seq_len - self.test_length
            end = total_length - self.seq_len
            self.use_index_list = np.arange(start, end)
        else:
            start = 0
            end = len(self.observed_values)
            self.use_index_list = np.arange(end)

        #print(f"\t{mode}: {start = } - {end = }\t{self.use_index_list = }\n")

    def __len__(self):
        return len(self.use_index_list)

    def __getitem__(self, idx):
        index = self.use_index_list[idx]

        s_begin = index
        s_end = s_begin + self.seq_len

        s = {
            "observed_data": self.observed_values[s_begin],
            "observed_mask": self.observed_masks[s_begin],
            "gt_mask": self.gt_masks[s_begin],
            'timepoints': np.arange(self.seq_len) * 1.0,
            'feature_id': np.arange(self.observed_values.shape[1]) * 1.0,
            "index": (s_begin),
        }

        #print(f"{self.observed_values[index].shape = } | {idx = }")
        #print(f"{self.observed_masks[index].shape = }")
        #print(f"{self.gt_masks[index].shape = }")

        return s


    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(datatype=None, device=None, batch_size=8, seq_len=36, ts_m=None, tr_ratio=0.7, missing_ratio=0.1, shuffle=True, normalize=True, sliding_windows=1, features=None, num_workers=0, seed=1, verbose=True):

    imputegap_dataset = DatasetCustom(datatype, mode='imputegap', seq_len=seq_len, ts_m=ts_m, tr_ratio=1, missing_ratio=missing_ratio, sliding_windows=sliding_windows, seed=seed, verbose=verbose)
    imputegap_loader = DataLoader(imputegap_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset = DatasetCustom(datatype, mode='train', seq_len=seq_len, ts_m=ts_m, tr_ratio=tr_ratio, missing_ratio=missing_ratio, sliding_windows=sliding_windows, seed=seed, verbose=verbose)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_dataset = DatasetCustom(datatype, mode='valid', seq_len=seq_len, ts_m=ts_m, tr_ratio=tr_ratio, missing_ratio=missing_ratio, sliding_windows=sliding_windows, seed=seed, verbose=verbose)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = DatasetCustom(datatype, mode='test', seq_len=seq_len, ts_m=ts_m, tr_ratio=tr_ratio, missing_ratio=missing_ratio, sliding_windows=sliding_windows, seed=seed, verbose=verbose)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, imputegap_loader, None, None