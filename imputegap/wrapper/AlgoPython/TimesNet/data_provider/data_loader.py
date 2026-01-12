# ===============================================================================================================
# SOURCE: https://github.com/thuml/TimesNet
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://iclr.cc/virtual/2023/poster/11976
# ===============================================================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from imputegap.wrapper.AlgoPython.TimesNet.utils.timefeatures import time_features
import warnings
from imputegap.wrapper.AlgoPython.TimesNet.utils.augmentation import run_augmentation_single

import imputegap.tools.utils as utils_imp

warnings.filterwarnings('ignore')

#HUGGINGFACE_REPO = "thuml/Time-Series-Library"


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, ts_m=None,reconstruction=False, tr_ratio=0, verbose=False, replicat=True, normalizer=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.reconstruction = reconstruction

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        here = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(here, "dataset/ETT-small/ETTh1.csv")
        print(f"\ntest for {filepath} - Authors results... {self.scale = } - {self.target = }\n")
        df_raw = pd.read_csv(filepath)
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.reconstruction:
            border1 = 0
            border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        mask_np = np.isnan(data)
        self.mask = torch.from_numpy(mask_np.astype(np.int32))
        self.mask = 1 - self.mask

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        mask = self.mask[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq


        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        here = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(here, "dataset/ETT-small/ETTh1.csv")

        df_raw = pd.read_csv(filepath)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        mask_np = np.isnan(data)
        self.mask = torch.from_numpy(mask_np.astype(np.int32))
        self.mask = 1 - self.mask

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        mask = self.mask[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, ts_m=None, reconstruction=False, tr_ratio=0.7, verbose=True, replicat=False, normalizer=False):

        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.ts_m = ts_m
        self.reconstruction = reconstruction
        self.verbose = verbose
        self.is_targeted = False
        self.replicat = replicat
        self.tr_ratio = tr_ratio
        self.__read_data__()

    def __read_data__(self):

        if self.scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        #local_fp = os.path.join(self.root_path, self.data_path)
        #cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        #if os.path.exists(local_fp):
        #    df_raw = pd.read_csv(local_fp)
        #else:
        #    ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
        #    split_name = "train" if "train" in ds else list(ds.keys())[0]
        #df_raw = ds[split_name].to_pandas()

        if not self.replicat:
            data = self.ts_m
            self.target = None

            start_time = pd.Timestamp("2025-01-01 00:00:00")
            date_index = pd.date_range(start=start_time, periods=data.shape[0], freq='H')
            series_cols = [f"series_{i}" for i in range(data.shape[1])]

            df_raw = pd.DataFrame(data, columns=series_cols)
            df_raw.insert(0, "date", date_index)  # put 'date' as first column

            if df_raw.isna().any().any():
                first_nan_col = df_raw.columns[df_raw.isna().any()][0]
            else:
                first_nan_col = df_raw.columns[-1]  # or whatever "XXX" should be

            self.target = first_nan_col

            if self.verbose and self.set_type == 0:
                print(f"\npre-processing... date time of time series values have been generated, and a target has been designated | {self.scale = }\n")
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(here, "dataset/ETT-small/ETTh1.csv")
            print(f"Reading {filepath}")
            df_raw = pd.read_csv(os.path.join(filepath))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')

        if self.target is not None and self.is_targeted: # or self.replicat:
            if self.verbose:
                print(f"target founded in the dataset: {self.target = }")
            cols.remove(self.target)
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            df_raw = df_raw[['date'] + cols]

        if self.reconstruction:
            num_train = int(len(df_raw))
            num_test = 0
            num_vali = 0
        else:
            ts_r, _ = utils_imp.sets_splitter_based_on_training(tr=self.tr_ratio, verbose=False)
            num_train = int(len(df_raw) * self.tr_ratio)
            num_test = int(len(df_raw) * ts_r)
            num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        mask_np = np.isnan(data)
        self.mask = torch.from_numpy(mask_np.astype(np.int32))
        self.mask = 1 - self.mask

        if not self.replicat:
            col_means = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_means, inds[1])

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        mask = self.mask[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

