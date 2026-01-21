import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from imputegap.wrapper.AlgoPython.NuwaTS.utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, percent=10, train_sensors=None, val_sensors=None, test_sensors=None, ts_m = None, reconstruction=None, tr_ratio=None, verbose = None, replicat=None, normalizer=None):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.train_sensors = train_sensors if train_sensors else ['HUFL', 'HULL']
        self.val_sensors = val_sensors if val_sensors else ['MUFL', 'MULL']
        self.test_sensors = test_sensors if test_sensors else ['LUFL', 'LULL']
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.reconstruction = reconstruction
        self.verbose = verbose
        self.__read_data__()

    def __read_data__(self):
        if self.scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        here = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(here, "dataset/ETT-small/ETTh1.csv")

        if self.verbose:
            print(f"Reading-Dataset_ETT_hour {filepath}")
        df_raw = pd.read_csv(os.path.join(filepath))

        if self.set_type == 0:
            df_data = df_raw[self.train_sensors]
        elif self.set_type == 1:
            df_data = df_raw[self.val_sensors]
        else:
            df_data = df_raw[self.test_sensors]

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

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

        self.data_x = data
        self.data_y = data
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
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None, percent=10, train_sensors=None, val_sensors=None, test_sensors=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.train_sensors = train_sensors if train_sensors else ['HUFL', 'HULL']
        self.val_sensors = val_sensors if val_sensors else ['MUFL', 'MULL']
        self.test_sensors = test_sensors if test_sensors else ['LUFL', 'LULL']
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.set_type == 0:
            df_data = df_raw[self.train_sensors]
        elif self.set_type == 1:
            df_data = df_raw[self.val_sensors]
        else:
            df_data = df_raw[self.test_sensors]

        #
        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

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

        self.data_x = data
        self.data_y = data

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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, percent=10, train_sensors=None, val_sensors=None, test_sensors=None, ts_m=None, reconstruction=False, tr_ratio=0.7, verbose=True, replicat=False, normalizer=False):

        # size [seq_len, label_len, pred_len]
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
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
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
        sensor_columns = [col for col in cols]
        num_sensors = len(sensor_columns)

        imputegap_opti = True

        if not imputegap_opti:
            num_sensors_per_set = num_sensors // 3

            self.train_sensors = sensor_columns[:num_sensors_per_set]
            self.val_sensors = sensor_columns[num_sensors_per_set:2 * num_sensors_per_set]
            self.test_sensors = sensor_columns[2 * num_sensors_per_set:3 * num_sensors_per_set]
            self.imputegap_sensors = sensor_columns[:num_sensors]
        else:
            import imputegap.tools.utils as utils_imp

            # ratios
            train_ratio = self.tr_ratio
            test_ratio, val_ratio = utils_imp.sets_splitter_based_on_training(train_ratio, verbose=False)

            # compute sizes (handle rounding so total == num_sensors)
            n_train = int(num_sensors * train_ratio)
            n_val = int(num_sensors * val_ratio)
            n_test = num_sensors - n_train - n_val  # remainder goes to test

            if n_val < 1:
                n_val = 1
                n_train = n_train - 1

            # split indices
            i0 = 0
            i1 = n_train
            i2 = n_train + n_val
            i3 = num_sensors

            self.train_sensors = sensor_columns[i0:i1]
            self.val_sensors = sensor_columns[i1:i2]
            self.test_sensors = sensor_columns[i2:i3]
            self.imputegap_sensors = sensor_columns[:]  # all sensors


        if self.reconstruction:
            df_data = df_raw[self.imputegap_sensors]
        else:
            if self.set_type == 0:
                df_data = df_raw[self.train_sensors]
                if self.verbose:
                    print(f"training set : {df_data.shape=}\n")
            elif self.set_type == 1:
                df_data = df_raw[self.val_sensors]
                if self.verbose:
                    print(f"validation set : {df_data.shape=}\n")
            else:
                df_data = df_raw[self.test_sensors]
                if self.verbose:
                    print(f"testing set : {df_data.shape=}\n")

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

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

        self.data_x = data
        self.data_y = data
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

