from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class InformerDataset:
    def __init__(
        self,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        ts_m= None,
        seq_len=48
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = ts_m # "../data/ETTh1.csv"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.mask = None

        # Read data
        self._read_data()

    def _get_borders(self):
        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        self.scaler = StandardScaler()
        #df = pd.read_csv(self.full_file_path_and_name)
        data = self.full_file_path_and_name

        mask = np.isnan(data)  # True where missing (NaN)
        self.mask = ~mask  # True where observed (not NaN)

        #df = pd.DataFrame(self.full_file_path_and_name, columns=[f"c{i}" for i in range(self.full_file_path_and_name.shape[1])], index=[f"r{i}" for i in range(self.full_file_path_and_name.shape[0])])
        start_time = pd.Timestamp("2026-01-01 00:00:00")
        #date_index = pd.date_range(start=start_time, periods=data.shape[0], freq='H')
        series_cols = [f"series_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=series_cols)
        #df.insert(0, "date", date_index)  # put 'date' as first column

        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        #df.drop(columns=["date"], inplace=True)
        #df = df.infer_objects(copy=False).interpolate(method="cubic")

        #data_splits = self._get_borders()

        train_data = df #[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df # df[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df # df[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len

        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            c_mask = self.mask[seq_start:seq_end, :].T

            return timeseries, input_mask, c_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1
