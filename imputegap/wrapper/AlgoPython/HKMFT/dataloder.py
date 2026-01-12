# ===============================================================================================================
# SOURCE: https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/8979178
# ===============================================================================================================


# Copyright (c) [2021] [wlicsnju]
# [HKMF-T] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
# See the Mulan PSL v2 for more details.  
import logging

import numpy as np
import pandas as pd

from datetime import datetime
from enum import Enum
from typing import Tuple, Union


class Dataset(Enum):
    BSD = ('dataset/BSD.csv', (datetime(2011, 1, 1), datetime(2012, 12, 31)))
    MVCD = ('dataset/MVCD.csv', (datetime(2014, 1, 1), datetime(2016, 12, 31)))
    EPCD = ('dataset/EPCD.csv', (datetime(2007, 1, 1), datetime(2009, 12, 31)))


class DataLoader(object):
    def __init__(self, dataset: Union[Dataset, str], time_window: Tuple[datetime, datetime] = None):
        """
        :param dataset: Dataset
        :param time_window:
        """
        self._mask = None
        if isinstance(dataset, Dataset):
            self._filename = dataset.value[0]
            self._time_window = time_window or dataset.value[1]
        elif isinstance(dataset, str):
            self._filename = dataset
            self._time_window = None
        self._index, self._data, self._tag, self.dim_name, self.tag_name = DataLoader._load(self._filename,
                                                                                            self._time_window)

    def norm(self, min_v: float = 0., max_v: float = 10.):
        """
        归一化, 目前为 min-max 归一化, 将会改成 max 归一化.
        :param min_v:
        :param max_v:
        :return:
        """
        # min-max 归一化
        # d_min = self._data.min()
        # d_max = self._data.max()
        # if d_max - d_min <= 1e-5 or max_v - min_v <= 1e-5:
        #     logging.error('params error, data error.')
        # self._data = ((self._data - d_min) / (d_max - d_min) * (max_v - min_v)) + min_v
        # self._mask = None
        # max 归一化.
        d_max = self._data.max()
        self._data = (self._data / d_max) * max_v
        self._mask = None

    def __len__(self):
        return len(self._index)

    def generate_mask(self, begin_idx: int, end_idx: int):
        """
        生成左闭右开 mask 形如：
        [1, 1, 1,  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        [start_idx ↑,  end_idx ↑)
        """
        if len(self) > begin_idx >= 0 and len(self) > end_idx > 0 and begin_idx < end_idx:
            self._mask = np.ones((self._data.shape[-1],), dtype=np.int8)
            self._mask[begin_idx: end_idx] = 0
        else:
            logging.error('error input!', ValueError)
            self._mask = None

    def get_data(self):
        """
        返回的数据中 0.0 表示缺失的数据。
        注意：data 和 mask 总是返回拷贝而不是引用，tag 总是返回引用，应防止修改 tag。
        :return: data, mask, tag, ground_truth
        """
        d = np.array(self._data)
        gt = d[:, self._mask == 0]
        d[:, self._mask == 0] = 0.0
        return d, np.array(self._mask), self._tag, gt

    @staticmethod
    def _load(filename: str, time_window: Tuple[datetime, datetime]) -> Tuple[list, np.ndarray, np.ndarray, list, list]:
        """
        此函数会将输入的 csv 文件转换成便于处理的 numpy 矩阵。
        注意：任何类型的 tag 都会重新 hash 成 0 开始的整数。
        :param filename: str
        :param time_window: [start, end]
        :return: index, data, tag, dim_name, tag_name
        """
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index)
        if time_window is not None:
            df = df[(df.index >= time_window[0]) & (df.index <= time_window[1])]
        cols = df.columns.to_list()
        dim_name = []
        for n in cols:
            if str(n).startswith('data_'):
                dim_name.append(str(n)[5:])
        if len(dim_name) <= 0:
            logging.error(f'Empty data columns in file {filename}, pls check csv head.')
        tag_name2id = {}
        tag_name_list = []
        for i, n in enumerate(dim_name):
            for t in df[f'tag_{n}'].unique():
                if t not in tag_name2id:
                    tag_name2id[t] = len(tag_name2id)
                    tag_name_list.append(t)
        data = np.zeros((len(dim_name), df.index.size))
        tag = np.zeros((len(dim_name), df.index.size), dtype=int)
        for i, n in enumerate(dim_name):
            data[i] = df[f'data_{n}']
            tmp = df[f'tag_{n}']
            tag[i] = [tag_name2id[t] for t in tmp]
        return df.index.to_list(), data, tag, dim_name, tag_name_list


if __name__ == '__main__':
    b = DataLoader(Dataset.BSD)
    b.generate_mask(4, 6)
    data, mask, tag, gt = b.get_data()
    e = DataLoader(Dataset.EPCD)
    m = DataLoader(Dataset.MVCD)
