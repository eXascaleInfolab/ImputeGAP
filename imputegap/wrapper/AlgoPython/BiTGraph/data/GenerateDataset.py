import os
import random
from typing import Optional, Sequence, List
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import torch

np.set_printoptions(threshold=np.inf)

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
class TSDataset(Dataset):

    def __init__(self, Data, Label,mask,masks_target):
        self.Data = Data
        self.Label = Label
        self.mask = mask
        self.masks_target = masks_target


    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        mask = torch.Tensor(self.mask[index])
        masks_target = torch.Tensor(self.masks_target[index])

        return data,label,mask,masks_target


def get_0_1_array(array,rate=0.2):
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array

def synthetic_data(mask_ratio,dataset):

    if(dataset=='Metr'):
        path = os.path.join('./data/metr_la/', 'metr_la.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        data = data[:, :, None]
        mask=get_0_1_array(data,mask_ratio)


    elif(dataset=='PEMS'):
        path = os.path.join('./data/pems_bay/', 'pems_bay.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        data = data[:, :, None]
        mask = get_0_1_array(data, mask_ratio)


    elif(dataset=='ETTh1'):
        df_raw = pd.read_csv('./data/ETT/ETTh1.csv')
        data=np.array(df_raw)
        data=data[::,1:]
        mask = get_0_1_array(data, mask_ratio)
        data = data[:, :, None].astype('float32')
        mask = mask[:, :, None].astype('int32')

    elif (dataset == 'Elec'):
        data_list = []
        with open('./data/Electricity/electricity.txt', 'r') as f:
            reader = f.readlines()
            for row in reader:
                data_list.append(row.split(','))

        data = np.array(data_list).astype('float')
        mask = get_0_1_array(data, mask_ratio)
        data = data[:, :, None].astype('float32')
        mask = mask[:, :, None].astype('int32')

    elif(dataset=='BeijingAir'):

        data = pd.DataFrame(pd.read_hdf('./data/air_quality/small36.h5', 'pm25'))
        data=np.array(data)
        eval_mask=~np.isnan(data)
        mask= get_0_1_array(data, mask_ratio)  #   ~np.isnan(data)
        data[np.isnan(data)]=0.0
        data = data[:, :, None].astype('float32')
        mask = mask[:, :, None].astype('int32')


    return data,mask


def split_data_by_ratio(x,y, mask,mask_target,val_ratio, test_ratio):
    idx = np.arange(x.shape[0])
    # print('idx shape:',idx.shape)
    idx_shuffle = idx.copy()
    # np.random.shuffle(idx_shuffle)
    data_len = x.shape[0]
    test_x = x[idx_shuffle[-int(data_len * test_ratio):]]
    test_y = y[idx_shuffle[-int(data_len * test_ratio):]]
    test_x_mask = mask[idx_shuffle[-int(data_len * test_ratio):]]
    test_y_mask = mask_target[idx_shuffle[-int(data_len * test_ratio):]]



    val_x = x[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]]
    val_y = y[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]]
    val_x_mask = mask[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]]
    val_y_mask = mask_target[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]]



    train_x = x[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]]
    train_y = y[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]]
    train_x_mask = mask[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]]
    train_y_mask = mask_target[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]]



    return train_x,train_y,train_x_mask,train_y_mask,val_x,val_y,val_x_mask,val_y_mask,test_x,test_y,test_x_mask,test_y_mask


def Add_Window_Horizon(data, mask, window=1, horizon=1):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []  #horizon
    masks=[]
    masks_target=[]
    index = 0

    while index < end_index:
        X.append(data[index:index+window])
        masks.append(mask[index:index+window])
        Y.append(data[index+window:index+window+horizon])
        masks_target.append(mask[index+window:index+window+horizon])
        index = index + 1
    X = np.array(X)  #backcast B,W,N,D
    Y = np.array(Y)  #forecast B,H,N,D
    masks = np.array(masks)
    masks_target=np.array(masks_target)

    return X, Y,masks,masks_target


def loaddataset(data_matrix, history_len=24, pred_len=16):
    mask = ~np.isnan(data_matrix)  # Mask where values are observed
    data_matrix[np.isnan(data_matrix)] = 0  # Replace NaNs with zeros for processing

    x, y, mask, mask_target = Add_Window_Horizon(data_matrix, mask, history_len, pred_len)

    print(f"Final Shape of x before reshaping: {x.shape}")  # Debugging
    print(f"Final Shape of y before reshaping: {y.shape}")  # Debugging

    # 🔹 Fix: Expand dimensions if necessary (batch, seq_len, num_nodes) → (batch, 1, seq_len, num_nodes)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=1)  # (batch, 1, seq_len, num_nodes)
        y = np.expand_dims(y, axis=1)  # (batch, 1, horizon, num_nodes)
        mask = np.expand_dims(mask, axis=1)
        mask_target = np.expand_dims(mask_target, axis=1)

    print(f"Fixed Shape of x: {x.shape}")  # Debugging
    print(f"Fixed Shape of y: {y.shape}")  # Debugging

    train_x, train_y, masks_tra, masks_target_tra, \
        val_x, val_y, masks_val, masks_target_val, \
        test_x, test_y, masks_test, masks_target_test = split_data_by_ratio(
        x, y, mask, mask_target, 0.2, 0.2)

    print(f"Train set size: {train_x.shape}, Validation set size: {val_x.shape}, Test set size: {test_x.shape}")

    scaler = StandardScaler(mean=train_x.mean(), std=train_x.std())
    x_tra, y_tra = scaler.transform(train_x), scaler.transform(train_y)
    x_val, y_val = scaler.transform(val_x), scaler.transform(val_y)
    x_test, y_test = scaler.transform(test_x), scaler.transform(test_y)

    train_dataset = TSDataset(x_tra, y_tra, masks_tra, masks_target_tra)
    val_dataset = TSDataset(x_val, y_val, masks_val, masks_target_val)
    test_dataset = TSDataset(x_test, y_test, masks_test, masks_target_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler



if __name__ == '__main__':
    print('')