import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import imputegap.tools.utils as utils_imp
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

    # also mark existing NaNs as 0 in the mask
    re_array[np.isnan(array)] = 0

    return re_array

def synthetic_data(ts_m, mask_ratio,dataset):

    if ts_m is None:
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

    else:
        data = ts_m.copy()

        if dataset == "reco":
            mask_ori = ~np.isnan(data)
            mask_tar = None
            data[~mask_ori] = 0.0
        else:
            mask_tar = ~np.isnan(data)
            mask_ori = get_0_1_array(data, mask_ratio).astype(bool)  # ensure bool  # ~np.isnan(data)
            data[~mask_tar] = 0.0

        data = data[:, :].astype('float32')
        mask_ori = mask_ori[:, :].astype('int32')
        if mask_tar is not None:
            mask_tar = mask_tar[:, :].astype('int32')

        return data, mask_ori, mask_tar

    return data,mask


def split_data_by_ratio(x, y, mask, mask_target, val_ratio, test_ratio):
    idx = np.arange(x.shape[0])
    # print('idx shape:',idx.shape)
    idx_shuffle = idx.copy()
    #np.random.shuffle(idx_shuffle)
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



def Add_Window_Horizon_Imputegap(data, mask_ori, mask_tar, seq_len=3, horizon=0, multivariate=False, sliding_windows=1, verbose=True):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''

    if multivariate:
        data = utils_imp.dataset_add_dimensionality(data, seq_length=seq_len, verbose=False)
        mask_ori = utils_imp.dataset_add_dimensionality(mask_ori, seq_length=seq_len, verbose=False)

        if mask_tar is not None:
            mask_tar = utils_imp.dataset_add_dimensionality(mask_tar, seq_length=seq_len, verbose=False)
    else:
        data = utils_imp.window_truncation(data, seq_len=seq_len, stride=sliding_windows, info="bitgraph - data | ", verbose=False)
        mask_ori = utils_imp.window_truncation(mask_ori, seq_len=seq_len, stride=sliding_windows, info="bitgraph - mask | ", verbose=False)
        if mask_tar is not None:
            mask_tar = utils_imp.window_truncation(mask_tar, seq_len=seq_len, stride=sliding_windows, info="bitgraph - mask | ", verbose=False)
        #length = len(data)
        #end_index = length - horizon - seq_len + 1

    X = data[:, :, :, None].astype('float32')
    masks = mask_ori[:, :, :, None].astype('int32')

    if mask_tar is not None:
        masks_target = mask_tar[:, :, :, None].astype('int32')
    else:
        masks_target = masks

    Y = X

    if verbose:
        print(f"\nadding window/sample and horizon:\n{X.shape = }")
        print(f"{Y.shape = }")
        print(f"{masks.shape = }")
        print(f"{masks_target.shape = }\n")

    return X, Y, masks, masks_target

def Add_Window_Horizon(data,mask, window=3, horizon=1):
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


def loaddataset(ts_m, history_len, pred_len, mask_ratio, args, sliding_windows=1, multivariate=False, verbose=True):
    data, mask_ori, mask_tar = synthetic_data(ts_m, args.mask_ratio, "train")
    data_imp, mask_ori_imp, mask_tar_imp = synthetic_data(ts_m, args.mask_ratio, "reco")

    if args.deep_verbose:
        print(f"\n{args=}\n")

    x, y, mask, mask_target = Add_Window_Horizon_Imputegap(data, mask_ori, mask_tar, history_len, pred_len, sliding_windows=sliding_windows, multivariate=multivariate, verbose=verbose)
    x_reco, y_reco, mask_reco, mask_target_reco = Add_Window_Horizon_Imputegap(data_imp, mask_ori_imp, mask_tar_imp, history_len, pred_len, sliding_windows=sliding_windows, multivariate=multivariate, verbose=False)

    train_x,train_y,masks_tra,masks_target_tra, val_x,val_y,masks_val,masks_target_val, test_x,test_y,masks_test,masks_target_test = split_data_by_ratio(x, y, mask, mask_target, args.val_r, args.ts_r)

    if verbose:
        print(f"\nsplitting:\n(TRAIN): {train_x.shape = }")
        print(f"(VAL): {val_x.shape = }")
        print(f"(TEST): {test_x.shape = }")
        print(f"(RECONSTRUCTION): {x_reco.shape = }\n")

    if args.batch_size > val_x.shape[0]:
        args.batch_size = val_x.shape[0]
        if verbose:
            print(f"\n(INFO) batch size adapted to the current val length: {args.batch_size = }\n")

    if args.norma:
        scaler = StandardScaler(mean=train_x.mean(), std=train_x.std())
        scaler_reco = StandardScaler(mean=x_reco.mean(), std=x_reco.std())
        train_x = scaler.transform(train_x)
        train_y = scaler.transform(train_y)
        val_x = scaler.transform(val_x)
        val_y = scaler.transform(val_y)
        test_x = scaler.transform(test_x)
        test_y = scaler.transform(test_y)

        x_reco = scaler_reco.transform(x_reco)
        y_reco = scaler_reco.transform(y_reco)
    else:
        scaler = None
        scaler_reco = None

    train_dataset = TSDataset(train_x, train_y,masks_tra,masks_target_tra)
    val_dataset = TSDataset(val_x, val_y,masks_val,masks_target_val)
    test_dataset = TSDataset(test_x, test_y,masks_test,masks_target_test)
    imp_dataset = TSDataset(x_reco, y_reco, mask_reco, mask_target_reco)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    imp_dataloader = DataLoader(imp_dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, imp_dataloader, scaler, scaler_reco


