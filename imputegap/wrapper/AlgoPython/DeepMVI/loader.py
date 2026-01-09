# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================

import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import _pickle as cPickle
import random
import math,copy
import torch.nn.functional as F
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence

class myDataset(torch.utils.data.Dataset):
    def __init__(self,feats,use_local,time_context=None):
        self.feats = feats.astype(float)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.size = int(self.feats.shape[0]*self.feats.shape[1])
        self.use_local = use_local
        
    def __getitem__(self,index):
        time_ = (index%self.feats.shape[0])
        tsNumber = int(index/self.feats.shape[0])
        lower_limit = min(time_,self.time_context)
        series = self.feats[time_-lower_limit:time_+self.time_context]
        residuals = np.nan_to_num(copy.deepcopy(self.feats[time_-lower_limit:time_+self.time_context,:]))
        residuals[:,tsNumber] = 0
        time_vector = torch.arange(series.shape[0])+(time_-lower_limit)
            
        series = series[:,tsNumber]
        series = copy.deepcopy(series)
        mask = np.ones(series.shape)
        mask[np.isnan(series)] = 0
        
        series = np.nan_to_num(series)

        context = [tsNumber]
        sz = mask.shape[0]
        attn_mask = torch.ones(450,sz)
        if (not self.use_local):
            attn_mask[:,mask==0] = 0
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
        return torch.FloatTensor(series),torch.BoolTensor(mask>0),context,torch.FloatTensor(residuals),0,time_vector,attn_mask.transpose(0,1)
        
    def __len__(self):
        return self.size


class myValDataset(torch.utils.data.Dataset):
    def __init__(self,feats,examples_file,test,use_local,time_context=None):
        self.feats = feats.astype(float)
        self.examples_file = examples_file.astype(int)
        self.time_context = time_context
        self.test = test
        self.use_local = use_local
        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]

        lower_limit = min(time_,self.time_context)
        out_series = self.feats[time_-lower_limit:time_+self.time_context+this_example[2]]
        residuals = np.nan_to_num(copy.deepcopy(self.feats[time_-lower_limit:time_+self.time_context+this_example[2],:]))
        residuals[:,this_example[1]] = 0
        start_time = time_-lower_limit

        time_ = lower_limit
        time_vector = torch.arange(out_series.shape[0])+start_time

        out_series = out_series[:,this_example[1]]
        
        upper_limit = this_example[2]
        
        mask = np.zeros(out_series.shape)
        mask[time_:time_+upper_limit] = 1
        if (not self.test):
            mask[np.isnan(out_series)] = 0
        out_series = np.nan_to_num(out_series)

        context = [this_example[1]]
        
        sz = mask.shape[0]
        attn_mask = torch.ones(450,sz)
        if (not self.use_local):
            attn_mask[:,mask==1] = 0
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
        
        return torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,torch.FloatTensor(residuals),start_time,time_vector,attn_mask.transpose(0,1)
        
    def __len__(self):
        return self.examples_file.shape[0]
        

def my_collate(batch):
    (series,mask,index,residuals,start_time,time_vector,attn_mask) = zip(*batch)
    return pad_sequence(series,batch_first=True),pad_sequence(mask,batch_first=True),pad_sequence(residuals,batch_first=True),\
        [torch.LongTensor(list(index)),torch.LongTensor(list(start_time)),pad_sequence(time_vector,batch_first=True),pad_sequence(attn_mask,batch_first=True,padding_value=float('-inf')).transpose(1,2)]

