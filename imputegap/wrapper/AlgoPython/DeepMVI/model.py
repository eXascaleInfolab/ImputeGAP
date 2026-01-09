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
device = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

    
class AttentionModule(nn.Module):
    def __init__(self, nhead=2, nhid=64, dquery=32,dkey=32,dvalue=32,dropout=0.0,mask_len=10):
        super(AttentionModule, self).__init__()
        from imputegap.wrapper.AlgoPython.DeepMVI.layer_transformer import AttentionLayer
        self.pos_encoder = PositionalEncoding(dquery, dropout)
        self.attention = AttentionLayer(dquery,dquery,dvalue,nhead, nhid, dropout)
        self.mask_len = mask_len
        
    def forward(self, value,query,mask_attn_in,test):
        value = value.transpose(1,2).transpose(0,1)
        query_pos = self.pos_encoder(query.transpose(1,2)).transpose(0,1)
        mask_attn = self._generate_square_subsequent_mask(value.shape[0],test).unsqueeze(0).repeat(mask_attn_in.shape[0],1,1)
        mask_attn += mask_attn_in
        output = self.attention(query_pos,query_pos,value,mask_attn.to(value.device))
        return output.transpose(0,1)

    def _generate_square_subsequent_mask(self,sz:int,test):
        mask_len = self.mask_len
        mask = torch.ones(sz,sz)
        if (test):
            return mask
        if (mask_len > 1):
            start_points = (torch.arange(sz)-torch.randint(low=0,high=mask_len-1,size=(sz,))).long()
        else :
            start_points = torch.arange(sz).long()
        for i in range(sz):
            mask[i,start_points[i]:start_points[i]+mask_len] = 0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
class OurModel(nn.Module):
    def __init__(self,sizes,kernel_size=10,block_size=10,time_len=1000,nhead=2,use_embed=True,use_context=True,use_local=True):
        super(OurModel, self).__init__()
        hidden_dim = 512
        #nkernel=int(32*np.log10(kernel_size))
        #nkernel=int(3.2*kernel_size)
        nkernel=64
        embedding_size=10
        
        self.kernel_size = kernel_size
        self.k = 10
        self.tau = 1
        self.use_embed = use_embed
        self.use_context = use_context
        self.use_local = use_local
        self.block_size = block_size
        
        if (self.use_context):
            self.atten_qdim = 32
            self.mapping_query = nn.Linear(2*nkernel,self.atten_qdim)
        else :
            self.atten_qdim = 32
            
        self.vdim = nkernel
        kernel_output_dim = 3*len(sizes)
        mask_len = min(int (block_size/kernel_size),1)
        
        if (use_embed):
            self.embeddings = []
            for x in sizes:
                self.embeddings.append(nn.Embedding(x, embedding_size))
            self.transformer_embeddings = nn.ModuleList(self.embeddings)

        self.conv = nn.Conv1d(1,nkernel,kernel_size = self.kernel_size,stride=self.kernel_size)
        self.deconv = nn.ConvTranspose1d(self.vdim,self.vdim,kernel_size = self.kernel_size,stride=self.kernel_size)
        self.pool = nn.AvgPool1d(kernel_size = self.kernel_size,stride = self.kernel_size)
        self.attention = AttentionModule(dquery=self.atten_qdim,dkey=self.atten_qdim,dvalue=self.vdim,nhid=hidden_dim,nhead=nhead,dropout=0.0,mask_len=mask_len)
        # self.mapping_value = nn.Linear(nkernel,self.vdim)
        
        if (use_local):
            assert self.kernel_size % self.block_size == 0
            self.pool2 = nn.AvgPool1d(kernel_size = self.block_size,stride = self.block_size)
            
        if (use_embed):
            final_input_dim = self.vdim+kernel_output_dim
        elif (use_embed):
            final_input_dim = kernel_output_dim
        else :
            final_input_dim = self.vdim
        if (use_local):
            final_input_dim += 1
            self.mean_outlier_layer = nn.Linear(final_input_dim,1)
        else : 
            self.mean_outlier_layer = nn.Linear(final_input_dim,1)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.std = None
        self.mlp = nn.Sequential(nn.Linear(self.vdim+embedding_size,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,self.vdim))

    def sibling_feats(self,y_context : List[torch.Tensor],indices):
        final1,final2,final3 = [],[],[]
        for i,embed in enumerate(self.transformer_embeddings):
            temp = torch.cdist(embed.weight[indices[i]].unsqueeze(0),embed.weight.unsqueeze(0),p=2.0)[0]+1e-3
            temp[torch.arange(temp.shape[0]),indices[i]] = 0
            temp_w = torch.exp(-temp/self.tau)
            temp_i = torch.argsort(temp_w)[:,-(self.k+1):-1]
            temp_s = y_context[i][torch.arange(y_context[i].shape[0])[:,None],:,temp_i]
            temp_w = temp_w[torch.arange(temp_w.shape[0])[:,None],None,temp_i].repeat(1,1,y_context[0].shape[1])
            temp_w[temp_s==0] = temp_w[temp_s==0]/1e9
            final1.append(temp_w.sum(dim=1,keepdim=True))
            final2.append((temp_w*temp_s).sum(dim=1,keepdim=True)/temp_w.sum(dim=1,keepdim=True))
            final3.append(torch.std(temp_s,dim=1,keepdim=True))
        return torch.cat(final1+final2+final3,dim=1).transpose(1,2)

    def context_feats (self,in_series,test):
        org_shape = in_series.shape[1]
        if (in_series.shape[1]%self.kernel_size != 0):
            in_series = torch.cat([in_series,torch.zeros(in_series.shape[0],self.kernel_size-in_series.shape[1]%self.kernel_size).to(in_series.device)],dim=1)
        feat1 = self.pool(in_series.unsqueeze(1))[:,0,:]
        feat1 = torch.repeat_interleave(feat1,self.kernel_size,dim=1)*self.kernel_size
        if (test):
            mask = (in_series != 0).int().float()
            den = self.pool(mask.unsqueeze(1))[:,0,:]
            den = (torch.repeat_interleave(den,self.kernel_size,dim=1)*self.kernel_size).clamp(min=1)
            out_feats = feat1/den
        else : 
            feat2 = self.pool2(in_series.unsqueeze(1))[:,0,:]
            feat2 = torch.repeat_interleave(feat2,self.block_size,dim=1)*self.block_size
            out_feats = (feat1 - feat2)/(self.kernel_size-self.block_size)
        return out_feats.unsqueeze(1)[:,:,:org_shape]                                                                                                                                                                         

    def core(self,in_series,residuals,context_info : List[torch.Tensor],test=False):
        value = self.conv(in_series.unsqueeze(1))
        attn_mask = self.pool(context_info[3])

        if (self.use_embed):
            # bs = residuals.shape[0]
            # ns = residuals.shape[1]
            # residuals = residuals.reshape(bs*ns,1,residuals.shape[2])
            # residuals_conv = self.conv(self.dropout(residuals)).clamp(min=0)
            # residuals_conv = residuals_conv.reshape(bs,ns,residuals_conv.shape[1],residuals_conv.shape[2])
            # siblings = self.sibling_feats(residuals_conv,context_info[0].transpose(0,1))
            siblings = self.sibling_feats([residuals],context_info[0].transpose(0,1))

        if (self.use_context):
            value_left = torch.zeros(value.shape).to(value.device)
            value_right = torch.zeros(value.shape).to(value.device)
            value_left[:,:,1:] = value[:,:,:-1]
            value_right[:,:,:-1] = value[:,:,1:]
            attn_query = self.mapping_query(torch.cat([value_left,value_right],dim=1).transpose(1,2)).transpose(1,2)
        else :
            attn_query = torch.zeros((value.shape[0],self.atten_qdim,value.shape[2])).to(value.device)

        # value = self.mapping_value(value.transpose(1,2)).transpose(1,2)
        hidden_state = self.attention(value.clamp(min=0),attn_query.clamp(min=0),attn_mask[:,:attn_mask.shape[2],:],test).clamp(min=0)
        # hidden_state = self.mlp(torch.cat([hidden_state,self.transformer_embeddings[0].weight[context_info[0][:,0]][:,None,:].expand(-1,hidden_state.shape[1],-1)],dim=-1))
        hidden_state = self.deconv(hidden_state.transpose(1,2)).transpose(1,2)
        
        if (in_series.shape[1]%self.kernel_size != 0):
            hidden_state = torch.cat([hidden_state,torch.zeros([hidden_state.shape[0],in_series.shape[1]-hidden_state.shape[1],hidden_state.shape[2]]).to(hidden_state.device)],dim=1)
                
        if (self.use_embed):
            feats = torch.cat([hidden_state,siblings],dim=2)
        else:
            feats = hidden_state
        if (self.use_local):
            local_feats = self.context_feats(in_series,test).transpose(1,2)
            feats = torch.cat([feats,local_feats],dim=2)
            
        mean = self.mean_outlier_layer(feats)[:,:,0]
        return mean
    

    def forward (self,in_series,mask,residuals,context_info : List[torch.Tensor]):
        mean = self.core(in_series,residuals,context_info)
        return {'mae':self.mae_loss(mean,in_series,mask,context_info).mean()}
    
    @torch.jit.export
    def validate(self,in_series,mask,residuals, context_info  : List[torch.Tensor],test=False):
        mean = self.core(in_series,residuals,context_info,test)
        loss = self.mae_loss(mean,in_series,mask,context_info)
        return {'loss_values':loss,'values':mean}

    def mae_loss(self,y,y_pred,mask,context_info):
        temp = torch.abs((y_pred - y).to(device) * self.std[context_info[0]].to(device))
        loss = temp[mask>0]
        #print (loss.mean())
        return loss

