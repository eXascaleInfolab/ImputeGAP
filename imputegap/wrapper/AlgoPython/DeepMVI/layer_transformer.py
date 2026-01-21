# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================

from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn import functional as F    
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self,d_query:int,d_key:int,d_value:int, nhead:int, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        # d_query = int(d_query/nhead)
        # d_key = int(d_key/nhead)

        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = d_value
        self.nhead = nhead
        self.scaling = float(d_query) ** -0.5
        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        hidden_dim = 256
        # self.query_linear = torch.nn.Linear(nhead*d_query,nhead*d_query)
        # self.key_linear = torch.nn.Linear(nhead*d_key,nhead*d_key)
        self.query_linear = torch.nn.Sequential(torch.nn.Linear(d_query,hidden_dim),torch.nn.ReLU(),torch.nn.Linear(hidden_dim,nhead*d_query))
        self.key_linear = torch.nn.Sequential(torch.nn.Linear(d_key,hidden_dim),torch.nn.ReLU(),torch.nn.Linear(hidden_dim,nhead*d_key))
        self.value_linear = torch.nn.Sequential(torch.nn.Linear(d_value,hidden_dim),torch.nn.ReLU(),torch.nn.Linear(hidden_dim,nhead*d_value))
        # self.query_linear = torch.nn.Linear(d_query,nhead*d_query)
        # self.key_linear = torch.nn.Linear(d_key,nhead*d_key)
        # self.value_linear = torch.nn.Linear(d_value,nhead*d_value)
        self.out_linear = torch.nn.Linear(nhead*d_value,d_value)
        
    def forward(self,query: Tensor,key: Tensor, value: Tensor,attn_mask: Optional[Tensor] = None) -> Tensor:
        tgt_len, bsz, _ = query.size()
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        query *= self.scaling
        
        query = query.contiguous().view(tgt_len, bsz * self.nhead, self.d_query).transpose(0, 1)
        key = key.contiguous().view(-1, bsz * self.nhead, self.d_key).transpose(0, 1)
        value = value.contiguous().view(-1, bsz * self.nhead, self.d_value).transpose(0, 1)

        attn_output_weights = torch.bmm(query, key.transpose(1, 2))
        if attn_mask is not None:
            attn_mask = torch.repeat_interleave(attn_mask, self.nhead, dim=0)
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        #print (torch.where(torch.isnan(attn_output_weights)))
        #exit()
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, value)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz,self.nhead*self.d_value)
        attn_output = self.out_linear(attn_output)
        return attn_output


class AttentionLayer(torch.nn.Module):
    def __init__(self,d_query,d_key,d_value,nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(AttentionLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_query,d_key,d_value, nhead,dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_value,d_value,d_value, nhead,dropout=dropout)

        self.linear1 = torch.nn.Linear(d_value, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_value)
        self.norm = torch.nn.LayerNorm(d_value)
        self.activation = torch.nn.ReLU()

    def forward(self, query: Tensor, key: Tensor, value: Tensor,src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.self_attn1(query, key, value, src_mask)
        src[torch.isnan(src)] = 0
        src = self.linear2(self.activation(self.linear1(src)))

        # src = self.self_attn2(query, key, src, src_mask)
        # src = self.linear2(self.dropout(self.activation(self.linear1(self.activation(src)))))
        # src[torch.isnan(src)] = 0

        # src = self.self_attn2(src, src, src, src_mask)
        # src = self.linear2(self.dropout(self.activation(self.linear1(self.activation(src)))))
        # src[torch.isnan(src)] = 0

        return src
