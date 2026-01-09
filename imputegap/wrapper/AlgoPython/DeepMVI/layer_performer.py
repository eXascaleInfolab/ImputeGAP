# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================


import copy
from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn import functional as F    
from utils import *
from functools import *

# S,N,E
# N,H,S,E
# SNE -> NSE -> NSE*H -> NSHE -> NHSE
# value is NHSE -> NSHE -> NSE-> SNE
# return sould be SNE
class FastCausalMultiheadAttention(torch.nn.Module):
    def __init__(self,d_query:int,d_key:int,d_value:int, nhead:int, backward=False):
        super(FastCausalMultiheadAttention, self).__init__()
        self.head_dim = d_value
        self.nhead = nhead
        self.scaling = float(d_query) ** -0.5
        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.query_linear = torch.nn.Linear(d_query,nhead*d_query)
        self.key_linear = torch.nn.Linear(d_key,nhead*d_key)
        self.value_linear = torch.nn.Linear(d_value,nhead*d_value)
        self.out_linear = torch.nn.Linear(nhead*d_value,d_value)

        nb_features = int(d_query * math.log(d_query))
        self.projection_matrix = gaussian_orthogonal_random_matrix(nb_rows = nb_features, nb_columns = d_query, scaling = 0)
        self.kernel_fn = torch.nn.ReLU()
        self.causal_linear_fn = causal_linear_attention
        self.mask_size = 10
        self.backward = backward
        
    def forward(self,query: Tensor,key: Tensor, value: Tensor,attn_mask: Optional[Tensor] = None) -> Tensor:
        tgt_len, bsz, _ = query.size()
        query = self.query_linear(query.transpose(0,1)).view(bsz,tgt_len,self.nhead,self.d_query).transpose(1,2)
        key = self.key_linear(key.transpose(0,1)).view(bsz,tgt_len,self.nhead,self.d_key).transpose(1,2)
        value = self.value_linear(value.transpose(0,1)).view(bsz,tgt_len,self.nhead,self.d_value).transpose(1,2)
        
        query *= self.scaling
        
        device = query.device
        
        query = softmax_kernel(query, is_query = True, projection_matrix = self.projection_matrix, device = device)
        key = softmax_kernel(key, is_query = False, projection_matrix = self.projection_matrix, device = device)

        if (self.training):
            mask_len = torch.randint(low=1,high=self.mask_size+1,size=(1,))[0]
        else :
            mask_len = 1
        dummy = torch.zeros((bsz,self.nhead,mask_len,self.d_value)).to(query.device)
        if (not self.backward):
            temp = self.causal_linear_fn(query[:,:,mask_len:,:], key[:,:,:-mask_len,:], value[:,:,:-mask_len,:])
            temp = torch.cat([dummy,temp],dim=2)
        else :
            temp = self.causal_linear_fn(query[:,:,:-mask_len,:], key[:,:,mask_len:,:], value[:,:,mask_len:,:])
            temp = torch.cat([temp,dummy],dim=2)
        # if (self.training):
        #     mask_len = torch.randint(low=1,high=self.mask_size+1,size=(1,))[0]
        #     dummy = torch.zeros((bsz,self.nhead,mask_len,self.d_value)).to(query.device)
        #     if (not self.backward):
        #         temp = self.causal_linear_fn(query[:,:,mask_len:,:], key[:,:,:-mask_len,:], value[:,:,:-mask_len,:])
        #         temp = torch.cat([dummy,temp],dim=2)
        #     else :
        #         temp = self.causal_linear_fn(query[:,:,:-mask_len,:], key[:,:,mask_len:,:], value[:,:,mask_len:,:])
        #         temp = torch.cat([temp,dummy],dim=2)
        # else :
        #     temp = self.causal_linear_fn(query, key, value)
            
        value = self.out_linear(temp.transpose(1,2).contiguous().view(bsz,tgt_len,-1)).transpose(0,1)

        return value



class PerformerEncoderLayer(torch.nn.Module):
    def __init__(self,d_query,d_key,d_value,nhead, dim_feedforward=2048, dropout=0.1, backward=False,activation="relu"):
        super(PerformerEncoderLayer, self).__init__()
        self.backward = backward
        self.self_attn1 = FastCausalMultiheadAttention(d_query,d_key,d_value, nhead,backward=self.backward)
        # self.self_attn2 = FastCausalMultiheadAttention(d_query,d_key,d_value, nhead,backward=self.backward)
        self.self_attn2 = FastCausalMultiheadAttention(d_value,d_value,d_value, nhead,backward=self.backward)
        self.linear1 = torch.nn.Linear(d_value, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_value)
        self.norm = torch.nn.LayerNorm(d_value)
        self.activation = torch.nn.ReLU()
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor,src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.self_attn1(query, key, value, src_mask)
        src = self.linear2(self.dropout(self.activation(self.linear1(self.activation(src)))))
        src[torch.isnan(src)] = 0

        # src = self.self_attn2(query, key, src, src_mask)
        # src = self.linear2(self.dropout(self.activation(self.linear1(self.activation(src)))))
        # src[torch.isnan(src)] = 0

        src = self.self_attn2(src, src, src, src_mask)
        src = self.linear2(self.dropout(self.activation(self.linear1(self.activation(src)))))
        src[torch.isnan(src)] = 0

        return src

# def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
#     b, h, *_ = data.shape

#     data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

#     ratio = (projection_matrix.shape[0] ** -0.5)

#     projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
#     projection = projection.type_as(data)

#     data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

#     diag_data = data ** 2
#     diag_data = torch.sum(diag_data, dim=-1)
#     diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
#     diag_data = diag_data.unsqueeze(dim=-1)

#     if is_query:
#         data_dash = ratio * (
#             torch.exp(data_dash - diag_data -
#                     torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
#     else:
#         data_dash = ratio * (
#             torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

#     return data_dash.type_as(data)



# # non-causal linear attention
# def linear_attention(q, k, v):
#     k_cumsum = k.sum(dim = -2)
#     D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
#     context = torch.einsum('...nd,...ne->...de', k, v)
#     out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
#     return out

# # efficient causal linear attention, created by EPFL
# # TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back

# # def causal_linear_attention(q, k, v,backward=False,mask=None):
# #     k_cumsum = k.cumsum(dim=-2)
# #     context = torch.einsum('...nd,...ne->...nde', k, v)
# #     context = context.cumsum(dim=-3)
# #     if (backward):
# #         k_cumsum = torch.flip(k_cumsum,dims = (-2,))
# #         context = torch.flip(context,dims = (-3,))
        
# #     context = context[:,:,mask,:,:]
# #     k_cumsum  = k_cumsum[:,:,mask,:]
# #     D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
# #     out = torch.einsum('...nde,...nd,...n->...ne', context, q, D_inv)
# #     return out

# def causal_linear_attention(q, k, v,backward=False,mask=None):
#     from fast_transformers.causal_product import CausalDotProduct
#     cuda_context = null_context

#     causal_dot_product_fn = CausalDotProduct.apply

#     k_cumsum = k.cumsum(dim=-2)
#     D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

#     with cuda_context():
#         out = causal_dot_product_fn(q, k, v)

#     out = torch.einsum('...nd,...n->...nd', out, D_inv)
#     return out


# def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
#     unstructured_block = torch.randn((cols, cols), device = device)
#     q, r = torch.qr(unstructured_block.cpu(), some = True)
#     q, r = map(lambda t: t.to(device), (q, r))

#     # proposed by @Parskatt
#     # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
#     if qr_uniform_q:
#         d = torch.diag(r, 0)
#         q *= d.sign()
#     return q.t()


# def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
#     nb_full_blocks = int(nb_rows / nb_columns)

#     block_list = []

#     for _ in range(nb_full_blocks):
#         q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
#         block_list.append(q)

#     remaining_rows = nb_rows - nb_full_blocks * nb_columns
#     if remaining_rows > 0:
#         q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
#         block_list.append(q[:remaining_rows])

#     final_matrix = torch.cat(block_list)

#     if scaling == 0:
#         multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
#     elif scaling == 1:
#         multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
#     else:
#         raise ValueError(f'Invalid scaling {scaling}')

#     return torch.diag(multiplier) @ final_matrix


# # if __name__ == "__main__":
# #     exit()


    
