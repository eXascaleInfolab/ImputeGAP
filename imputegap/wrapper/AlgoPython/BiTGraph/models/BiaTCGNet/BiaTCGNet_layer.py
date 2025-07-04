# ===============================================================================================================
# SOURCE: https://github.com/chenxiaodanhit/BiTGraph
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=O9nZCwdGcG
# ===============================================================================================================

import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm


class getweight(nn.Module):
    def __init__(self,inputsize):
        super(getweight, self).__init__()
        self.mlp=nn.Linear(inputsize,10)
        self.mlp_input=nn.Linear(24,10)
        self.mlp_theta=nn.Linear(24,10)
        self.dimendefrom=nn.Conv2d(10,1,(1,1))
        self.embedding_1=nn.Linear(10,10)
        self.embedding_2=nn.Linear(10,10)
    def forward(self,mask,x,theta):
        mask_projection=self.mlp(mask)
        input_projection=self.mlp_input(x)
        theta_projection=self.mlp_theta(theta)
        mask_input=torch.cat([mask_projection,input_projection,theta_projection],dim=1) #,theta_projection
        mask_input=self.dimendefrom(mask_input)
        mask_q=self.embedding_1(mask_input)
        mask_k=self.embedding_2(mask_input)
        mask_weight=torch.matmul(mask_q,mask_k.transpose(2,3))
        mask_weight=torch.sigmoid(mask_weight/(torch.sqrt(torch.tensor(10.0))))
        return mask_weight

class nconv(nn.Module):
    def __init__(self,inputsize,num_nodes,input_len,pred_len):
        super(nconv,self).__init__()
        self.pred_len = pred_len
        self.input_len = input_len
        self.num_nodes = num_nodes
        self.inputsize = inputsize
        self.mlp = nn.Linear(inputsize, 10)
        self.getweight = getweight(inputsize)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha1 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=self.device))
        self.alpha2 = nn.Parameter(torch.zeros(num_nodes, num_nodes, device=self.device))

        self.getweight = getweight(self.input_len)
        #self.triu_matrix = torch.from_numpy(np.triu(np.ones((self.num_nodes, self.num_nodes)), 1)).cuda().float()
        #self.tril_matrix = torch.from_numpy(np.tril(np.ones((self.num_nodes, self.num_nodes)), 1)).cuda().float()

        self.triu_matrix = torch.from_numpy(np.triu(np.ones((self.num_nodes, self.num_nodes)), 1).astype(np.float32)).to(self.device)
        self.tril_matrix = torch.from_numpy(np.tril(np.ones((self.num_nodes, self.num_nodes)), 1).astype(np.float32)).to(self.device)

    def forward(self,x, A,mask,k):

        B,C1,N,T=mask.shape
        B,C,N,T=x.shape

        mask_projection = self.mlp(mask.float())
        mask_weight = torch.matmul(mask_projection, mask_projection.transpose(-2, -1))
        mask_weight = mask_weight + 0.001 * self.triu_matrix * self.alpha1.triu() + 0.001 * self.tril_matrix * self.alpha2.tril()

        mask_weight = torch.sigmoid(mask_weight / torch.sqrt(torch.tensor(10.0)))
        A = A.unsqueeze(0).unsqueeze(0) + 0.002 * mask_weight

        _, topk_A = torch.topk(A, k=5, dim=-1, largest=True)
        topk_A = topk_A.unsqueeze(-1).expand(-1, C1, -1, -1, self.inputsize)
        mask = mask.unsqueeze(2).expand(-1, C1, self.num_nodes, -1, -1)
        mask_select = mask.gather(3, topk_A)
        mask ,_= torch.max(mask_select, dim=-2)

        x = torch.einsum('ncwl,nbvw->ncvl', (x, A))

        return x.contiguous(),mask

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha,inputsize,num_nodes,input_len,pred_len):
        ##16 16 2 0.3 0.05
        super(mixprop, self).__init__()
        self.nconv = nconv(inputsize,num_nodes,input_len,pred_len)
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj,mask,k,flag=0):

        adj = adj + torch.eye(adj.size(0), device=adj.device)

        d = adj.sum(1)
        h = x
        out = [h]

        a = adj / d.view(-1, 1)
        for i in range(self.gdep): #2
            #0.05
            state,mask=self.nconv(h,a,mask,k)
            h = self.alpha*x + (1-self.alpha)*state
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)

        return ho,mask

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):

        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout,kernel_set, dilation_factor=2):

        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.mconv=nn.ModuleList()
        self.kernel_set = kernel_set#[2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(weight_norm(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor))))
            self.mconv.append(nn.Conv2d(1,1,(1,kern),dilation=(1,dilation_factor)))
        # self.init_weight()
    def init_weight(self):
        for name,module in self.named_modules():
            if 'mconv' in name and isinstance(module,nn.Conv2d):
                in_size,out_size,h,w=module.weight.shape
                module.weight=nn.Parameter(torch.ones(in_size,out_size,h,w),requires_grad=False)
    def forward(self,input,mask):
        x = []
        mask_list=[]
        mask=mask[::,:1,::,::]
        for i in range(len(self.kernel_set)):
            feature_x=self.tconv[i](input)
            mask_weight=self.mconv[i](mask)
            feature_x=feature_x*mask_weight
            x.append(feature_x)
            mask_list.append(mask_weight)
            #todo add mask

        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
            mask_list[i] = mask_list[i][..., -mask_list[-1].size(3):]
        x = torch.cat(x,dim=1)
        mask=torch.cat(mask_list,dim=1)
        mask,_=torch.max(mask,dim=1)
        mask=mask.unsqueeze(1)
        return x,mask


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):

        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)

        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))

        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(self.device), requires_grad=True)
        ##to(device)
    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
