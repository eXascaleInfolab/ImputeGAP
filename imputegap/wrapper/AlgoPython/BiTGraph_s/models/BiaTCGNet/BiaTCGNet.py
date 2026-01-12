from imputegap.wrapper.AlgoPython.BiTGraph_s.models.BiaTCGNet.BiaTCGNet_layer import *
import torch.nn as nn
import torch
#from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm


class Model(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes,kernel_set, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=5, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_len=12, out_dim=1, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(Model, self).__init__()
        self.gcn_true = gcn_true #true
        self.buildA_true = buildA_true #true
        self.num_nodes = num_nodes #137
        self.kernel_set= kernel_set
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.output_dim=out_dim
        self.device = device
        self.start_conv = nn.Conv2d(in_channels=in_dim, #1
                                    out_channels=residual_channels, #16
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.seq_length = seq_length
        kernel_size = self.kernel_set[-1]#7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            dilationsize=[]#[18,12,6]
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                #print(f"{seq_length = }")
                #print(f"{kernel_size = }")
                #print(f"{j = }")
                #print(f"{(kernel_size-1)*j = }")
                assert (seq_length-(kernel_size-1)*j) >0, 'Please decrease the kernel size or increase the input length'
                dilationsize.append(seq_length-(kernel_size-1)*j)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels,kernel_set, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels,kernel_set, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha,dilationsize[j-1],num_nodes,self.seq_length,out_len,self.device))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha,dilationsize[j-1],num_nodes,self.seq_length,out_len,self.device))

                if self.seq_length>self.receptive_field: #
                    self.norm.append(LayerNorm((residual_channels, num_nodes,self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes,  self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline)) #

                new_dilation *= dilation_exponential #2

        self.layers = layers
        self.end_conv_1 = weight_norm(nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True))
        self.end_conv_2 = weight_norm(nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_len*out_dim,
                                             kernel_size=(1,1),
                                             bias=True))
        if self.seq_length > self.receptive_field:
            self.skip0 = weight_norm(nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True))
            self.skipE = weight_norm(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True))

        else:
            self.skip0 = weight_norm(nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True))
            self.skipE = weight_norm(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True))


        self.idx = torch.arange(self.num_nodes, device=self.device)


    def forward(self, input,mask, k, idx=None):#tx,id

        input=input.transpose(1,3)
        mask=mask.transpose(1,3).float()

        input=input*mask
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        if self.gcn_true:
            if self.buildA_true:#True
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)

        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers): #5
            residual = x
            filter,mask_filter = self.filter_convs[i](x,mask)
            filter = torch.tanh(filter)
            gate,mask_gate = self.gate_convs[i](x,mask)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_true:
                state1,mask=self.gconv1[i](x, adp,mask_filter,k,flag=0)
                state2,mask2=self.gconv2[i](x, adp.transpose(1,0),mask_filter,k,flag=0)
                x = state1+state2
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        B,T,N,D=x.shape
        x=x.reshape(B,-1,self.output_dim,N)

        x=x.permute(0,1,3,2)

        return x
