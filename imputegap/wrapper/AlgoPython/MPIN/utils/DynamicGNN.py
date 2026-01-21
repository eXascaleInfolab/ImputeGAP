import copy
import os
import random
import sys
import time
sys.path.append('/home/xiao/Documents/OCW')
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import chain
import json
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

from torch_geometric.nn import knn_graph,radius_graph

# class DynamicEdgeConv(EdgeConv):
#     def __init__(self, in_channels, out_channels, k=6):
#         super().__init__(in_channels, out_channels)
#         self.k = k
#
#     def forward(self, x, batch=None):
#         edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
#         return super().forward(x, edge_index)


class DynamicGAT(pyg_nn.GATConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGCN(pyg_nn.GCNConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class DynamicGraphSAGE(pyg_nn.SAGEConv):

    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(DynamicGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, batch=None):
        if self.k is not None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        else:
            edge_index = radius_graph(x, self.radius, loop=False)

        return super().forward(x, edge_index)

class StaticGraphSAGE(pyg_nn.SAGEConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGraphSAGE, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index


class StaticGCN(pyg_nn.GCNConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGCN, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index

class StaticGAT(pyg_nn.GATConv):
    def __init__(self, in_channels, out_channels, k=None, radius=None):
        super(StaticGAT, self).__init__(in_channels, out_channels)
        self.radius = radius
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index == None:
            if self.k is not None:
                edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            else:
                edge_index = radius_graph(x, self.radius, loop=False)
            return super().forward(x, edge_index), edge_index
        else:
            return super().forward(x, edge_index), edge_index



