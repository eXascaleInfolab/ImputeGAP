# ===============================================================================================================
# SOURCE: https://github.com/LMZZML/PriSTI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/10184808
# ===============================================================================================================
import os

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from sklearn.metrics.pairwise import haversine_distances


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_adj_AQI36():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data/pm25/SampleData/pm25_latlng.txt")
    df = pd.read_csv(path)
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj


def get_similarity_metrla(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/metr_la/metr_la_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_similarity_pemsbay(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/pems_bay/pems_bay_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


import numpy as np
import torch

def get_similarity_imputegap(train_loader, thr=0.1, force_symmetric=False, sparse=False):
    import numpy as np
    import torch
    from scipy.spatial.distance import pdist, squareform

    all_data = []

    for batch in train_loader:
        data = batch['observed_data']  # Correct key from your dict
        all_data.append(data)

    all_data = torch.cat(all_data, dim=0)  # Shape: [B, T, N] or [B, N, T]

    if all_data.ndim == 3:
        all_data = all_data.permute(0, 2, 1).reshape(-1, all_data.shape[2])  # -> [B*T, N]

    data_np = all_data.cpu().numpy()
    data_np = np.nan_to_num(data_np)  # Replace NaNs with zero

    dist = squareform(pdist(data_np.T, metric='euclidean'))  # Distance between features
    sigma = dist[~np.isinf(dist)].std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.0

    if force_symmetric:
        adj = np.maximum(adj, adj.T)
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)

    return adj


# in Graph-wavenet
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support


def adj_from_corr(X_train, k=10, thr=None, sym=True):
    # X_train: (T, N)  T timesteps, N sensors
    C = np.corrcoef(X_train, rowvar=False)          # (N, N)
    S = np.abs(C)                                   # similarity
    np.fill_diagonal(S, 0.0)

    if thr is not None:                             # threshold (optional)
        S = np.where(S >= thr, S, 0.0)

    if k is not None:                               # top-k per node (sparsify)
        N = S.shape[0]
        idx = np.argpartition(S, -k, axis=1)[:, -k:]
        mask = np.zeros_like(S, dtype=bool)
        mask[np.arange(N)[:,None], idx] = True
        S = np.where(mask, S, 0.0)

    if sym:
        S = np.maximum(S, S.T)                      # undirected

    return S.astype(np.float32)