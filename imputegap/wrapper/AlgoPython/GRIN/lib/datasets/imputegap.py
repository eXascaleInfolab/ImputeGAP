# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import os

import numpy as np
import pandas as pd

from imputegap.wrapper.AlgoPython.GRIN.lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask
import imputegap.tools.utils as utils_imp


class ImputeGAP(PandasDataset):
    def __init__(self, ts_m):
        df, dist, mask = self.load(ts_m)
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='bay', freq='5T', aggr='nearest')

    def load(self, ts_m, impute_zeros=True, deep_verbose=False):
        #N, S, F = ts_m.shape
        N, F = ts_m.shape

        # Build a proper DateTimeIndex with known frequency
        start = pd.Timestamp('2000-01-01 00:00:00')  # any anchor is fine
        index = pd.date_range(start=start, periods=N, freq='5min')

        # Create DataFrame with time on rows, sensors on columns
        df = pd.DataFrame(ts_m, index=index)

        datetime_idx = sorted(df.index)

        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5min')

        df = df.reindex(index=date_range)

        mask = ~np.isnan(df.values)
        df.ffill(inplace=True)   # forward fill

        dist = np.ones(shape=(df.shape[0], df.shape[0]))
        #dist = self.load_distance_matrix_imputegap(list(df.columns), ts_m)

        if deep_verbose:
            print(f"\n\t{df.shape =}")
            print(f"\t{dist.shape =}")
            print(f"\t{mask.shape =}\n")

        return df.astype('float32'), dist, mask.astype(bool)   # or just mask

    def load_distance_matrix_imputegap(self, ids, ts_m):
        """
        Build synthetic distance matrix using pairwise similarity between time series.

        ts_m : np.ndarray of shape (T, N)
            Time series matrix where each column is a node/sensor.
        """
        num_sensors = len(ids)
        # Compute 1 - correlation as a pseudo-distance
        corr = np.corrcoef(ts_m.T)  # (N, N)
        dist = 1 - corr
        # Replace NaNs (from constant columns) by 1
        dist = np.nan_to_num(dist, nan=1.0)
        # Ensure no self-distance
        np.fill_diagonal(dist, 0.0)

        return dist.astype(np.float32)

    def load_distance_matrix(self, ids):
        path = os.path.join(datasets_path['bay'], 'pems_bay_dist.npy')
        try:
            dist = np.load(path)
        except:
            distances = pd.read_csv(os.path.join(datasets_path['bay'], 'distances_bay.csv'))
            num_sensors = len(ids)
            dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
            # Builds sensor id to index map.
            sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

            # Fills cells in the matrix with distances.
            for row in distances.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
            np.save(path, dist)
        return dist

    def get_similarity_imputegap(self, type='uniform', thr=0.1, force_symmetric=False, sparse=False, ts_m=None):
        """
        Return a simple or computed similarity matrix among nodes.
        If no distance data is available, defaults to a fully connected graph.

        Parameters
        ----------
        type : str
            'uniform' for all-ones adjacency
            'corr' for correlation-based similarity
            'dcrnn' or 'stcn' for traditional formulas (if self.dist is available)
        ts_m : np.ndarray, optional
            Time series matrix (T, N), required if type='corr'
        """
        import numpy as np

        # Fallback: if self.dist is missing or None
        if getattr(self, 'dist', None) is None and type not in ('uniform', 'corr'):
            type = 'uniform'

        if type == 'uniform':
            n = self.df.shape[1]  # number of sensors/nodes
            adj = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(adj, 0.0)

        elif type == 'corr':
            if ts_m is None:
                ts_m = self.df.values
            # Compute Pearson correlation between node signals
            corr = np.corrcoef(ts_m.T)
            corr = np.nan_to_num(corr, nan=0.0)
            # Convert correlation to similarity (positive only)
            adj = np.maximum(corr, 0.0)
            np.fill_diagonal(adj, 0.0)

        elif type == 'dcrnn':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std() if finite_dist.size > 0 else 1.0
            adj = np.exp(-np.square(self.dist / sigma))

        elif type == 'stcn':
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)

        else:
            raise NotImplementedError(f"Unknown type '{type}'")

        # Thresholding and post-processing
        adj[adj < thr] = 0.0
        if force_symmetric:
            adj = np.maximum(adj, adj.T)
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj


    def get_similarity(self, type='dcrnn', thr=0.1, force_symmetric=False, sparse=False):
        """
        Return similarity matrix among nodes. Implemented to match DCRNN.

        :param type: type of similarity matrix.
        :param thr: threshold to increase saprseness.
        :param trainlen: number of steps that can be used for computing the similarity.
        :param force_symmetric: force the result to be simmetric.
        :return: and NxN array representig similarity among nodes.
        """
        if type == 'dcrnn':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            adj = np.exp(-np.square(self.dist / sigma))
        elif type == 'stcn':
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)
        else:
            raise NotImplementedError
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesImputeGAP(ImputeGAP):
    SEED = 56789

    def __init__(self, ts_m, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesImputeGAP, self).__init__(ts_m)
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=12,
                                max_seq=12 * 4,
                                rng=self.rng)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitters(self, dataset, val_len=0.2, test_len=0.4, window=0):
        import math

        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len =  math.ceil(test_len * len(idx))
        if val_len < 1:
            val_len = math.ceil(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start], idx[val_start:test_start], idx[test_start:]]

    def splitter_imputegap(self, dataset, val_len=0, test_len=0, window=0):
        return np.arange(len(dataset))
