# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, RandomSampler

from .. import TemporalDataset, SpatioTemporalDataset
from ..preprocessing import StandardScaler, MinMaxScaler
from ...utils import ensure_list
from ...utils.parser_utils import str_to_bool


class SpatioTemporalDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for TimeSeriesDatasets
    """

    def __init__(self, dataset: TemporalDataset,
                 scale=True,
                 scaling_axis='samples',
                 scaling_type='std',
                 scale_exogenous=None,
                 train_idxs=None,
                 val_idxs=None,
                 test_idxs=None,
                 imputed_idxs=None,
                 batch_size=32,
                 workers=1,
                 samples_per_epoch=None):
        super(SpatioTemporalDataModule, self).__init__()
        self.torch_dataset = dataset
        # splitting
        self.trainset = Subset(self.torch_dataset, train_idxs if train_idxs is not None else [])
        self.valset = Subset(self.torch_dataset, val_idxs if val_idxs is not None else [])
        self.testset = Subset(self.torch_dataset, test_idxs if test_idxs is not None else [])
        self.imputed_set = Subset(self.torch_dataset, imputed_idxs if imputed_idxs is not None else [])
        # preprocessing
        self.scale = scale
        self.scaling_type = scaling_type
        self.scaling_axis = scaling_axis
        self.scale_exogenous = ensure_list(scale_exogenous) if scale_exogenous is not None else None
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.samples_per_epoch = samples_per_epoch



    @property
    def is_spatial(self):
        return isinstance(self.torch_dataset, SpatioTemporalDataset)

    @property
    def n_nodes(self):
        return self.torch_dataset.n_nodes if self.is_spatial else None

    @property
    def d_in(self):
        return self.torch_dataset.n_channels

    @property
    def d_out(self):
        return self.torch_dataset.horizon

    @property
    def train_slice(self):
        return self.torch_dataset.expand_indices(self.trainset.indices, merge=True)

    @property
    def val_slice(self):
        return self.torch_dataset.expand_indices(self.valset.indices, merge=True)

    @property
    def test_slice(self):
        return self.torch_dataset.expand_indices(self.testset.indices, merge=True)

    def get_scaling_axes(self, dim='global'):
        scaling_axis = tuple()
        if dim == 'global':
            scaling_axis = (0, 1, 2)
        elif dim == 'channels':
            scaling_axis = (0, 1)
        elif dim == 'nodes':
            scaling_axis = (0,)
        # Remove last dimension for temporal datasets
        if not self.is_spatial:
            scaling_axis = scaling_axis[:-1]

        if not len(scaling_axis):
            raise ValueError(f'Scaling axis "{dim}" not valid.')

        return scaling_axis

    def get_scaler(self):
        if self.scaling_type == 'std':
            return StandardScaler
        elif self.scaling_type == 'minmax':
            return MinMaxScaler
        else:
            return NotImplementedError

    def setup(self, stage=None):

        if self.scale:
            scaling_axis = self.get_scaling_axes(self.scaling_axis)
            train = self.torch_dataset.data.numpy()[self.train_slice]
            train_mask = self.torch_dataset.mask.numpy()[self.train_slice] if 'mask' in self.torch_dataset else None
            scaler = self.get_scaler()(scaling_axis).fit(train, mask=train_mask, keepdims=True).to_torch()
            self.torch_dataset.scaler = scaler

            if self.scale_exogenous is not None:
                for label in self.scale_exogenous:
                    exo = getattr(self.torch_dataset, label)
                    scaler = self.get_scaler()(scaling_axis)
                    scaler.fit(exo[self.train_slice], keepdims=True).to_torch()
                    setattr(self.torch_dataset, label, scaler.transform(exo))

    def _data_loader(self, dataset, shuffle=False, batch_size=None, froms=None, deep_verbose=False, **kwargs):
        batch_size = self.batch_size if batch_size is None else batch_size

        if deep_verbose :
            inc = 0
            if froms == "train":
                fs = self.trainset
            elif froms == "val":
                fs = self.valset
            elif froms == "test":
                fs = self.testset
            else:
                fs = self.imputed_set

            for item in fs:  # first sample
                x = item[0]

                print(f"{froms}-{inc}: {x['x'].shape =}")
                inc = inc+1

        dload = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.workers, **kwargs)

        return dload

    def train_dataloader(self, shuffle=True, batch_size=None):
        if self.samples_per_epoch is not None:
            sampler = RandomSampler(self.trainset, replacement=True, num_samples=self.samples_per_epoch)
            return self._data_loader(self.trainset, False, batch_size, sampler=sampler, drop_last=True)
        return self._data_loader(self.trainset, shuffle, batch_size, "train", deep_verbose=False, drop_last=True)

    def val_dataloader(self, shuffle=False, batch_size=None):
        return self._data_loader(self.valset, shuffle, batch_size, "val", deep_verbose=False)

    def test_dataloader(self, shuffle=False, batch_size=None):
        return self._data_loader(self.testset, shuffle, batch_size, "test", deep_verbose=False)

    def imputed_dataloader(self, shuffle=False, batch_size=None):
        return self._data_loader(self.imputed_set, shuffle, batch_size, "imputed", deep_verbose=False)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--scaling-axis', type=str, default="channels")
        parser.add_argument('--scaling-type', type=str, default="std")
        parser.add_argument('--scale', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--samples-per-epoch', type=int, default=None)
        return parser
