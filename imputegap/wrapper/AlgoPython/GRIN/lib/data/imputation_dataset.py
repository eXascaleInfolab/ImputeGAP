# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import numpy as np
import torch

from . import TemporalDataset, SpatioTemporalDataset


class ImputationDataset(TemporalDataset):

    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None,
                 multivariate=False):
        if mask is None:
            mask = np.ones_like(data)
        if exogenous is None:
            exogenous = dict()
        exogenous['mask_window'] = mask
        if eval_mask is not None:
            exogenous['eval_mask_window'] = eval_mask
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                exogenous=exogenous,
                                                trend=trend,
                                                scaler=scaler,
                                                freq=freq,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride,
                                                multivariate=multivariate)

    def get(self, item, preprocess=False):
        res, transform = super(ImputationDataset, self).get(item, preprocess)
        res['x'] = torch.where(res['mask'], res['x'], torch.zeros_like(res['x']))
        return res, transform


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    pass
