# ===============================================================================================================
# SOURCE: https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/8979178
# ===============================================================================================================

# Copyright (c) [2021] [wlicsnju]
# [HKMF-T] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
# See the Mulan PSL v2 for more details.  
import logging

import numpy as np


class LinearInterpolation(object):
    def __init__(self):
        self._data = None
        self._mask = None
        self._dim = 1

    def put_and_reset(self, data, mask, tag):
        if data.shape != tag.shape:
            logging.error(f'\'data\'{data.shape} must be same to \'tag\'{tag.shape}.', ValueError)
            return
        if len(mask.shape) != 1:
            logging.error(f'\'mask\'{mask.shape} must be 1-d array.', ValueError)
            return
        if len(data.shape) == 1:
            self._data = data[np.newaxis, :]
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._dim = self._data.shape[0]
        self._mask = mask

    def train(self, *args):
        logging.warning('Linear Interpolation class do not train for result.')

    def get_result(self):
        blackout_l = np.sum(self._mask == 0)
        rs = np.zeros((self._dim, blackout_l), dtype=float)

        idx = np.arange(self._data.shape[1])
        for d in range(self._dim):
            xp = idx[self._mask > 0]
            fp = self._data[d, self._mask > 0]
            est_x = idx[self._mask <= 0]
            if len(est_x) > 0:
                rs[d, :] = np.interp(est_x, xp, fp)
        return rs
