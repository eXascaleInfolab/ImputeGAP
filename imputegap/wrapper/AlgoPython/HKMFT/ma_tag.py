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

from imputegap.wrapper.AlgoPython.HKMFT.hankel_methods import find_max_continue_0


class MATag(object):
    def __init__(self):
        self._data = None
        self._mask = None
        self._tag = None
        self._dim = 1
        self._q = 1

    def put_and_reset(self, data, mask, tag, q=None):
        if data.shape != tag.shape:
            logging.error(f'\'data\'{data.shape} must be same to \'tag\'{tag.shape}.', ValueError)
            return
        if len(mask.shape) != 1:
            logging.error(f'\'mask\'{mask.shape} must be 1-d array.', ValueError)
            return
        l_b = find_max_continue_0(mask)
        if q is None:
            q = l_b + 2
        if q < l_b + 2:
            logging.warning(f'\'p\' is less than length of blackouts {l_b}+2, set \'p\' to {l_b + 2}.')
            q = l_b + 2
        if len(data.shape) == 1:
            self._data = data[np.newaxis, :]
            tag = tag[np.newaxis, :]
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._q = q
        self._dim = self._data.shape[0]
        self._mask = mask
        self._tag = tag

    def train(self, *args):
        logging.warning('MA Tag class do not train for result.')

    def get_result(self):
        blackout_l = np.sum(self._mask == 0)
        idx = np.arange(self._data.shape[1])

        moving_average = np.zeros(self._data.shape, dtype=float)
        for d in range(self._dim):
            for i in range(self._data.shape[1]):
                sum_val = 0.0
                count = 0
                for j in range(max(0, i - self._q), i + 1):
                    if self._mask[j]:
                        sum_val += self._data[d, j]
                        count += 1
                if count > 0:
                    moving_average[d, i] = sum_val / float(count)
        tag_non = np.array(self._tag)
        tag_non[:, self._mask == 0] = -1
        tag_count = np.max(self._tag) + 1
        mean_diff = {}
        for d in range(self._dim):
            for tag in range(tag_count):
                if tag not in mean_diff:
                    mean_diff[tag] = np.zeros((self._dim, ))
                diff = self._data[d, tag_non[d, :] == tag] - moving_average[d, tag_non[d, :] == tag]
                if len(diff) > 0:
                    mean_diff[tag][d] = np.average(diff)
        for d in range(self._dim):
            xp = idx[self._mask > 0]
            fp = moving_average[d, self._mask > 0]
            est_x = idx[self._mask <= 0]
            if len(est_x) > 0:
                moving_average[d, self._mask <= 0] = np.interp(est_x, xp, fp)

        rs = np.zeros((self._dim, blackout_l), dtype=float)
        for d in range(self._dim):
            j = 0
            for i, m in enumerate(self._mask):
                if m == 0:
                    rs[d, j] = moving_average[d, i] + mean_diff[self._tag[d, i]]
                    j += 1
        return rs
