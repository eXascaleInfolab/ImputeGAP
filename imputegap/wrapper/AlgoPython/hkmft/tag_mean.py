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


class TagMean(object):
    def __init__(self):
        self._data = None
        self._mask = None
        self._tag = None
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
            tag = tag[np.newaxis, :]
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._dim = self._data.shape[0]
        self._mask = mask
        self._tag = tag

    def train(self, *args):
        logging.warning('Tag Mean class do not train for result.')

    def get_result(self):
        all_avg = np.zeros((self._dim, ))
        tag_sum = {}
        tag_count = {}
        for i, m in enumerate(self._mask):
            tag = self._tag[0, i]
            if m > 0:
                if tag not in tag_sum:
                    tag_sum[tag] = np.zeros((self._dim, ))
                    tag_count[tag] = 0
                all_avg += self._data[:, i]
                tag_sum[tag] += self._data[:, i]
                tag_count[tag] += 1
        all_avg = all_avg / float(np.sum(self._mask != 0))

        blackout_l = np.sum(self._mask == 0)
        rs = np.zeros((self._dim, blackout_l), dtype=float)

        j = 0
        for i, m in enumerate(self._mask):
            tag = self._tag[0, i]
            if m == 0:
                if tag not in tag_sum:
                    rs[:, j] = all_avg
                else:
                    rs[:, j] = tag_sum[tag] / float(tag_count[tag])
                j += 1
        return rs
