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


def find_max_continue_0(m: np.ndarray):
    if len(m.shape) == 1:
        tmp = [m]
    elif len(m.shape) == 2:
        tmp = m
    else:
        logging.error('Do not support mask with dim > 2.', ValueError)
        return 0
    max_con0 = 0
    for t in tmp:
        count = 0
        for x in t:
            if x != 0:
                max_con0 = max(max_con0, count)
                count = 0
            else:
                count += 1
    return max_con0


def hankelization(data, mask, tag, p):
    if data.shape != tag.shape or len(mask.shape) != 1:
        return
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
        tag = tag[np.newaxis, :]
    elif len(data.shape) == 2:
        pass
    elif len(data.shape) > 2:
        return
    data_d = data.shape[0]
    data_l = data.shape[1]
    Hpx = np.zeros((data_d * p, data_l))
    Hpx_mask = np.zeros((data_d * p, data_l), dtype=np.int64)
    Hpx_tag = np.full((data_d * p, data_l), -1, dtype=np.int64)
    for i in range(len(data)):
        for j in range(p):
            Hpx[i * p + j, :(data_l - j)] = data[i, j:]
            Hpx_mask[i * p + j, :(data_l - j)] = mask[j:]
            Hpx_tag[i * p + j, :(data_l - j)] = tag[i, j:]
    return Hpx, Hpx_mask, Hpx_tag


def get_hankel_result(A: np.ndarray, mask: np.ndarray, p: int) -> np.ndarray:
    blackout_l = np.sum(mask[0, :] == 0)
    data_d = mask.shape[0] // p
    data_l = mask.shape[1]
    rs = np.zeros((data_d, blackout_l), dtype=float)
    for i in range(data_d):
        for j in range(p):
            c = 0
            for k, x in enumerate(mask[i * p + j, :(data_l - j)]):
                if x == 0:
                    rs[i, c] += A[i * p + j, k]
                    c += 1
        rs[i, :] /= float(p)
    return rs
