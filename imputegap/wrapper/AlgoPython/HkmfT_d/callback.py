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

from abc import ABC

from imputegap.wrapper.AlgoPython.HkmfT_d.hankel_methods import get_hankel_result


class AbstractConvergeCallback(ABC):
    def __init__(self):
        self._epoch = 0

    # def on_one_update(self, i, t, old_U, old_V, old_E,
    #                   new_U_i, new_V_t, new_E_it,
    #                   new_E_bar, *args, **kwargs) -> bool:
    #     """
    #     在每次随机梯度下降计算后回调，
    #     返回 False 将会结束训练。
    #     """
    #     return True

    def on_epoch(self, p, it_set, err, old_A_hat, new_A_hat,
                 Hpx_mask, Hpx_tag, verbose, *args, **kwargs) -> bool:
        """
        在每次 epoch 结束后回调，
        返回 False 将会结束训练。
        """
        if len(it_set) <= 0:
            return False
        self._epoch += 1
        x_idx = np.ndarray((len(it_set),), dtype=int)
        y_idx = np.ndarray((len(it_set),), dtype=int)
        for i, (x, y) in enumerate(it_set):
            x_idx[i] = x
            y_idx[i] = y
        avg_err = np.average(err[x_idx, y_idx])
        if verbose:
            logging.info(f'epoch {self._epoch}: {avg_err}')
        return True


class MaxDiffConvergeCallback(AbstractConvergeCallback):
    def __init__(self, threshold: float = 0.001):
        super().__init__()
        self._threshold = threshold

    def on_epoch(self, p, it_set, err, old_A_hat, new_A_hat,
                 Hpx_mask, Hpx_tag, *args, **kwargs) -> bool:
        # if not super().on_epoch(p, it_set, err, old_A_hat, new_A_hat,
        #                         Hpx_mask, Hpx_tag, *args, **kwargs):
        #     return False
        old_rs = get_hankel_result(old_A_hat, Hpx_mask, p)
        new_rs = get_hankel_result(new_A_hat, Hpx_mask, p)
        if np.max(np.abs(new_rs - old_rs)) <= self._threshold:
            return False
        else:
            return True


class EpochConvergeCallback(AbstractConvergeCallback):
    def __init__(self, max_epoch: int = None, verbose=True):
        super().__init__()
        if max_epoch is None:
            self._max_epoch = 100000000
        else:
            self._max_epoch = int(max_epoch)

    def on_epoch(self, p, it_set, err, old_A_hat, new_A_hat,
                 Hpx_mask, Hpx_tag, verbose, *args, **kwargs) -> bool:
        super().on_epoch(p, it_set, err, old_A_hat, new_A_hat,
                         Hpx_mask, Hpx_tag, verbose, *args, **kwargs)
        self._max_epoch -= 1
        if self._max_epoch <= 0:
            return False
        else:
            return True
