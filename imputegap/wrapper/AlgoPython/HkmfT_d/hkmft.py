# ===============================================================================================================
# SOURCE: https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/8979178
# ===============================================================================================================

# [HKMF-T] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
# See the Mulan PSL v2 for more details.  
import logging

import numpy as np

from imputegap.wrapper.AlgoPython.HkmfT_d.callback import AbstractConvergeCallback
from collections import namedtuple
from typing import Iterable

from imputegap.wrapper.AlgoPython.HkmfT_d.hankel_methods import get_hankel_result, find_max_continue_0, hankelization

HKMFTTrainParam = namedtuple('HKMFTTrainParam',
                             ['eta', 'lambda_s', 'lambda_o', 'lambda_e', 'stop_rate', 'random', 'verbose'])
HKMFT_MAX_EPOCH = 5000


class HKMFT(object):
    def __init__(self):
        self._p = 0
        self._Hpx = None
        self._Hpx_mask = None
        self._Hpx_tag = None
        self._U = None
        self._V = None
        self._E = None
        self._A_hat = None
        self._data = None
        self._data_original = None

    def put_and_reset(self, data, mask, tag, original, r=None, p=None):
        if data.shape != tag.shape:
            logging.error(f'\'data\'{data.shape} must be same to \'tag\'{tag.shape}.', ValueError)
            return
        if len(mask.shape) != 1:
            logging.error(f'\'mask\'{mask.shape} must be 1-d array.', ValueError)
            return
        l_b = find_max_continue_0(mask)
        if p is None:
            p = l_b + 1
        if p < l_b + 1:
            logging.warning(f'\'p\' is less than length of blackouts {l_b}+1, set \'p\' to {l_b + 1}.')
            p = l_b + 1
        if r is None:
            r = p
        if r > p:
            logging.warning(f'\'r\' is great than \'p\' {p}, set \'r\' to {p}.')
            r = p
        if len(data.shape) == 1:
            self._data = data[np.newaxis, :]
            tag = tag[np.newaxis, :]
        elif len(data.shape) == 2:
            self._data = data
        elif len(data.shape) > 2:
            logging.error('Do not support data with dim > 2.', ValueError)
            return
        self._p = p
        data_d = self._data.shape[0]
        data_l = self._data.shape[1]
        self._U = np.random.rand(data_d * p, r)
        self._V = np.random.rand(r, data_l)
        self._E = np.random.rand(data_d * p, data_l)
        self._A_hat = (self._U @ self._V) + self._E
        self._Hpx, self._Hpx_mask, self._Hpx_tag = hankelization(data, mask, tag, self._p)
        self._data_original = original

    def train(self, param: HKMFTTrainParam, callbacks: Iterable[AbstractConvergeCallback]):
        epoch = 0
        data_l = self._Hpx.shape[1]
        it_set = set()
        all_it_list = []
        for i, _ in enumerate(self._Hpx_mask):
            for t, x in enumerate(_):
                if x != 0:
                    it_set.add((i, t))
                    all_it_list.append((i, t))
        max_eliminate = int(0.005 * len(it_set))
        stop_it = set()
        lr = param.eta * 2.0
        while epoch < HKMFT_MAX_EPOCH:
            epoch += 1
            e = np.zeros(self._Hpx.shape, dtype=np.float64)
            if param.random:
                it_tmp = list(it_set)
                np.random.shuffle(it_tmp)
            else:
                it_tmp = all_it_list
            for i, t in it_tmp:
                if (i, t) not in it_set:
                    continue
                err_it = self._Hpx[i, t] - ((self._U[i, :] @ self._V[:, t]) + self._E[i, t])
                e[i, t] = err_it
                new_U_i_V_t = self._U[i, :] @ self._V[:, t]
                new_U_i_V_t_s1 = self._U[i, :] @ self._V[:, t - 1]
                # if t > 0:
                #     new_U_i_V_t_s1 = self._U[i, :] @ self._V[:, t - 1]
                # else:
                #     new_U_i_V_t_s1 = 0.0
                new_U_i_V_t_p1 = self._U[i, :] @ self._V[:, (t + 1) % data_l]
                # if t < data_l - 1:
                #     new_U_i_V_t_p1 = self._U[i, :] @ self._V[:, t + 1]
                # else:
                #     new_U_i_V_t_p1 = 0.0
                # U_i
                U_i = self._U[i, :] + (lr * ((err_it * self._V[:, t]) -
                                             (param.lambda_o * self._U[i, :])))
                # if t > 0:
                U_i -= (lr * param.lambda_s *
                        (new_U_i_V_t - new_U_i_V_t_s1) *
                        (self._V[:, t] - self._V[:, t - 1]))
                # if t < data_l - 1:
                U_i -= (lr * param.lambda_s *
                        (new_U_i_V_t_p1 - new_U_i_V_t) *
                        (self._V[:, (t + 1) % data_l] - self._V[:, t]))
                # V_t
                V_t = self._V[:, t] + (lr * ((err_it * self._U[i, :]) -
                                             (param.lambda_o * self._V[:, t]) -
                                             (2.0 * param.lambda_s * new_U_i_V_t * self._U[i, :])))
                # if t > 0:
                V_t += (lr * param.lambda_s *
                        new_U_i_V_t_s1 * self._U[i, :])
                # if t < data_l - 1:
                V_t += (lr * param.lambda_s *
                        new_U_i_V_t_p1 * self._U[i, :])
                # E_ij
                E_bar = np.average(np.compress((self._Hpx_mask.ravel() != 0) &
                                               (self._Hpx_tag.ravel() == self._Hpx_tag[i, t]),
                                               self._E))
                # E_bar = np.average(self._E[(self._Hpx_mask != 0) & (self._Hpx_tag == self._Hpx_tag[i, t])])
                E_it = self._E[i, t] + (lr * (err_it -
                                              (param.lambda_o * self._E[i, t]) -
                                              (param.lambda_e * (self._E[i, t] - E_bar))))
                # for c in callbacks:
                #     if isinstance(c, AbstractConvergeCallback):
                #         if not c.on_one_update(i, t, self._U, self._V, self._E, U_i, V_t, E_it, E_bar):
                #             self._U[i, :] = U_i
                #             self._V[:, t] = V_t
                #             self._E[i, t] = E_it
                #             logging.info(f'Early stop in {c.__class__.__name__}.on_one_update().')
                #             return
                self._U[i, :] = U_i
                self._V[:, t] = V_t
                self._E[i, t] = E_it
            new_A_hat = (self._U @ self._V) + self._E
            for c in callbacks:
                if isinstance(c, AbstractConvergeCallback):
                    if not c.on_epoch(self._p, it_set, e, self._A_hat, new_A_hat, self._Hpx_mask, self._Hpx_tag, param.verbose):
                        self._A_hat = new_A_hat
                        if param.verbose:
                            logging.info(f'Early stop in {c.__class__.__name__}.on_epoch().')
                        return
            self._A_hat = new_A_hat
            if len(it_set) / (len(stop_it) + len(it_set)) <= param.stop_rate:
                continue
            # eliminate upper and lower 0.5% case
            error_sort_idx = np.unravel_index(e.argsort(axis=None), e.shape)
            for x_idx, y_idx in ((error_sort_idx[0], error_sort_idx[1]),
                                 (error_sort_idx[0][::-1], error_sort_idx[1][::-1])):
                eliminate_count = 0
                for i, t in zip(x_idx, y_idx):
                    if ((len(it_set) / (len(stop_it) + len(it_set)) <= param.stop_rate) or
                            (eliminate_count >= max_eliminate)):
                        break
                    if (i, t) in it_set:
                        it_set.remove((i, t))
                        stop_it.add((i, t))
                        eliminate_count += 1

    def get_result(self):
        tag_mean = {}
        for i, _ in enumerate(self._Hpx_mask):
            for t, x in enumerate(_):
                tag = self._Hpx_tag[i, t]
                if x == 0 and tag >= 0:
                    if tag not in tag_mean:
                        tag_mean[tag] = np.average(np.compress((self._Hpx_mask.ravel() != 0) &
                                                               (self._Hpx_tag.ravel() == tag),
                                                               self._E))
                        # tag_mean[tag] = np.average(self._E[(self._Hpx_mask != 0) & (self._Hpx_tag == tag)])
                    self._E[i, t] = tag_mean[tag]
        self._A_hat = (self._U @ self._V) + self._E
        return get_hankel_result(self._A_hat, self._Hpx_mask, self._p)


    def get_imputed_matrix(self):
        hankel_rec = get_hankel_result(self._A_hat, self._Hpx_mask, self._p)  # re
        data_imputed = self._data_original.copy()

        nan_mask = np.isnan(data_imputed)  # shape (1, 150)
        data_imputed[nan_mask] = hankel_rec.ravel()

        return data_imputed

