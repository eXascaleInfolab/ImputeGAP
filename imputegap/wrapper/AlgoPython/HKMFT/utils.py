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
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from typing import Union

from imputegap.wrapper.AlgoPython.HKMFT.dataloder import DataLoader
from imputegap.wrapper.AlgoPython.HKMFT.dataloder_imputegap import DataLoaderImputeGAP
from imputegap.wrapper.AlgoPython.HKMFT.dataloder import Dataset
from imputegap.wrapper.AlgoPython.HKMFT.dataloder_imputegap import DatasetImputeGAP


def show_gt_rs(gt: np.ndarray, rs: np.ndarray, begin_idx, end_idx, method_name: str = 'HKMF-T') -> None:
    if gt.shape != rs.shape:
        logging.error(f'Ground truth shape {gt.shape} do not match to result shape {rs.shape}!', ValueError)
        return
    if len(gt.shape) == 2 and gt.shape[0] == 1:
        gt = gt[0, :]
        rs = rs[0, :]
    if len(gt.shape) == 1:
        plt.plot(range(begin_idx, end_idx), gt, label='ground truth')
        plt.plot(range(begin_idx, end_idx), rs, label=method_name)
        plt.legend()
        plt.show()
    elif len(gt.shape) == 2:
        for i in range(gt.shape[0]):
            plt.plot(range(begin_idx, end_idx), gt[i], label=f'ground truth[{i}]')
            plt.plot(range(begin_idx, end_idx), rs[i], label=f'{method_name}[{i}]')
        plt.xticks(range(begin_idx, end_idx))
        plt.legend()
        plt.show()
    else:
        logging.error(f'Ground truth {gt.shape} must 1-dim or 2-dim!', ValueError)
        return


def lens_to_list(lens: Union[int, str]) -> Union[list, None]:
    if isinstance(lens, int):
        lens_list = [lens, ]
    elif isinstance(lens, str):
        _ = lens.split('-')
        if len(_) != 2:
            logging.error(f'blackouts_lens {lens} must int or str(int-int)!', ValueError)
            return None
        lens_list = range(int(_[0]), int(_[1]) + 1)
    else:
        logging.error(f'blackouts_lens {lens} must int or str(int-int)!', ValueError)
        return None
    return lens_list


def dataset_load(dataset) -> Union[DataLoader, None]:
    ds = None

    if isinstance(dataset, str):
        for d in Dataset:
            if dataset == d.name:
                ds = d
                break
        if ds is None:
            logging.error(f'dataset {dataset} do not exist!', ValueError)
            return None

        return DataLoader(ds)

def dataset_load_imputegap(dataset, strategy="full", seq_len=7, tags=None, verbose=True, deep_verbose=False) -> Union[DataLoaderImputeGAP, None]:
    ds = None
    return DataLoaderImputeGAP(dataset, time_window=None, strategy=strategy, seq_len=seq_len, tags=tags, verbose=verbose, deep_verbose=deep_verbose)



def result_save(filename: str, params: dict, start_idx: list, results: list) -> None:
    with open(filename, 'wb') as fp:
        pickle.dump({
            'params': params,
            'start_idx': start_idx,
            'results': results,
        }, fp)
        fp.close()


def _find_fall(li: list, s: int) -> int:
    r = s
    for i in range(s + 1, len(li)):
        if li[i] < li[r]:
            return i
        r += 1


def results_load(filenames: Union[tuple, list, str]) -> dict:
    """
    load result files, return dict like {'dataset_name': {blackout_lens: (start_idx, results, params)}}
    :param filenames: list of filenames.
    :return: dict.
    """
    if isinstance(filenames, str):
        filenames = (filenames, )
    results = {}
    for fn in filenames:
        if os.path.exists(fn):
            with open(fn, 'rb') as fp:
                try:
                    obj = pickle.load(fp)
                except IOError:
                    logging.warning(f'Input file {fn} is not a pickle file.')
                else:
                    if (isinstance(obj, dict) and
                            ('params' in obj) and
                            ('start_idx' in obj) and
                            ('results' in obj)):
                        ds_name = obj['params']['dataset']
                        lens = lens_to_list(obj['params']['blackouts_lens'])
                        if ds_name not in results:
                            results[ds_name] = {}
                        l = 0
                        for i in lens:
                            r = _find_fall(obj['start_idx'], l)
                            if i not in results[ds_name]:
                                params = dict(obj['params'])
                                params['blackouts_lens'] = i
                                results[ds_name][i] = (obj['start_idx'][l:r],
                                                       obj['results'][l:r],
                                                       params)
                            else:
                                logging.warning(f'Input file {fn} {ds_name} {i} repeated, ignore.')
                            l = r
                    else:
                        logging.warning(f'Input file {fn} is not a hkmf results file.')
        else:
            logging.warning(f'Input file {fn} do not exists.')
    return results
