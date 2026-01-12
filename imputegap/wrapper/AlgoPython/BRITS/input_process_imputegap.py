# ===============================================================================================================
# SOURCE: https://github.com/caow13/BRITS
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
# ===============================================================================================================

import os
import numpy as np
import pandas as pd
import ujson as json
from imputegap.tools import utils as utils_imp


def data_loader_brits(incomp_matrix, seq_len=24, sliding_windows=1, dataset_name="json", tr_ratio=0.9, seed=42, verbose=True, deep_verbose=False):

    if verbose:
        print("\nDataset processing for BRITS logic...")

    if sliding_windows == 0:
        mulivariate = True
    else:
        mulivariate = False

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset_saving_dir = os.path.join(here, "imputegap_assets/models/brits")

    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    dataset_path = os.path.join(dataset_saving_dir, dataset_name)

    fs = open(dataset_path, 'w')  # single open

    # incomp_matrix shape: (1000, 50)
    N, M = incomp_matrix.shape

    # attributes: "sensor #1", ..., "sensor #D"
    attributes = [f"sensor_{i + 1}" for i in range(M)]

    # mean and std for each column, ignoring NaNs
    #mean = np.nanmean(incomp_matrix, axis=0)
    #std = np.nanstd(incomp_matrix, axis=0)

    if deep_verbose:
        print("attributes =", attributes)
        #print("mean =", mean)
        #print("std =", std)

    def to_time_bin(x):
        h, m = map(int, x.split(':'))
        return h

    def parse_data(x, attributes):
        # build Parameter->Value dict
        x = x.set_index('Parameter')['Value'].to_dict()
        values = []
        for attr in attributes:
            values.append(x[attr] if attr in x else np.nan)   # <- no has_key
        return values

    def parse_delta(masks, L, M, dir_):
        if dir_ == 'backward':
            masks = masks[::-1]
        deltas = []
        for h in range(L):
            if h == 0:
                deltas.append(np.ones(M))
            else:
                deltas.append(np.ones(M) + (1 - masks[h]) * deltas[-1])
        return np.array(deltas)

    def parse_rec(M, L, values, masks, evals, eval_masks, dir_):
        deltas = parse_delta(masks, L, M, dir_)
        # only used in GRU-D
        forwards = (
            pd.DataFrame(values)
              .fillna(method='ffill')
              .fillna(0.0)
              .to_numpy()                              # <- was .as_matrix()
        )
        rec = {
            'values': np.nan_to_num(values).tolist(),
            'masks': masks.astype('int32').tolist(),
            'evals': np.nan_to_num(evals).tolist(),    # imputation ground-truth
            'eval_masks': eval_masks.astype('int32').tolist(),
            'forwards': forwards.tolist(),
            'deltas': deltas.tolist(),
        }
        return rec

    def parse_id(i, id_, seq_len, seed, verbose=True, deep_verbose=False):

        evals = np.array(id_)

        has_nans = np.isnan(id_).any()
        if verbose:
            print(f"\nPre-processing for the sample id {i} {id_.shape} : {has_nans = }:")

        #evals = (np.array(evals) - mean) / std

        shp = evals.shape
        flat = evals.reshape(-1)

        if deep_verbose:
            print(f"\tNATERQ_____; {shp = }")
            print(f"\tNATERQ_____; {flat.shape = }\n")

        # randomly eliminate 10% values as the imputation ground-truth
        rng = np.random.default_rng(seed)
        indices = np.where(~np.isnan(flat))[0]

        artificial_testing = round((1-tr_ratio)*100)
        num_to_mask = int(np.round(indices.size * (artificial_testing / 100)))
        if num_to_mask == 0 and indices.size > 0:
            num_to_mask = 1  # ensure at least one

        if indices.size:
            indices = rng.choice(indices, num_to_mask, replace=False)

        values = flat.copy()

        if indices.size:
            values[indices] = np.nan

        if deep_verbose:
            print(f"\tNATERQ_____; {artificial_testing = }")
            print(f"\tNATERQ_____; {num_to_mask = }\n")
            print(f"\tNATERQ_____; {np.array(indices).shape = }\n")

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(flat))

        masks_true = masks.sum()
        eval_masks_true = eval_masks.sum()

        if deep_verbose:
            print(f"\tNATERQ_____; {flat = }\n")
            print(f"\tNATERQ_____; indices : ", *indices, "\n")
            print(f"\tNATERQ_____; {values = }\n")
            print(f"\tNATERQ_____; {masks = }\n")
            print(f"\tNATERQ_____; {eval_masks = }\n")
            print(f"\tNATERQ_____; {masks_true = }")
            print(f"\tNATERQ_____; {eval_masks_true = }\n")

        evals = flat.reshape(shp)
        values = values.reshape(shp)
        masks = masks.reshape(shp)
        eval_masks = eval_masks.reshape(shp)

        if verbose:
            print(f"\t{masks.shape = }")
            print(f"\t{eval_masks.shape = }")

        overlap = np.logical_and(masks, eval_masks)  # True where both are True
        n_overlap = overlap.sum()  # number of positions where both are True

        # If you want to check if they NEVER overlap:
        no_overlap = n_overlap == 0

        if has_nans:
            label = 1
        else:
            label = 0

        if verbose:
            print(f"masks with no overlap: {no_overlap} - label set to {label}\n")

        rec = {'label': int(label)}
        rec['forward'] = parse_rec(M, seq_len, values, masks, evals, eval_masks, dir_='forward')
        rec['backward'] = parse_rec(M, seq_len, values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

        fs.write(json.dumps(rec) + '\n')

    if mulivariate:
        ids = utils_imp.dataset_add_dimensionality(incomp_matrix, seq_len, reshapable=True, adding_nans=True, verbose=verbose, deep_verbose=False)
    else:
        ids = utils_imp.window_truncation(incomp_matrix, seq_len=seq_len, stride=sliding_windows, info="brits", verbose=verbose)

    N, T, L = ids.shape

    for i, id_ in enumerate(ids):
        try:
            parse_id(i, id_, seq_len, seed, verbose=verbose, deep_verbose=deep_verbose)
        except Exception as e:
            print(e)
            continue

    if verbose:
        if mulivariate:
            print(f"\n\nresults:\n\tnumber of samples {N}")
            print(f"\tsize of the samples {T}")
            print(f"\tnumber of features {L}\n\n")
        else:
            print(f"\n\nresults:\n\tnumber of windows {N}")
            print(f"\tsize of the windows {T}")
            print(f"\tnumber of series {L}\n\n")

    fs.close()



