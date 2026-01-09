# ===============================================================================================================
# SOURCE: https://github.com/ermongroup/CSDI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://neurips.cc/virtual/2021/poster/26846
# ===============================================================================================================

import math
import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import imputegap.tools.utils as utils_imp

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1, seq_len=48, i=0, imputegap=False, deep_verbose=False, set=None):

    if not imputegap:

        """
        observed_values = np.array(id_)
        observed_masks = ~np.isnan(id_)

        # randomly set some percentage as ground-truth
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(obs_indices, math.ceil(len(obs_indices) * missing_ratio), replace=False)
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)

        # print(f"{masks = }")
        # print(f"{gt_masks = }\n\n\n")

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")
        """
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "data/physio/set-a/"+str(id_)+".txt")

        data = pd.read_csv(config_path)
        # set hour
        data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

        # create data for 48 hours x 35 attributes
        observed_values = []
        for h in range(48):
            observed_values.append(parse_data(data[data["Time"] == h]))
        observed_values = np.array(observed_values)
        observed_masks = ~np.isnan(observed_values)

        # randomly set some percentage as ground-truth
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False)
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")
    else:
        data = np.array(id_)
        observed_values = np.array(data)

        observed_values = np.array(observed_values)
        observed_masks = ~np.isnan(observed_values)

        #if set is not None:
        #    mean_val = np.nanmean(observed_values)
        #    nan_mask = np.isnan(observed_values)
        #    observed_values[nan_mask] = mean_val


        #print(f"\n\n\n{observed_values = }")
        #print(f"{observed_masks = }")

        # randomly set some percentage as ground-truth
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(obs_indices, math.ceil(len(obs_indices) * missing_ratio), replace=False)
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)

        #print(f"{masks = }")
        #print(f"{gt_masks = }\n\n\n")

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")

        #print(f"\n\n{observed_values = }")
        #print(f"\n\n{observed_masks = }")
        #print(f"\n\n{gt_masks = }\n\n")

    return observed_values, observed_masks, gt_masks


def get_idlist():
    patient_id = []
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "data/physio/set-a")

    print(f"{config_path = }\n")
    for filename in os.listdir(config_path):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "data/replicat.pk")
        patient_path = os.path.join(here, "data/small_patient.txt")
        path = config_path

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            print(f"loading the dataset...")
            idlist = get_idlist()
            for id_ in idlist:
                try:
                    observed_values, observed_masks, gt_masks = parse_id(
                        id_, missing_ratio
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            tmp_values = self.observed_values.reshape(-1, 35)
            tmp_masks = self.observed_masks.reshape(-1, 35)
            mean = np.zeros(35)
            std = np.zeros(35)
            for k in range(35):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            #print(f"{self.observed_values = }")
            #print(f"{self.observed_values.shape = }")

            #recovery_matrix = utils_imp.dataset_reverse_dimensionality(matrix=self.observed_values, expected_n=7824, verbose=True)

            #print(f"{recovery_matrix = }")
            #print(f"{recovery_matrix.shape = }")
            save_txt=False
            recovery_matrix=None
            if save_txt:
                with open(patient_path, "w", encoding="utf-8") as f:
                    for row in recovery_matrix:
                        # 7 chars wide, 3 decimals, zero-padded -> e.g. 000.000, -04.633
                        row_str = " ".join(f"{val:07.3f}" for val in row)
                        f.write(row_str + "\n")


            #with open(path, "wb") as f:
                    #    pickle.dump(
                # [self.observed_values, self.observed_masks, self.gt_masks], f
                #)
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)



class ImputeGAP_Dataset(Dataset):
    def __init__(self, ts_m=None, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0, features=35, verbose=True, set="", normalize=False):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        #path = ("./data/imputegap.pk")
        self.set = set
        self.ts_m = ts_m


        ids = utils_imp.dataset_add_dimensionality(ts_m, self.eval_length, reshapable=True, adding_nans=True, verbose=verbose, deep_verbose=False)
        #idlist = get_idlist()

        if verbose:
            print(f"preprocessing CSDI: creating and formatting each sample with author contamination: {len(ids)}, for index list : {use_index_list}")

        sampled=True
        if sampled:
            for i, id_ in enumerate(ids):
                try:
                    observed_values, observed_masks, gt_masks = parse_id(id_=id_, missing_ratio=missing_ratio, seq_len=self.eval_length, i=i, imputegap=True, set=set)
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)

                except Exception as e:
                    print(id_, e)
                    continue
        else:
            try:
                observed_values, observed_masks, gt_masks = parse_id(id_=ids, missing_ratio=missing_ratio, seq_len=self.eval_length, i=0, imputegap=True, set=set)
                self.observed_values = observed_values
                self.observed_masks = observed_masks
                self.gt_masks = gt_masks
            except Exception as e:
                print(ids, e)

        self.observed_values = np.array(self.observed_values)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)

        if verbose==False:
            print(f"{self.set}: {self.observed_values.shape = }")
            print(f"{self.set}: {self.observed_masks.shape = }")
            print(f"{self.set}: {self.gt_masks.shape = }\n")

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = self.observed_values.reshape(-1, features)
        tmp_masks = self.observed_masks.reshape(-1, features)

        if normalize:
            mean = np.zeros(features)
            std = np.zeros(features)
            for k in range(features):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = ((self.observed_values - mean) / std * self.observed_masks)

        #with open(path, "wb") as f:
        #    pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list


    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, replicat=False, ts_m=None, seq_len=24, features=35, num_workers=0, shuffle=True, tr_ratio=0.7, normalize=True, verbose=True):

    if not replicat:
        dataset = ImputeGAP_Dataset(ts_m=ts_m, missing_ratio=missing_ratio, seed=seed, eval_length=seq_len, features=features, set="load")

        ts_ratio, val_ratio = utils_imp.sets_splitter_based_on_training(tr_ratio)

        if verbose:
            print(f"{ts_ratio = }, {val_ratio = }, {tr_ratio = }, {shuffle = }")

        if shuffle:
            indlist = np.arange(len(dataset))
            np.random.seed(seed)
            np.random.shuffle(indlist)

            start = (int)(nfold * ts_ratio * len(dataset))
            end = (int)((nfold + 1) * ts_ratio * len(dataset))

            test_index = indlist[start:end]
            remain_index = np.delete(indlist, np.arange(start, end))

            np.random.seed(seed)
            np.random.shuffle(remain_index)

            num_train = (int)(len(dataset) * tr_ratio)
            train_index = remain_index[:num_train]
            valid_index = remain_index[num_train:]

            all_index = np.sort(np.concatenate([train_index, valid_index, test_index]))

        else:
            indlist = np.arange(len(dataset))
            start = int(nfold * ts_ratio * len(dataset))
            end = int((nfold + 1) * ts_ratio * len(dataset))
            test_index = indlist[start:end]
            remain_index = np.delete(indlist, np.arange(start, end))
            num_train = int(len(dataset) * tr_ratio)
            train_index = remain_index[:num_train]
            valid_index = remain_index[num_train:]
            all_index = np.sort(np.concatenate([train_index, valid_index, test_index]))

        if shuffle:
            i_shuffle = True
        else:
            i_shuffle = False

        dataset = ImputeGAP_Dataset(ts_m=ts_m, use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, eval_length=seq_len, features=features, set="train", normalize=normalize)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=i_shuffle, num_workers=num_workers)

        valid_dataset = ImputeGAP_Dataset(ts_m=ts_m, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, eval_length=seq_len, features=features, set="val", normalize=normalize)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        test_dataset = ImputeGAP_Dataset(ts_m=ts_m, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, eval_length=seq_len, features=features, set="test", normalize=normalize)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        imputegap_dataset = ImputeGAP_Dataset(ts_m=ts_m, use_index_list=all_index, missing_ratio=missing_ratio, seed=seed, eval_length=seq_len, features=features, set="recon", normalize=normalize)
        imputegap_loader = DataLoader(imputegap_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else: # TEST =======================================================================================================
        # only to obtain total length of dataset
        dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
        indlist = np.arange(len(dataset))

        np.random.seed(seed)
        np.random.shuffle(indlist)

        # 5-fold test
        start = (int)(nfold * 0.2 * len(dataset))
        end = (int)((nfold + 1) * 0.2 * len(dataset))

        test_index = indlist[start:end]
        remain_index = np.delete(indlist, np.arange(start, end))

        np.random.seed(seed)
        np.random.shuffle(remain_index)
        num_train = (int)(len(dataset) * 0.7)
        train_index = remain_index[:num_train]
        valid_index = remain_index[num_train:]

        dataset = Physio_Dataset(use_index_list=train_index, missing_ratio=missing_ratio, seed=seed)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
        valid_dataset = Physio_Dataset(use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
        test_dataset = Physio_Dataset(use_index_list=test_index, missing_ratio=missing_ratio, seed=seed)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
        imputegap_loader = None

    return train_loader, valid_loader, test_loader, imputegap_loader, None, None
