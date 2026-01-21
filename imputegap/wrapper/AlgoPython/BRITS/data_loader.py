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
import ujson as json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, replicat=False):
        super(MySet, self).__init__()

        saving_path = "json"
        if replicat:
            print(f"\nloading from the replicat test: physionet\n")
            here = os.path.dirname(os.path.abspath(__file__))
            dataset_saving = os.path.join(here, "json")
        else:
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            dataset_saving = os.path.join(here, "imputegap_assets/models/brits")
        dataset_saving_dir = os.path.join(here, dataset_saving, saving_path)

        self.content = open(dataset_saving_dir).readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    # materialize lists (not iterators)
    forward = [r['forward'] for r in recs]
    backward = [r['backward'] for r in recs]

    def to_tensor_dict(records):
        values     = torch.tensor([r['values']     for r in records], dtype=torch.float32)
        masks      = torch.tensor([r['masks']      for r in records], dtype=torch.float32)
        evals      = torch.tensor([r['evals']      for r in records], dtype=torch.float32)
        eval_masks = torch.tensor([r['eval_masks'] for r in records], dtype=torch.float32)
        deltas     = torch.tensor([r['deltas']     for r in records], dtype=torch.float32)
        forwards   = torch.tensor([r['forwards']   for r in records], dtype=torch.float32)
        return {
            'values': values,
            'masks': masks,
            'evals': evals,
            'eval_masks': eval_masks,
            'deltas': deltas,
            'forwards': forwards,
        }

        #print(f"{values.shape = }")
        #print(f"{masks.shape = }")
        #print(f"{evals.shape = }")
        #print(f"{eval_masks.shape = }")
        #print(f"{deltas.shape = }")
        #print(f"{forwards.shape = }")

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward),
        'labels': torch.tensor([r['label'] for r in recs], dtype=torch.float32),
        'is_train': torch.tensor([r['is_train'] for r in recs], dtype=torch.float32),
    }

    return ret_dict


def get_loader(batch_size = 64, shuffle = True, num_workers=4, replicat=False):
    data_set = MySet(replicat)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = num_workers, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
