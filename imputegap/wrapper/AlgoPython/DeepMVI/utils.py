# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================

import numpy as np
import copy
from contextlib import contextmanager

@contextmanager
def null_context():
    yield

def is_blackout(matrix):
    arr = (np.sum(np.isnan(matrix).astype(int),axis=1) == matrix.shape[1])
    return arr.astype(int).sum() > 0


# def get_block_length(matrix):
#     num_missing = len(np.where(np.isnan(matrix))[0])
#     num_blocks = 0
#     for j in range(matrix.shape[1]):
#         temp = matrix[:,j]
#         for i in range(len(temp)-1):
#             if (np.isnan(temp[i]) and ~np.isnan(temp[i+1])):
#                 num_blocks += 1
#         if (np.isnan(temp[-1])):
#             num_blocks += 1
#     #num_blocks *= matrix.shape[1]
#     return int(num_missing/num_blocks)

# def get_block_length(matrix):
#     temp = np.where(np.isnan(matrix))
#     time = temp[0][0]
#     ts = temp[1][0]
#     i = 0
#     while (np.isnan(matrix[time+i,ts])):
#         i += 1
#     return i

# def get_block_length(matrix):
#     tss = np.unique(np.where(np.isnan(matrix))[1])
#     block_size = float('inf')
#     for ts in tss:
#         time = np.where(np.isnan(matrix[:,ts]))[0][0]
#         i = 0
#         while (time+i < matrix.shape[0] and np.isnan(matrix[time+i,ts])):
#             i += 1
#         block_size = min(block_size,i)
#     return int(block_size)

def make_validation (matrix,num_missing=20, deep_verbose=False):
    np.random.seed(0)
    nan_mask = np.isnan(matrix)

    padded_mat = np.concatenate([np.zeros((1,nan_mask.shape[1])), nan_mask, np.zeros((1,nan_mask.shape[1]))], axis=0)
    indicator_mat = (padded_mat[1:,:]-padded_mat[:-1,:]).T
    pos_start = np.where(indicator_mat==1)
    pos_end = np.where(indicator_mat==-1)
    lens = (pos_end[1]-pos_start[1])[:,None]
    start_index = pos_start[1][:,None]
    time_series = pos_start[0][:,None]
    test_points = np.concatenate([start_index,time_series,lens],axis=1)
    temp = np.copy(test_points[:, 2])
    if (temp.shape[0]>1):
        block_size = temp[int(temp.shape[0]/10):-int(temp.shape[0]/10)-1].mean()
    else :
        block_size = temp.mean()

    w = int(10*np.log10(block_size))
    val_block_size = int(min(block_size,w))

    if deep_verbose:
        print(f"NATERQ__LOGS__; {w = }")
        print(f"NATERQ__LOGS__; {block_size = }")
        print(f"NATERQ__LOGS__; {num_missing = }")
        print(f"NATERQ__LOGS__; {val_block_size = }")

    num_missing = int(num_missing/val_block_size)
    train_matrix = copy.deepcopy(matrix)
    val_points = []
    
    for _ in range(num_missing):
        validation_points = np.random.uniform(0, matrix.shape[0]-val_block_size,(matrix.shape[1])).astype(int)
        for i,x in enumerate(validation_points) :
            train_matrix[x:x+val_block_size,i] = np.nan
            val_points.append([x,i,val_block_size])
            
    return train_matrix, matrix,np.array(val_points), test_points, int(block_size), w