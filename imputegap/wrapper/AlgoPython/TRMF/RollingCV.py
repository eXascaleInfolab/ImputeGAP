# ===============================================================================================================
# SOURCE: https://github.com/SemenovAlex/trmf/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
# ===============================================================================================================

"""
Rolling cross-validation for timeseries.
"""

import numpy as np
from Metrics import ND, NRMSE

def get_slice(data, T_train, T_test, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start+T_train].copy()
    test = data[:, T_start+T_train:T_start+T_train+T_test].copy()

    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())
        
        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
        test -= mean_train.repeat(T_test).reshape(N, T_test)
        test /= std_train.repeat(T_test).reshape(N, T_test)
    
    return train, test

def RollingCV(model, data, T_train, T_test, T_step, metric='ND', normalize=True):
    scores = np.array([])
    for T_start in range(0, data.shape[1]-T_train-T_test+1, T_step):
        train, test = get_slice(data, T_train, T_test, T_start, normalize=normalize)
        model.fit(train)
        test_preds = model.predict(T_test)
        if metric == 'ND':
            scores = np.append(scores, ND(test_preds, test))
        if metric == 'NRMSE':
            scores = np.append(scores, NRMSE(test_preds, test))
    return scores

