# ===============================================================================================================
# SOURCE: https://github.com/SemenovAlex/trmf/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
# ===============================================================================================================

""" Metrics """

import numpy as np

def ND(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return abs((prediction - Y) * mask).sum() / abs(Y).sum()

def NRMSE(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return pow((pow(prediction - Y, 2) * mask).sum(), 0.5) / abs(Y).sum() * pow(mask.sum(), 0.5)


def RMSE(prediction, Y, mask=None):
    prediction = np.array(prediction)
    Y = np.array(Y)

    if mask is None:
        mask = ~np.isnan(Y)

    diff = (prediction - Y)[mask]
    return np.sqrt(np.mean(diff ** 2))

