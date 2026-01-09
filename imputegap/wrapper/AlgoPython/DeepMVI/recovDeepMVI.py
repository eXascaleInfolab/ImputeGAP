# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================

#!/usr/bin/python
import numpy as np

from imputegap.wrapper.AlgoPython.DeepMVI.transformer import transformer_recovery

# end function


def deep_mvi_recovery(input, max_epoch=1000, patience=2, lr=1e-3, batch_size=16, tr_ratio=0.9, seed=0, verbose=True):

    if verbose:
        print(f"(IMPUTATION) DEEP-MVI\n\tMatrix: {input.shape[0]}, {input.shape[1]}\n\tmax_epoch: {max_epoch}\n\tpatience: {patience}\n\tlr: {lr}\n\tbatch_size: {batch_size}\n\tseed: {seed}\n")
        print(f"call: deepmvi.impute(params={{'max_epoch': {max_epoch}, 'patience': {patience}, 'lr': {lr}, 'batch_size': {batch_size}}})\n")

    recov = np.copy(input)
    mask = np.isnan(input)

    matrix_imputed = transformer_recovery(input, max_epoch=max_epoch, patience=patience, lr=lr, batch_size=batch_size, seed=seed, verbose=verbose, deep_verbose=False)

    recov[mask] = matrix_imputed[mask]

    return recov