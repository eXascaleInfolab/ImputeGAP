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
import torch
import torch.optim as optim
import numpy as np
import imputegap
import imputegap.wrapper.AlgoPython.BRITS.utils as utils
import imputegap.wrapper.AlgoPython.BRITS.models as models
import argparse
import imputegap.wrapper.AlgoPython.BRITS.data_loader as data_loader
import imputegap.tools.utils as utils_imp
import imputegap.wrapper.AlgoPython.BRITS.models.rits_i, imputegap.wrapper.AlgoPython.BRITS.models.brits_i, imputegap.wrapper.AlgoPython.BRITS.models.rits, imputegap.wrapper.AlgoPython.BRITS.models.brits, imputegap.wrapper.AlgoPython.BRITS.models.m_rnn
from imputegap.wrapper.AlgoPython.BRITS.input_process_imputegap import data_loader_brits

def handle_parser(argv=None):

    import argparse

    parser = argparse.ArgumentParser(description='BRITS')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str)
    parser.add_argument('--hid_size', type=int)
    parser.add_argument('--impute_weight', type=float)
    parser.add_argument('--label_weight', type=float)

    args, _unknown = parser.parse_known_args(argv)

    return args

args = handle_parser()


def train(model, num_workers=4, shuffle=False, replicat=False, verbose=True, deep_verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, replicat=replicat)

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0
        size = len(data_iter) - 1

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            if verbose and idx == size:
                print ('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        imputations, evals = evaluate(model, data_iter, verbose, deep_verbose)

    return imputations, evals


def evaluate(model, val_iter, verbose=True, deep_verbose=False):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    evals_imp = []
    imputations_imp = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

        #print(f"loop :  {np.array(imputation).shape = }")

        imputations_imp.append(imputation)
        evals_imp.append(eval_)

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    imputations_imp = np.concatenate(imputations_imp, axis=0)
    evals_imp = np.concatenate(evals_imp, axis=0)

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    if deep_verbose:
        print(f"\t{imputations_imp.shape = }")
        print(f"\t{evals_imp.shape = }")

    if verbose:
        #print('\tAUC {}'.format(metrics.roc_auc_score(labels, preds)))
        print('\tMAE', np.abs(evals - imputations).mean())
        print('\tMRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    outdir = os.path.join(here, "imputegap_assets/models/brits/result")
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, f"{args.model}_data.npy"), save_impute)
    np.save(os.path.join(outdir, f"{args.model}_label.npy"), save_label)

    return imputations_imp, evals_imp


#  python main.py -- brits --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 1.0 --hid_size 108
def recovBRITS(incomp_data, seq_len=24, model_name="rits_i", epochs=1000, batch_size=64, sliding_windows=1, impute_weight=0.3, label_weight=1, hid_size=108, num_workers=4, shuffle=False, tr_ratio=0.9, seed=42, verbose=True, replicat=False):

    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if verbose:
        print(f"\n(IMPUTATION) {model_name.upper()}\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tseq_len: {seq_len}\n\tepochs: {epochs}\n\tbatch_size: {batch_size}\n\tsliding_windows: {sliding_windows}\n\thid_size: {hid_size}\n\timpute_weight: {impute_weight}\n\tnum_workers: {num_workers}\n\ttr_ratio: {tr_ratio}\n")
        print(f"call: brits.impute(params={{'model': {model_name}, 'seq_len': {seq_len}, 'epochs': {epochs}, 'batch_size': {batch_size}, 'sliding_windows': {sliding_windows}, 'hidden_layers': {hid_size}, 'impute_weight': {impute_weight}, 'num_workers': {num_workers}}})\n")

    if not replicat:
        data_loader_brits(incomp_data, seq_len=seq_len, sliding_windows=sliding_windows, dataset_name="json", tr_ratio=tr_ratio, seed=seed, verbose=verbose, deep_verbose=False)
        features = recov.shape[1]
    else:
        seq_len, features, epochs = 48, 35, 1000

    args.model = model_name
    args.impute_weight = impute_weight
    args.label_weight = label_weight
    args.batch_size = batch_size
    args.epochs = epochs
    args.hid_size = hid_size

    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight, seq_len, features)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    imputations, evals = train(model, num_workers, shuffle=shuffle, replicat=replicat, verbose=verbose, deep_verbose=False)

    if verbose:
        print(f"\t{imputations.shape =}")
        print(f"\t{evals.shape =}")

    if sliding_windows == 0:
        recovery = utils_imp.dataset_reverse_dimensionality(imputations, recov.shape[0], verbose)
    else:
        recovery = utils_imp.reconstruction_windowd_based(preds=imputations, nbr_timestamps=incomp_data.shape[0], sliding_windows=sliding_windows, verbose=verbose, deep_verbose=False)

    if verbose:
        print(f"{recovery.shape =}")

    recov[m_mask] = recovery[m_mask]

    return recov

