# ===============================================================================================================
# SOURCE: https://github.com/ermongroup/CSDI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://neurips.cc/virtual/2021/poster/26846
# ===============================================================================================================

import numpy as np
import torch
import json
import yaml
import os
from imputegap.wrapper.AlgoPython.CSDI.main_model import CSDI_Physio
from imputegap.wrapper.AlgoPython.CSDI.main_model import CSDI_Custom
from imputegap.wrapper.AlgoPython.CSDI.utils import train, evaluate
import imputegap.tools.utils as utils_imp

def handle_parser(argv=None):

    import argparse

    parser = argparse.ArgumentParser(description='CSDI')
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
    parser.add_argument("--unconditional", type=int, default=0)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)

    args, _unknown = parser.parse_known_args(argv)

    return args

def recovCSDI(ts_m, seq_len=-1, batch_size=-1, epochs=200, num_workers=0, nsamples=100, sliding_windows=1, target_strategy="mix", shuffle=True, normalize=True, tr_ratio=0.7, seed=1, verbose=True, deep_verbose=False, replicat=False):

    data_m = np.copy(ts_m)
    recov = np.copy(ts_m)
    m_mask = np.isnan(ts_m)

    multivariate, seq_len, batch_size, sliding_windows, error = utils_imp.prepare_deep_learning_params(incomp_data=ts_m, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)

    if error:
        return ts_m

    args = handle_parser()
    args.seq_len = seq_len
    args.features = ts_m.shape[1]
    args.num_workers = num_workers
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.shuffle = shuffle
    args.normalize = normalize
    args.nsample = nsamples
    args.target_strategy = target_strategy
    if args.unconditional == 0:
        args.unconditional = False
    else:
        args.unconditional = True

    if replicat: # physio
        args.testmissingratio = 0.1
        args.features = 35
        args.seq_len = 48
        batch_size = 16
        args.shuffle = True
        args.normalize = True
        args.seed = 1
        multivariate = True
        sliding_windows = args.seq_len

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config/" + args.config)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if deep_verbose:
        print(f"{path = }")

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["test_missing_ratio"] = args.testmissingratio
    config["model"]["target_strategy"] = args.target_strategy
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs


    if verbose:
        print(f"(IMPUTATION) CSDI\n\tMatrix: {ts_m.shape[0]}, {ts_m.shape[1]}\n\tseq_len: {args.seq_len}\n\tbatch_size: {config["train"]["batch_size"]}\n\tepochs: {config["train"]["epochs"]}\n\tsliding_windows: {sliding_windows}\n\ttarget_strategy: {args.target_strategy}\n\tnsample: {args.nsample}\n\tfeatures: {args.features}\n\tnum_workers: {args.num_workers}\n\ttr_ratio: {tr_ratio}\n\tshuffle: {args.shuffle}\n\tinner normalizer: {args.normalize}\n\tseed: {seed}\n\tverbose: {verbose}\n\tdevice: {args.device}\n")

    if deep_verbose:
        print(args)
        print(json.dumps(config, indent=4))

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    foldername = os.path.join(here, "imputegap_assets/models/csdi/")

    if deep_verbose:
        print(f"{foldername = }")
    os.makedirs(foldername, exist_ok=True)

    if multivariate:
        from imputegap.wrapper.AlgoPython.CSDI.dataset_physio import get_dataloader
        train_loader, valid_loader, test_loader, imputegap_loader, _, _ = get_dataloader(seed=args.seed, nfold=0, batch_size=config["train"]["batch_size"], missing_ratio=config["model"]["test_missing_ratio"], replicat=replicat, ts_m=data_m, seq_len=args.seq_len, features=args.features, num_workers=args.num_workers, shuffle=args.shuffle, normalize=args.normalize, tr_ratio=tr_ratio, verbose=verbose)
    else:
        from imputegap.wrapper.AlgoPython.CSDI.dataset_custom import get_dataloader
        train_loader, valid_loader, test_loader, imputegap_loader, _, _ = get_dataloader(seed=args.seed, batch_size=config["train"]["batch_size"], missing_ratio=config["model"]["test_missing_ratio"], sliding_windows=sliding_windows, ts_m=data_m, seq_len=args.seq_len, features=args.features, num_workers=args.num_workers, shuffle=args.shuffle, normalize=args.normalize, tr_ratio=tr_ratio, verbose=verbose)

    if not multivariate:
        model = CSDI_Physio(config, args.device, target_dim=args.features).to(args.device)
    else:
        model = CSDI_Custom(config, args.device, target_dim=args.features).to(args.device)

    train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername, verbose=verbose)

    _ = evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=path, n_timestamps=None, multivariate=multivariate, sliding_windows=sliding_windows, verbose=verbose)

    if not replicat:
        imputed_matrix = evaluate(model, imputegap_loader, nsample=args.nsample, scaler=1, foldername=path, n_timestamps=ts_m.shape, multivariate=multivariate, sliding_windows=sliding_windows, verbose=verbose)
    else:
        return recov

    recov[m_mask] = imputed_matrix[m_mask]

    if verbose:
        print(f"{imputed_matrix.shape = }")

    return recov