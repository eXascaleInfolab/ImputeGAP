# ===============================================================================================================
# SOURCE: https://github.com/LMZZML/PriSTI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/10184808
# ===============================================================================================================

import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np
import imputegap.tools.utils as utils_imp

from imputegap.wrapper.AlgoPython.PriSTI.utils import train, evaluate

def handle_parser(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='PriSTI')

    parser.add_argument("--config", type=str, default="imputegap.yaml")
    parser.add_argument('--device', default='cuda', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--targetstrategy", type=str, default="block", choices=["mix", "random", "block"])
    parser.add_argument("--nsample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="block")  # block|point

    args, _unknown = parser.parse_known_args(argv)

    return args

def recovPriSTI(incomp_data, seq_len=-1, batch_size=-1, epochs=200, adj_function="none", nsamples=100, sliding_windows=1, target_strategy="mix", num_workers=0, shuffle=True, normalize=False, tr_ratio=0.7, seed=42, verbose=True, deep_verbose=False, replicat=False):
    args = handle_parser()

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if not replicat:
        multivariate, seq_len, batch_size, sliding_windows, error = utils_imp.prepare_deep_learning_params(incomp_data=incomp_data, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)

        if error:
            return incomp_data

        if multivariate:
            ts_m = utils_imp.dataset_add_dimensionality(ts_m, seq_length=seq_len, three_dim=False, verbose=verbose)

    args.seed = seed
    SEED = args.seed

    foldername = ""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config/" + args.config)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["diffusion"]["shape"] = ts_m.shape[1]
    config["diffusion"]["adj_file"] = adj_function
    config["seed"] = SEED
    config["train"]["epochs"] = epochs
    config["train"]["batch_size"] = batch_size
    config["model"]["target_strategy"] = target_strategy

    if target_strategy == "block":
        args.missing_pattern = "block"
    else:
        args.missing_pattern = "point"

    args.targetstrategy = target_strategy
    args.num_workers = num_workers

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_len, val_len = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=False)

    if verbose:
        print(f"(IMPUTATION) PriSTI\n\tMatrix Shape: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tseq_length: {seq_len}\n\tbatch_size: {batch_size}\n\tepochs: {config['train']['epochs']}\n\tadjacency function: {adj_function}\n\tsamples: {nsamples}\n\tsliding_windows: {sliding_windows}\n\ttarget_strategy: {config['model']['target_strategy']}\n\tunconditional: {config['model']['is_unconditional']}\n\tnum_workers: {args.num_workers}\n\ttr_ratio: {tr_ratio}\n\tts_ratio: {test_len}\n\tval_ratio: {val_len}\n\tseed: {config['seed']}\n")
        print(f"call: pristi.impute(params={{'seq_len': {seq_len}, 'batch_size': {batch_size}, 'epochs': {config['train']['epochs']}, 'sliding_windows': {sliding_windows}, 'target_strategy': '{config['model']['target_strategy']}', 'nsamples': {nsamples}, 'num_workers': {args.num_workers}}})\n")

        print("\npre-processing...\n")

    if replicat:
        from imputegap.wrapper.AlgoPython.PriSTI.dataset_aqi36 import get_dataloader
        from imputegap.wrapper.AlgoPython.PriSTI.main_model import PriSTI_aqi36
        config["diffusion"]["adj_file"] = 'AQI36'

        train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(config["train"]["batch_size"], device=args.device, val_len=val_len, is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers, target_strategy=args.targetstrategy, mask_sensor=config["model"]["mask_sensor"])
        model = PriSTI_aqi36(X=train_loader, config=config, device=args.device).to(args.device)
    else:
        from imputegap.wrapper.AlgoPython.PriSTI.dataset_imputegap import get_dataloader
        from imputegap.wrapper.AlgoPython.PriSTI.main_model import PriSTI_ImputeGAP

        train_loader, valid_loader, test_loader, scaler, mean_scaler, imputegap = get_dataloader(ts_m, seq_length=seq_len, batch_size=batch_size, device=args.device, missing_pattern=args.missing_pattern,
            is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers, target_strategy=args.targetstrategy, test_len=test_len, val_len=val_len, multivariate=multivariate, verbose=verbose, deep_verbose=False)

        model = PriSTI_ImputeGAP(train_loader, config, args.device, target_dim=ts_m.shape[1], seq_len=seq_len).to(args.device)

    if verbose:
        print("\n\n\ntraining...\n")

    if args.modelfolder == "":
        train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername, verbose=verbose)
    else:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_path_model = os.path.join(here, "imputegap_assets/models/pristi")
        model.load_state_dict(torch.load(output_path_model))


    if verbose:
        print("\n\n\nevaluation...\n")

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    if verbose:
        logging.info("model_name={}".format(args.modelfolder))

    _ = evaluate(model, test_loader, nsample=nsamples, scaler=scaler, mean_scaler=mean_scaler, foldername=foldername, imputegap=False, verbose=verbose, deep_verbose=False)

    if verbose:
        print("\n\n\nreconstruction...\n")

    logging.basicConfig(filename=foldername + '/imputegap_model.log', level=logging.DEBUG)
    if verbose:
        logging.info("model_name={}".format(args.modelfolder))
    imputed_imputegap = evaluate(
        model,
        imputegap,
        nsample=nsamples,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        imputegap=True,
        verbose=verbose,
        deep_verbose=False
    )

    if multivariate:
        recovery = utils_imp.dataset_reverse_dimensionality(imputed_imputegap, recov.shape[0], verbose)
    else:
        recovery = utils_imp.reconstruction_window_based(preds=imputed_imputegap, nbr_timestamps=recov.shape[0], sliding_windows=sliding_windows, verbose=verbose, deep_verbose=False)

    if hasattr(recovery, "detach"):
        rec_pristi = recovery.detach().cpu().numpy()
    else:
        rec_pristi = np.asarray(recovery)

    if verbose:
        print(f"{rec_pristi.shape =}")

    recov[m_mask] = rec_pristi[m_mask]

    return np.array(recov)

