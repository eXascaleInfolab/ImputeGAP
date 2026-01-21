# ===============================================================================================================
# SOURCE: https://github.com/WenjieDu/SAITS
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://doi.org/10.1016/j.eswa.2023.119619
# ===============================================================================================================

"""
The script for running (including training and testing) all models in this repo.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023.

"""


import argparse
import math
import os
import warnings
from configparser import ConfigParser, ExtendedInterpolation
import imputegap.tools.utils as utils_imp
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from imputegap.wrapper.AlgoPython.SAITS.dataset_generating_scripts.gene_imputegap_dataset import sait_loader_imputegap
from imputegap.wrapper.AlgoPython.SAITS.dataset_generating_scripts.gene_imputegap_w_dataset import sait_loader_w_imputegap

try:
    import nni
except ImportError:
    pass

from imputegap.wrapper.AlgoPython.SAITS.modeling.saits import SAITS
from imputegap.wrapper.AlgoPython.SAITS.modeling.unified_dataloader import UnifiedDataLoader
from imputegap.wrapper.AlgoPython.SAITS.modeling.utils import (
    Controller,
    setup_logger,
    save_model,
    load_model,
    check_saving_dir_for_model,
    masked_mae_cal,
    masked_rmse_cal,
    masked_mre_cal,
)

warnings.filterwarnings("ignore")  # if to ignore warnings

MODEL_DICT = { "SAITS": SAITS, }
OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}

def read_arguments(arg_parser, cfg_parser):
    arg_parser.eval_every_n_steps = cfg_parser.getint("dataset", "eval_every_n_steps")
    # training settings
    arg_parser.MIT = cfg_parser.getboolean("training", "MIT")
    arg_parser.ORT = cfg_parser.getboolean("training", "ORT")
    arg_parser.lr = cfg_parser.getfloat("training", "lr")
    arg_parser.optimizer_type = cfg_parser.get("training", "optimizer_type")
    arg_parser.weight_decay = cfg_parser.getfloat("training", "weight_decay")
    arg_parser.early_stop_patience = cfg_parser.getint("training", "early_stop_patience")
    arg_parser.model_saving_strategy = cfg_parser.get("training", "model_saving_strategy")
    arg_parser.max_norm = cfg_parser.getfloat("training", "max_norm")
    arg_parser.imputation_loss_weight = cfg_parser.getfloat("training", "imputation_loss_weight")
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat("training", "reconstruction_loss_weight")
    arg_parser.model_type = cfg_parser.get("model", "model_type")
    return arg_parser

def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """write summary into tensorboard file"""
    summary_writer.add_scalar(f"total_loss/{stage}", info_dict["total_loss"], step)
    summary_writer.add_scalar(
        f"imputation_loss/{stage}", info_dict["imputation_loss"], step
    )
    summary_writer.add_scalar(
        f"imputation_MAE/{stage}", info_dict["imputation_MAE"], step
    )
    summary_writer.add_scalar(
        f"reconstruction_loss/{stage}", info_dict["reconstruction_loss"], step
    )
    summary_writer.add_scalar(
        f"reconstruction_MAE/{stage}", info_dict["reconstruction_MAE"], step
    )


def result_processing(results, args=None):
    """process results and losses for each training step"""
    results["total_loss"] = torch.tensor(0.0, device=args.device)
    if args.model_type == "BRITS":
        results["total_loss"] = (
            results["consistency_loss"] * args.consistency_loss_weight
        )
    results["reconstruction_loss"] = (
        results["reconstruction_loss"] * args.reconstruction_loss_weight
    )
    results["imputation_loss"] = (
        results["imputation_loss"] * args.imputation_loss_weight
    )
    if args.MIT:
        results["total_loss"] += results["imputation_loss"]
    if args.ORT:
        results["total_loss"] += results["reconstruction_loss"]
    return results


def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger, model=None, args=None, verbose=True):

    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage="train")
    # apply gradient clipping if args.max_norm != 0
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    results["total_loss"].backward()
    optimizer.step()

    summary_write_into_tb(summary_writer, results, state_dict["train_step"], "train")
    if state_dict["train_step"] % args.eval_every_n_steps == 0:
        state_dict_from_val = validate(model, val_dataloader, summary_writer, training_controller, logger, optimizer=optimizer, args=args, verbose=verbose)
        if state_dict_from_val["should_stop"]:
            if verbose:
                logger.info(f"Early stopping worked, stop now...")
            return True
    return False


def model_processing(data, model, stage, optimizer=None, val_dataloader=None, summary_writer=None, training_controller=None, logger=None, args=None, verbose=True):
    if stage == "train":
        optimizer.zero_grad()
        if not args.MIT:
            indices, X, missing_mask = map(lambda x: x.to(args.device), data)
            inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}
            results = result_processing(model(inputs, stage), args=args)
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger, model, args=args, verbose=verbose)
        else:
            indices, X, missing_mask, X_holdout, indicating_mask = map(
                lambda x: x.to(args.device), data
            )
            inputs = {
                "indices": indices,
                "X": X,
                "missing_mask": missing_mask,
                "X_holdout": X_holdout,
                "indicating_mask": indicating_mask,
            }
            results = result_processing(model(inputs, stage), args=args)
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger, model, args=args, verbose=verbose)
        return early_stopping
    else:  # in val/test stage
        indices, X, missing_mask, X_holdout, indicating_mask = map(
                lambda x: x.to(args.device), data
        )
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
            "X_holdout": X_holdout,
            "indicating_mask": indicating_mask,
        }
        results = model(inputs, stage)
        results = result_processing(results, args)
        return inputs, results


def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller, logger, args=None, verbose=True):
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        for idx, data in enumerate(train_dataloader):
            model.train()
            early_stopping = model_processing(data, model, "train", optimizer, test_dataloader, summary_writer, training_controller, logger, args=args, verbose=verbose)
            if early_stopping:
                break
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    if verbose:
        logger.info("Finished all epochs. Stop training now.")


def validate(model, val_iter, summary_writer, training_controller, logger, optimizer=None, args=None, verbose=True):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    (
        total_loss_collector,
        imputation_loss_collector,
        reconstruction_loss_collector,
        reconstruction_MAE_collector,
    ) = ([], [], [], [])

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, "val", args=args)
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])

            total_loss_collector.append(results["total_loss"].data.cpu().numpy())
            reconstruction_MAE_collector.append(
                results["reconstruction_MAE"].data.cpu().numpy()
            )
            reconstruction_loss_collector.append(
                results["reconstruction_loss"].data.cpu().numpy()
            )
            imputation_loss_collector.append(
                results["imputation_loss"].data.cpu().numpy()
            )

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
    info_dict = {
        "total_loss": np.asarray(total_loss_collector).mean(),
        "reconstruction_loss": np.asarray(reconstruction_loss_collector).mean(),
        "imputation_loss": np.asarray(imputation_loss_collector).mean(),
        "reconstruction_MAE": np.asarray(reconstruction_MAE_collector).mean(),
        "imputation_MAE": imputation_MAE.cpu().numpy().mean(),
    }
    state_dict = training_controller("val", info_dict, logger, verbose=verbose)
    summary_write_into_tb(summary_writer, info_dict, state_dict["val_step"], "val")
    if args.param_searching_mode:
        nni.report_intermediate_result(info_dict["imputation_MAE"])
        if args.final_epoch or state_dict["should_stop"]:
            nni.report_final_result(state_dict["best_imputation_MAE"])

    if (state_dict["save_model"] and args.model_saving_strategy) or args.model_saving_strategy == "all":
        saving_path = os.path.join(
            args.model_saving,
            "model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}".format(
                state_dict["train_step"],
                state_dict["val_step"],
                info_dict["imputation_MAE"],
            ),
        )
        # save_model(model, optimizer, state_dict, args, saving_path)
        save_model(model, optimizer, state_dict, args, os.path.join(args.model_saving, "saits_model"))
    return state_dict


def check_train_model(model, test_dataloader, args=None, logger=None, verbose=True):
    if verbose:
        logger.info(f"Start evaluating on whole test set...")
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, "test", args=args)
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs["X_holdout"])
            evalMask_collector.append(inputs["indicating_mask"])
            imputations_collector.append(results["imputed_data"])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
        imputation_RMSE = masked_rmse_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )
        imputation_MRE = masked_mre_cal(
            imputations_collector, evalX_collector, evalMask_collector
        )

    assessment_metrics = {
        "imputation_MAE on the test set": imputation_MAE,
        "imputation_RMSE on the test set": imputation_RMSE,
        "imputation_MRE on the test set": imputation_MRE,
        "trainable parameter num": args.total_params,
    }
    with open(os.path.join(args.result_saving_path, "overall_performance_metrics.out"), "w") as f:
        if verbose:
            logger.info("...")
            logger.info("Overall performance metrics are listed as follows:")
        for k, v in assessment_metrics.items():
            if verbose:
                logger.info(f"{k}: {v}")
            f.write(k + ":" + str(v))
            f.write("\n")


def impute_all_missing_data(model, train_data, val_data, test_data, args=None, logger=None, origins=None, multivariate=False, verbose=False):
    if verbose:
        logger.info(f"Start imputing all missing data in all train/val/test sets...")

    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ["train", "val", "test"]):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}
                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples

            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, "imputations.h5")
    with h5py.File(imputation_saving_path, "w") as hf:
        hf.create_dataset("imputed_train_set", data=imputed_data_dict["train"])
        hf.create_dataset("imputed_val_set", data=imputed_data_dict["val"])
        hf.create_dataset("imputed_test_set", data=imputed_data_dict["test"])
    if verbose:
        logger.info(f"Done saving all imputed data into {imputation_saving_path}.")

    if multivariate:
        imputed_matrix = np.vstack([imputed_data_dict["train"], imputed_data_dict["val"], imputed_data_dict["test"]])
    else:
        train = utils_imp.reconstruction_window_based(preds=imputed_data_dict["train"], nbr_timestamps=origins[0], sliding_windows=origins[3], verbose=verbose, deep_verbose=False)
        val = utils_imp.reconstruction_window_based(preds=imputed_data_dict["val"], nbr_timestamps=origins[1], sliding_windows=origins[3], verbose=verbose, deep_verbose=False)
        test = utils_imp.reconstruction_window_based(preds=imputed_data_dict["test"], nbr_timestamps=origins[2], sliding_windows=origins[3], verbose=verbose, deep_verbose=False)
        imputed_matrix = np.vstack([train, val, test])

    return np.array(imputed_matrix)

def recov_saits(incomp_data, seq_len=-1, batch_size=-1, epochs=10, sliding_windows=1, n_head=8, num_workers=0, shuffle=True, seed=26, tr_ratio=0.7, verbose=False, replicat=False):

    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    multivariate, seq_len, batch_size, sliding_windows, error = utils_imp.prepare_deep_learning_params(incomp_data=incomp_data, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)

    if error:
        return incomp_data

    np.random.seed(seed)
    torch.manual_seed(seed)
    feature_num = incomp_data.shape[1]

    here = os.path.dirname(os.path.abspath(__file__))
    dshere = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    if replicat:
        seq_len = 48
        feature_num = 38
        batch_size = 128
        tr_ratio = 0.8
        seed = 26
        conf = "PhysioNet2012_SAITS_best.ini"
        multivariate = True
        epochs = 10000
        n_head = 4
    else:
        conf = "imputegap_SAITS_base.ini"

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args.config_path = os.path.join(here, "configs", conf)

    # load settings from config file
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)
    args.test_mode = False
    args.param_searching_mode = False
    assert os.path.exists(args.config_path), f'Given config file "{args.config_path}" does not exists'

    args.epochs = epochs
    args.seq_len = seq_len
    args.num_workers = num_workers
    args.result_saving_base_dir = "imputegap_assets"
    args.model_name = ""
    args.feature_num = feature_num
    args.batch_size = batch_size
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_head = n_head
    dshere = os.path.join(dshere, args.result_saving_base_dir)

    if verbose:
        print(f"(IMPUTATION) SAITS\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tseq_len: {args.seq_len}\n\tepoch: {args.epochs}\n\tbatch_size: {args.batch_size}\n\tsliding_windows: {sliding_windows}\n\tmultivariate: {multivariate}\n\tn_head: {args.n_head}\n\tnum_workers: {args.num_workers}\n\tseed: {seed}\n\tconfig_path: {args.config_path}\n\ttr_ratio: {tr_ratio}\n\tverbose: {verbose}\n\tdevice: {args.device}\n")
        print(f"\ncall: saits.impute(params={{'seq_len': {args.seq_len}, 'batch_size': {args.batch_size}, 'epochs': {args.epochs}, 'sliding_windows': {sliding_windows}, 'n_head': {args.n_head}, 'num_workers': {args.num_workers}}})\n\n")

    if multivariate:
        dataset_saving_dir, origins = sait_loader_imputegap(incomp_data, seq_len, dshere, tr_ratio=tr_ratio, shuffle=shuffle, verbose=verbose, replicat=replicat)
    else:
        dataset_saving_dir, origins = sait_loader_w_imputegap(incomp_data, seq_len, features=incomp_data.shape[1], here=dshere, sliding_windows=sliding_windows, tr_ratio=tr_ratio, shuffle=shuffle, verbose=verbose)

    args.dataset_path = dataset_saving_dir

    if args.model_type in ["SAITS"]:  # if SA-based model
        args.input_with_mask = cfg.getboolean("model", "input_with_mask")
        args.n_groups = cfg.getint("model", "n_groups")
        args.n_group_inner_layers = cfg.getint("model", "n_group_inner_layers")
        args.param_sharing_strategy = cfg.get("model", "param_sharing_strategy")
        assert args.param_sharing_strategy in ["inner_group","between_group", ], 'only "inner_group"/"between_group" sharing'
        args.d_model = cfg.getint("model", "d_model")
        args.d_inner = cfg.getint("model", "d_inner")
        args.d_k = cfg.getint("model", "d_k")
        args.d_v = cfg.getint("model", "d_v")
        args.dropout = cfg.getfloat("model", "dropout")
        args.diagonal_attention_mask = cfg.getboolean("model", "diagonal_attention_mask")

        dict_args = vars(args)
        if args.param_searching_mode:
            tuner_params = nni.get_next_parameter()
            dict_args.update(tuner_params)
            experiment_id = nni.get_experiment_id()
            trial_id = nni.get_trial_id()
            args.model_name = f"{args.model_name}/{experiment_id}/{trial_id}"
            dict_args["d_k"] = dict_args["d_model"] // dict_args["n_head"]
        model_args = {
            "device": args.device,
            "MIT": args.MIT,
            "n_groups": dict_args["n_groups"],
            "n_group_inner_layers": args.n_group_inner_layers,
            "d_time": args.seq_len,
            "d_feature": args.feature_num,
            "dropout": dict_args["dropout"],
            "d_model": dict_args["d_model"],
            "d_inner": dict_args["d_inner"],
            "n_head": dict_args["n_head"],
            "d_k": dict_args["d_k"],
            "d_v": dict_args["d_v"],
            "input_with_mask": args.input_with_mask,
            "diagonal_attention_mask": args.diagonal_attention_mask,
            "param_sharing_strategy": args.param_sharing_strategy,
        }
    else:
        assert (ValueError), f"Given model_type {args.model_type} is not in {MODEL_DICT.keys()}"

    assert args.model_saving_strategy.lower() in ["all", "best", "none", ], "model saving strategy must be all/best/none"
    if args.model_saving_strategy.lower() == "none":
        args.model_saving_strategy = False
    assert (args.optimizer_type in OPTIMIZER.keys()), f"optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}"
    assert args.device in ["cpu", "cuda"], "device should be cpu or cuda"

    time_now = "imp_saits_imputegap"  # datetime.now().__format__("%Y-%m-%d_T%H_%M_%S")
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now, dshere)


    logger = setup_logger(args.log_saving + "_" + time_now, "w")
    if verbose:
        logger.info(f"args: {args}")
        logger.info(f"Config file path: {args.config_path}")
        logger.info(f"Model name: {args.model_name}")

    unified_dataloader = UnifiedDataLoader(
        args.dataset_path,
        args.seq_len,
        args.feature_num,
        args.model_type,
        args.batch_size,
        args.num_workers,
        args.MIT,
    )

    model = MODEL_DICT[args.model_type](**model_args)
    args.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        logger.info(f"Num of total trainable params is: {args.total_params}")

    # if utilize GPU and GPU available, then move
    if "cuda" in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if verbose:
        logger.info(f"Creating {args.optimizer_type} optimizer...")

    optimizer = OPTIMIZER[args.optimizer_type](model.parameters(), lr=dict_args["lr"], weight_decay=args.weight_decay)
    if verbose:
        logger.info("Entering training mode...")

    train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
    training_controller = Controller(early_stop_patience=args.early_stop_patience)

    train_set_size = unified_dataloader.train_set_size
    if verbose:
        logger.info(f"train set len is {train_set_size}, batch size is {args.batch_size}," f"so each epoch has {math.ceil(train_set_size / args.batch_size)} steps")
        logger.info(f"with seq_len :{unified_dataloader.seq_len = }, and feature num: {unified_dataloader.feature_num}")

    tb_summary_writer = SummaryWriter(os.path.join(args.log_saving, "tensorboard"))
    train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger, args=args, verbose=verbose)

    # TEST ============================================================================================================
    if verbose:
        logger.info("Entering testing mode...")

    args.step_313 = "saits_model"
    args.model_path = os.path.join(args.model_saving, args.step_313)

    args.result_saving_path = os.path.join(dshere, "models/saits")

    os.makedirs(args.result_saving_path) if not os.path.exists(args.result_saving_path) else None
    model = load_model(model, args.model_path, logger, verbose=verbose)
    test_dataloader = unified_dataloader.get_test_dataloader()
    check_train_model(model=model, test_dataloader=test_dataloader, args=args, logger=logger, verbose=verbose)

    (train_data, val_data, test_data,) = unified_dataloader.prepare_all_data_for_imputation()
    imputed_matrix = impute_all_missing_data(model, train_data, val_data, test_data, args, logger, origins=origins, multivariate=multivariate, verbose=verbose)

    if multivariate:
        imputed_matrix = utils_imp.dataset_reverse_dimensionality(imputed_matrix, expected_n=recov.shape[0], verbose=verbose)

    if verbose:
        logger.info("All Done.")
        print(f"{ imputed_matrix.shape = }\n")

    recov[m_mask] = imputed_matrix[m_mask]

    return recov