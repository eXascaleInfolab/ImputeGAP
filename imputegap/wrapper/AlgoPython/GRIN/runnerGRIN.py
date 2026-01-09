# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
import imputegap.tools.utils as utils_imp

from imputegap.wrapper.AlgoPython.GRIN.lib import fillers, datasets, config
from imputegap.wrapper.AlgoPython.GRIN.lib.data.datamodule import SpatioTemporalDataModule
from imputegap.wrapper.AlgoPython.GRIN.lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from imputegap.wrapper.AlgoPython.GRIN.lib.nn import models
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metric_base import MaskedMetric
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from imputegap.wrapper.AlgoPython.GRIN.lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from imputegap.wrapper.AlgoPython.GRIN.lib.utils.parser_utils import str_to_bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.MPGRUNet, models.BiMPGRUNet]


def silence_console_noise():
    import os, warnings, logging

    # --- Python warnings (pandas, torch, your code) ---
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # if you want to be extra aggressive:
    # warnings.filterwarnings("ignore")

    # --- Lightning logs like "GPU available..." and other INFO ---
    for name in [
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # --- Optional: hide TensorFlow/XLA style noise if it appears ---
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # --- Optional: Lightning can also honor this for some logs ---
    os.environ.setdefault("PL_DISABLE_FORK", "1")

def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    elif model_str == 'mpgru':
        model, filler = models.MPGRUNet, fillers.GraphFiller
    elif model_str == 'bimpgru':
        model, filler = models.BiMPGRUNet, fillers.GraphFiller
    elif model_str == 'var':
        model, filler = models.VARImputer, fillers.Filler
    elif model_str == 'gain':
        model, filler = models.RGAINNet, fillers.RGAINFiller
    elif model_str == 'birnn':
        model, filler = models.BiRNNImputer, fillers.MultiImputationFiller
    elif model_str == 'rnn':
        model, filler = models.RNNImputer, fillers.Filler
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(ts_m, dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay_block':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la_block':
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif dataset_name == 'imputegap':
        dataset = datasets.MissingValuesImputeGAP(ts_m, p_fault=0., p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset

def handle_parser(argv=None):

    import argparse

    parser = argparse.ArgumentParser(description='GRIN')

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='imputegap')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    known_args, _ = parser.parse_known_args(argv)
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args, _unknown = parser.parse_known_args(argv)

    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def runGRIN(incomp_data, seq_len=1, sim_type="corr", epochs=50, batch_size=32, sliding_windows=1, lr=0.001, alpha=10, patience=40, num_workers=0, seed=42, tr_ratio=0.7, verbose=True, deep_verbose=False):

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if not verbose:
        silence_console_noise()

    if seq_len ==-1 or batch_size==-1:
        seq_len, batch_size = utils_imp.auto_seq_llms(data_x=ts_m, goal="seq", subset=True, high_limit=50, exception=True, verbose=verbose)

    if sliding_windows == 0:
        multivariate = True
        seq_len=1
        ts_m = utils_imp.dataset_add_dimensionality(ts_m, seq_length=seq_len, three_dim=False, verbose=False)
        strat = "samples"
    else:
        multivariate = False
        strat = "windows"

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    foldername = os.path.join(here, "imputegap_assets/logs/grin/")
    os.makedirs(foldername, exist_ok=True)
    config['logs'] = foldername

    args = handle_parser()
    args = copy.deepcopy(args)

    args.lr = lr
    args.epochs = epochs
    args.patience = patience
    args.batch_size = batch_size
    args.window = seq_len
    args.stride = sliding_windows
    args.alpha = alpha
    args.workers = num_workers
    args.test_len, args.val_len = utils_imp.sets_splitter_based_on_training(tr=tr_ratio, verbose=verbose)

    if verbose:
        print(f"\n(IMPUTATION) GRIN\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tseq_len: {args.window}\n\tsim_type: {sim_type}\n\tepochs: {args.epochs}\n\tbatch_size: {args.batch_size}\n\tsliding_windows: {sliding_windows}\n\talpha: {args.alpha}\n\tpatience: {args.patience}\n\tnum_workers: {args.workers}\n\ttr_ratio: {tr_ratio}\n\tts_ratio: {args.test_len}\n\tval_ratio: {args.val_len}\n")
        print(f"call: gain.impute(params={{'seq_len': {seq_len}, 'sim_type': '{sim_type}', 'epochs': {epochs}, 'batch_size': {batch_size}, 'sliding_windows': {sliding_windows}, 'alpha': {alpha}, 'patience': {patience}, 'num_workers': {num_workers}}})\n")

    if args.seed < 0:
        args.seed = seed
    torch.set_num_threads(1)
    pl.seed_everything(args.seed, verbose=False)

    if verbose:
        print("\n\npre-processing")
    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(ts_m=ts_m, dataset_name=args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride,
                                multivariate=multivariate
                                )

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitters, return_dict=True)

    train_idxs, val_idxs, test_idxs = dataset.splitters(torch_dataset, **split_conf)
    imputed_idxs = dataset.splitter_imputegap(torch_dataset, **split_conf)

    if verbose:
        print(f"\n\tdataset shape :{dataset.df.shape}")
        print(f"\t\t(TRAIN) {len(train_idxs)} {strat} of size {seq_len}")
        print(f"\t\t(VAL) {len(val_idxs)} {strat} of size {seq_len}")
        print(f"\t\t(TEST) {len(test_idxs)} {strat} of size {seq_len}")
        print(f"\t(RECONSTRUCTION) {len(imputed_idxs)} {strat} of size {seq_len}")

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs, imputed_idxs=imputed_idxs, **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity_imputegap(type=sim_type, thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    ########################################
    # predictor                            #
    ########################################
    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams}, target_cls=model_cls, return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn), metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False), 'mape': MaskedMAPE(compute_on_step=False), 'mse': MaskedMSE(compute_on_step=False), 'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr, 'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={'eta_min': 0.0001, 'T_max': args.epochs},
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)

    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams}, target_cls=filler_cls, return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    if verbose:
        print("\ntraining...")

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         #gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         check_val_every_n_epoch=1,
                         log_every_n_steps=1,
                         enable_progress_bar=verbose,
                         enable_model_summary=verbose)

    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    if verbose:
        print("\ntesting...")

    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=device, weights_only=False)  # weights_only=False by default
    filler.load_state_dict(checkpoint["state_dict"])

    filler.freeze()
    trainer.test(model=filler, datamodule=dm, ckpt_path=None, verbose=verbose)
    filler.eval()

    if torch.cuda.is_available():
        filler = filler.to(device)

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)

    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    for aggr_by, df_hat in df_hats.items():
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            if verbose:
                print(f' {metric_name}: {error:.4f}')

    if deep_verbose:
        print(f"\n\t{y_true.shape = }")
        print(f"\t{y_hat.shape = }")
        print(f"\t{mask.shape = }\n")

    ########################################
    # reconstruct                          #
    ########################################

    if verbose:
        print("\nreconstruct...")

    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=device, weights_only=False)  # weights_only=False by default
    filler.load_state_dict(checkpoint["state_dict"])

    filler.freeze()
    trainer.test(model=filler, datamodule=dm, ckpt_path=None, verbose=verbose)
    filler.eval()

    if torch.cuda.is_available():
        filler = filler.to(device)

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.imputed_dataloader(), return_mask=True)

    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    if verbose:
        print(f"\n\timputed output : {y_hat.shape = }")

    if multivariate:
        y_hat = utils_imp.dataset_reverse_dimensionality(y_hat, expected_n=incomp_data.shape[0])
    else:
        y_hat = utils_imp.reconstruction_windowd_based(preds=y_hat, nbr_timestamps=incomp_data.shape[0], verbose=verbose, deep_verbose=False)

    recov[m_mask] = y_hat[m_mask]

    return recov

