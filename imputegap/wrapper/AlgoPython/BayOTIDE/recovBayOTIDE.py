import os
import numpy as np
import torch 
import sys
from imputegap.wrapper.AlgoPython.BayOTIDE.dataBayOTIDE import generate_data_bayotide
sys.path.append("../")
import tqdm
import yaml
torch.random.manual_seed(300)
import imputegap.wrapper.AlgoPython.BayOTIDE.utils_BayOTIDE as utils
from imputegap.wrapper.AlgoPython.BayOTIDE.model_BayOTIDE import BayTIDE
import imputegap.tools.utils as utils_imp

import time
import warnings
warnings.filterwarnings("ignore")
args = utils.parse_args_dynamic_streaming()


def recovBayOTIDE(incomp_data, K_trend=None, K_season=None, n_season=None, K_bias=None, time_scale=None, a0=None, b0=None, v=None, num_fold=1, tr_ratio=0.7, reversed=True, verbose=True, deep_verbose=False, replicat=False):

    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if reversed:
        incomp_data = incomp_data.T

    ts_m = np.copy(incomp_data)
    torch.random.manual_seed(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    if replicat:
        config_path = os.path.join(here, "notebook/config_guangzhou.yaml")
    else:
        config_path = os.path.join(here, "config/imputegap.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ts_r, val_r = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=False)
    args.num_fold = num_fold

    if not replicat:
        config["K_trend"] = K_trend
        config["K_season"] = K_season
        config["n_season"] = n_season
        config["K_bias"] = K_bias
        config["time_scale"] = time_scale
        config["a0"] = a0
        config["b0"] = b0
        config["v"] = v
        config["fold"] = num_fold
        config["R"] = ts_r
        args.dataset = "imputegap"

    hyper_dict = utils.make_hyper_dict(config, args)

    if verbose:
        print(f"(IMPUTATION) BayOTIDE\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tK_trend: {config['K_trend']}\n\tK_season: {config['K_season']}\n\tn_season: {config['n_season']}\n\tK_bias: {config['K_bias']}\n\ttime_scale: {config['time_scale']}\n\ta0: {config['a0']}\n\tb0: {config['b0']}\n\tb0: {config['v']}\n\tnum_fold: {config['fold']}\n\ttr_ratio: {tr_ratio}\n\tts_ratio: {config['R']}\n\tval_ratio: {val_r}\n\tdevice: {hyper_dict['device']}\n")
        print(f"\ncall: bayotide.impute(params={{'K_trend': {K_trend}, 'K_season': {K_season}, 'n_season': {n_season}, 'K_bias': {K_bias}, 'time_scale': {time_scale}, 'a0': {a0}, 'b0': {b0}, 'v': {v}, 'num_fold': {num_fold}}})\n")


    INNER_ITER = hyper_dict["INNER_ITER"]
    EVALU_T = hyper_dict["EVALU_T"]

    test_rmse = []
    test_MAE = []
    test_CRPS = []
    test_NLLK = []
    train_time = []

    before_smooth_MAE_list = []
    before_smooth_RMSE_list = []

    result_dict = {}

    if not replicat:
        data_file = generate_data_bayotide(ts_m=ts_m, replace="neg", ts_r=ts_r, val_r=val_r, verbose=verbose, deep_verbose=deep_verbose)

    if verbose:
        print("\ntraining...\n")

    d_tqdm = not verbose
    for fold_id in range(args.num_fold):

        data_dict = utils.make_data_dict(hyper_dict, data_file, fold=fold_id, deep_verbose=deep_verbose)

        train_time_start = time.time()

        model = BayTIDE(hyper_dict, data_dict)

        model.reset()

        # one-pass along the time axis
        for T_id in tqdm.tqdm(range(model.T), disable=d_tqdm):
            model.filter_predict(T_id)
            model.msg_llk_init()

            if model.mask_train[:,T_id].sum()>0: # at least one obseved data at current step
                for inner_it in range(INNER_ITER):

                    flag = (inner_it == (INNER_ITER - 1))

                    model.msg_approx_U(T_id)
                    model.filter_update(T_id,flag)

                    model.msg_approx_W(T_id)
                    model.post_update_W(T_id)

                model.msg_approx_tau(T_id)
                model.post_update_tau(T_id)

            else:
                model.filter_update_fake(T_id)

            if T_id % EVALU_T == 0 or T_id == model.T - 1:
                _, loss_dict = model.model_test(T_id)

                if verbose:
                    print("T_id = {}, train_rmse = {:.3f}, test_rmse= {:.3f}".format(T_id, loss_dict["train_RMSE"], loss_dict["test_RMSE"]))

                # to add: store running loss?

        before_smooth_MAE = loss_dict["test_MAE"]
        before_smooth_RMSE = loss_dict["test_RMSE"]
        model.smooth()
        model.post_update_U_after_smooth(0)
        train_time_end = time.time()

        if verbose:
            print("\nevaluation...\n")

        imputation, loss_dict = model.model_test(T=T_id, prob=True)

        if verbose:
            print("fold = {}, after smooth: \n test_rmse= {:.3f}, \n test_MAE= {:.3f}, \n CRPS= {:.3f},\n neg-llk= {:.3f}".format(fold_id,loss_dict["test_RMSE"], loss_dict["test_MAE"],loss_dict["CRPS"], loss_dict["neg-llk"]))

        test_rmse.append(loss_dict["test_RMSE"])
        test_MAE.append(loss_dict["test_MAE"])
        test_CRPS.append(loss_dict["CRPS"])
        test_NLLK.append(loss_dict["neg-llk"])
        train_time.append(train_time_end - train_time_start)

        if verbose:
            print("fold = {}, train-time = {:.3f} sec\n".format(fold_id, train_time_end - train_time_start))
            print("\nreconstruction...\n")

        before_smooth_MAE_list.append(before_smooth_MAE)
        before_smooth_RMSE_list.append(before_smooth_RMSE)

    test_rmse = np.array(test_rmse)
    test_MAE = np.array(test_MAE)
    test_CRPS = np.array(test_CRPS)
    test_NLLK = np.array(test_NLLK)
    train_time = np.array(train_time)

    before_smooth_MAE = np.array(before_smooth_MAE_list)
    before_smooth_RMSE = np.array(before_smooth_RMSE_list)

    result_dict["RMSE"] = test_rmse
    result_dict["MAE"] = test_MAE
    result_dict["CRPS"] = test_CRPS
    result_dict["NLLK"] = test_NLLK

    result_dict["before_smooth_MAE"] = before_smooth_MAE
    result_dict["before_smooth_RMSE"] = before_smooth_RMSE
    result_dict["time"] = np.sum(train_time)

    imputation = imputation.cpu().detach().numpy()

    if reversed:
        imputation = imputation.T

    if verbose:
        print(f"{imputation.shape =}\n")

    recov[m_mask] = imputation[m_mask]

    return recov
