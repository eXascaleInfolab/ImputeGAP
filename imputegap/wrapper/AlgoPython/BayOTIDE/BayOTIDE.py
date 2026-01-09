import numpy as np
import torch 
import sys
sys.path.append("../")

import tqdm
import yaml
torch.random.manual_seed(300)
import matplotlib.pyplot as plt

import utils_BayOTIDE as utils

from model_BayOTIDE import BayTIDE

from model_LDS import LDS_GP_streaming

import time
import warnings
warnings.filterwarnings("ignore")
args = utils.parse_args_dynamic_streaming()

torch.random.manual_seed(args.seed)


config_path = "./config/r_"+ str(args.r)+"/config_" + args.dataset + "_" +args.task +".yaml"

print("config_path: ", config_path)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_file = config["data_path"]
hyper_dict = utils.make_hyper_dict(config, args)

print('dataset: ', args.dataset,  'task: ', args.task)

INNER_ITER = hyper_dict["INNER_ITER"]
EVALU_T = hyper_dict["EVALU_T"]

test_rmse = []
test_MAE = []
test_CRPS = []
test_NLLK = []
train_time = []

before_smooth_MAE_list = []
before_smooth_RMSE_list = []

running_rmse = []
running_MAE = []
running_N = []
running_T = []

time_list = []

result_dict = {}

start_time = time.time()

for fold_id in range(args.num_fold):

    data_dict = utils.make_data_dict(hyper_dict,data_file,fold=fold_id)

    train_time_start = time.time()

    model = BayTIDE(hyper_dict,data_dict)

    model.reset()

    # one-pass along the time axis 
    for T_id in tqdm.tqdm(range(model.T)):
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
            print("T_id = {}, train_rmse = {:.3f}, test_rmse= {:.3f}".format(T_id, loss_dict["train_RMSE"], loss_dict["test_RMSE"]))

            # to add: store running loss?

    before_smooth_MAE = loss_dict["test_MAE"]
    before_smooth_RMSE = loss_dict["test_RMSE"]

    print('smoothing back...')
    model.smooth()
    model.post_update_U_after_smooth(0)
    print('finish training!')

    train_time_end = time.time()
    print('ELAVUATION...')
    _, loss_dict = model.model_test(T=T_id, prob=True)

    print("fold = {}, after smooth: \n test_rmse= {:.3f}, \n test_MAE= {:.3f}, \n CRPS= {:.3f},\n neg-llk= {:.3f}".format(fold_id,loss_dict["test_RMSE"], loss_dict["test_MAE"],loss_dict["CRPS"], loss_dict["neg-llk"]))

    test_rmse.append(loss_dict["test_RMSE"])
    test_MAE.append(loss_dict["test_MAE"])
    test_CRPS.append(loss_dict["CRPS"])
    test_NLLK.append(loss_dict["neg-llk"])
    train_time.append(train_time_end - train_time_start)

    before_smooth_MAE_list.append(before_smooth_MAE)
    before_smooth_RMSE_list.append(before_smooth_RMSE)

    print("fold = {}, train-time = {:.3f} sec \n\n".format(fold_id, train_time_end - train_time_start))


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
utils.make_log(args, hyper_dict, result_dict,args.other_para)






