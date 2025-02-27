import configparser
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
import numpy as np

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from models.BiaTCGNet.BiaTCGNet import Model
import models
import argparse
import os
import yaml
from data.GenerateDataset import loaddataset
# from tsl.data.utils import WINDOW
import datetime

torch.multiprocessing.set_sharing_strategy('file_system')
node_number=207
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--task', default='prediction',type=str)
parser.add_argument("--adj-threshold", type=float, default=0.1)
parser.add_argument('--dataset',default='Elec')
parser.add_argument('--val_ratio',default=0.2)
parser.add_argument('--test_ratio',default=0.2)
parser.add_argument('--column_wise',default=False)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument("--model-name", type=str, default='spin')
parser.add_argument("--dataset-name", type=str, default='air36'
                                                        '')
parser.add_argument('--fc_dropout', default=0.2, type=float)
parser.add_argument('--head_dropout', default=0, type=float)
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
##############transformer config############################

parser.add_argument('--enc_in', type=int, default=node_number, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=node_number, help='decoder input size')
parser.add_argument('--c_out', type=int, default=node_number, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--num_nodes', type=int, default=node_number, help='dimension of fcn')
parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')
#######################AGCRN##########################
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--rnn_units', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--cheb_k', default=2, type=int)
parser.add_argument('--default_graph', type=bool, default=True)

#############GTS##################################
parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')

parser.add_argument("--config_filename", type=str, default='')
#####################################################
parser.add_argument("--config", type=str, default='imputation/spin.yaml')
parser.add_argument('--output_attention', type=bool, default=False)
# Splitting/aggregation params
parser.add_argument('--val-len', type=float, default=0.2)
parser.add_argument('--test-len', type=float, default=0.2)
parser.add_argument('--mask_ratio',type=float,default=0.1)
# Training params
parser.add_argument('--lr', type=float, default=0.001)  #0.001
# parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=40)
parser.add_argument('--l2-reg', type=float, default=0.)
# parser.add_argument('--batches-epoch', type=int, default=300)
parser.add_argument('--batch-inference', type=int, default=32)
parser.add_argument('--split-batch-in', type=int, default=1)
parser.add_argument('--grad-clip-val', type=float, default=5.)
parser.add_argument('--loss-fn', type=str, default='l1_loss')
parser.add_argument('--lr-scheduler', type=str, default=None)
parser.add_argument('--seq_len',default=24,type=int) # 96
# parser.add_argument('--history_len',default=24,type=int) #96
parser.add_argument('--label_len',default=12,type=int) #48
parser.add_argument('--pred_len',default=24,type=int)
parser.add_argument('--horizon',default=24,type=int)
parser.add_argument('--delay',default=0,type=int)
parser.add_argument('--stride',default=1,type=int)
parser.add_argument('--window_lag',default=1,type=int)
parser.add_argument('--horizon_lag',default=1,type=int)

# Connectivity params
# parser.add_argument("--adj-threshold", type=float, default=0.1)
args = parser.parse_args()
criteron=nn.L1Loss().cuda()

if(args.dataset=='Metr'):
    node_number=207
    args.num_nodes=207
    args.enc_in=207
    args.dec_in=207
    args.c_out=207
elif(args.dataset=='PEMS'):
    node_number=325
    args.num_nodes=325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif(args.dataset=='ETTh1'):
    node_number=7
    args.num_nodes=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif(args.dataset=='Elec'):
    node_number=321
    args.num_nodes=321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif(args.dataset=='BeijingAir'):
    node_number=36
    args.num_nodes=36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36

def train(model, train_data, mask, lr=0.001, epochs=20, seq_len=7, seed=42):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    np.random.randint(seed)
    torch.set_num_threads(1)

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{seed}"
    logdir = os.path.join('./log_dir', exp_name)
    os.makedirs(logdir, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, scaler = loaddataset(train_data, seq_len)

    best_loss = 9999999.99
    k = 0
    for epoch in range(epochs):
        model.train()
        for i, (x, y, mask, target_mask) in enumerate(train_dataloader):
            x, y, mask, target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
            x = x * mask
            y = y * target_mask
            x_hat = model(x, mask, k)
            loss = torch.sum(torch.abs(x_hat - y) * target_mask) / torch.sum(target_mask)
            optimizer.zero_grad()  # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = evaluate(model, val_dataloader, scaler)
        print('epoch, loss:', epoch, loss)
        if (loss < best_loss):
            best_loss = loss
            best_model = copy.deepcopy(model.state_dict())
            os.makedirs('./output_BiaTCGNet_miss', exist_ok=True)
            torch.save(best_model, './output_BiaTCGNet_miss/best.pth')


def evaluate(model, val_iter, scaler):
    model.eval()
    loss = 0.0
    k = 0
    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(val_iter):
            x, y, mask, target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()

            print(f"Shape of x before model: {x.shape}")  # Debugging

            # 🔹 Ensure x has correct shape: (batch, 1, seq_len, num_nodes)
            if x.shape != (x.shape[0], 1, x.shape[2], x.shape[3]):
                x = x.permute(0, 1, 3, 2)  # (batch, 1, num_nodes, seq_len) → (batch, 1, seq_len, num_nodes)

            print(f"Fixed Shape of x before passing to model: {x.shape}")  # Debugging

            x_hat = model(x, mask, k)  # Model now gets correctly shaped tensor

            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat - y) * target_mask) / torch.sum(target_mask)
            loss += losses

    return loss / len(val_iter)




def run(miss_data, kernel_set=[2,3,6,7], dropout=0.3, subgraph_size=5, node_dim=3, seq_len=16, lr=0.001, epochs=20, seed=42):
    input_data = np.copy(miss_data)

    node_number = input_data.shape[1] if len(input_data.shape) > 1 else 1
    mask = ~np.isnan(input_data)  # Mask where values are observed
    input_data[np.isnan(input_data)] = 0  # Replace NaNs with zeros for processing

    if seq_len == 0:
        seq_len = input_data.shape[1]


    model = Model(True, True, 2, node_number, kernel_set,
              'cuda:0', predefined_A=None,
              dropout=dropout, subgraph_size=subgraph_size, node_dim=node_dim,
              dilation_exponential=1, conv_channels=8, residual_channels=8,
              skip_channels=16, end_channels= 32, seq_length=seq_len, in_dim=1, out_len=input_data.shape[0], out_dim=1,
              layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True) #2 4 6

    if torch.cuda.is_available():
        model = model.cuda()

    best_model = train(model, miss_data, mask, lr, epochs, seq_len, seed)
    model.load_state_dict(best_model)
    model.eval()

    with torch.no_grad():
        train_tensor = torch.tensor(miss_data, dtype=torch.float32).cuda()
        mask_tensor = torch.tensor(mask, dtype=torch.float32).cuda()
        x_hat = model(train_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), 0)
        imputed_data = x_hat.squeeze(0).cpu().numpy()

    return imputed_data

if __name__ == '__main__':
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))  # shape 64x256
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.mcar(ts_1.data)

    imputation = run(ts_mask)

    print("imputation done", imputation)


