import copy
import torch
import torch.optim as optim
import numpy as np
from imputegap.wrapper.AlgoPython.BitGraph.models.BiaTCGNet.BiaTCGNet import Model
import os
from imputegap.wrapper.AlgoPython.BitGraph.data.GenerateDataset import loaddataset
import datetime

# python main.py --epochs 200 --mask_ratio 0.2 --dataset-name Elec

import imputegap.tools.utils as utils_imp
torch.multiprocessing.set_sharing_strategy('file_system')

def handle_parser(argv=None):
    import argparse
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
    parser.add_argument("--dataset-name", type=str, default='air36' '')
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
    parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')
    ##############transformer config############################

    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
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
    parser.add_argument('--num_nodes', type=int, default=1, help='dimension of fcn')
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

    args, _unknown = parser.parse_known_args(argv)

    return args



# Connectivity params
# parser.add_argument("--adj-threshold", type=float, default=0.1)
args = handle_parser()


def train(ts_m, model, sliding_windows=1, multivariate=False, verbose=True, saved=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    foldername = os.path.join(here, "imputegap_assets/logs/bitgraph/")
    logdir = os.path.join(foldername, exp_name)
    os.makedirs(logdir, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, imp_dataloader, scaler, scaler_reco = loaddataset(ts_m, args.seq_len, args.pred_len, args.mask_ratio, args, sliding_windows=sliding_windows, multivariate=multivariate, verbose=verbose)

    best_loss=9999999.99
    k=0
    for epoch in range(args.epochs):
        model.train()
        for i, (x, y, mask, target_mask) in enumerate(train_dataloader):

            x, mask = x.to(args.device), mask.to(args.device)
            y, target_mask = y.to(args.device), target_mask.to(args.device)

            x= x * mask
            y= y * target_mask
            x_hat=model(x,mask,k)

            loss = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)

            optimizer.zero_grad()  # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_loss=evaluate(model, val_dataloader, scaler)

        if args.verbose:
            print(f'epoch {epoch}, loss: {epoch,model_loss}')

        if(loss<best_loss):
            best_loss=model_loss

            if saved:
                best_model = copy.deepcopy(model.state_dict())
                os.makedirs('./output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task,exist_ok=True)
                torch.save(best_model, './output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task+'/best.pth')

    loss, imputed_matrix = matrix_recon(model, imp_dataloader, scaler_reco)

    if verbose:
        print(f"model loss: {model_loss}\nreconstruction loss : {loss}")

    return imputed_matrix

"""
def reconstruct(model, imp_dataloader, scaler):
    model.eval()
    loss = 0.0
    k = 0
    imputed_matrices = []
    with torch.no_grad():
        for i, (x, y, mask, target_mask) in enumerate(imp_dataloader):

            x, mask = x.to(args.device), mask.to(args.device)
            y, target_mask = y.to(args.device), target_mask.to(args.device)
            x_hat = model(x, mask, k)

            #print(f"{x.shape = }")
            #print(f"{x_hat.shape = }")

            if scaler is not None:
                x_hat = scaler.inverse_transform(x_hat)
                y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat - y) * target_mask) / torch.sum(target_mask)
            losses = torch.nan_to_num(losses, nan=0.0)

            loss += losses

            imp = x_hat.detach().cpu().numpy()
            imputed_matrices.append(imp)

    if args.verbose:
        print(f"\ntotal loss : {loss/len(imp_dataloader)}\n")

    imputed_matrix = np.concatenate(imputed_matrices, axis=0)

    if imputed_matrix.shape[1] == 1:
        imputed_matrix = np.squeeze(imputed_matrix)
    else:
        imputed_matrix = np.squeeze(imputed_matrix)
        imputed_matrix = utils_imp.dataset_reverse_dimensionality(imputed_matrix, expected_n=args.dim, verbose=args.verbose)

    return imputed_matrix

"""

def matrix_recon(model, val_iter, scaler):
    model.eval()
    loss=0.0
    k=0
    imputed_matrix=[]
    with torch.no_grad():
        for i, (x,y,mask,target_mask) in enumerate(val_iter):
            x, mask = x.to(args.device), mask.to(args.device)
            y, target_mask = y.to(args.device), target_mask.to(args.device)

            x_hat=model(x,mask,k)

            if scaler is not None:
                x_hat = scaler.inverse_transform(x_hat)
                y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
            loss+=losses

            imputation = x_hat.detach().cpu()
            imputed_matrix.append(imputation)

    imputed_matrix = torch.cat(imputed_matrix, dim=0).numpy()
    imputed_matrix = imputed_matrix.squeeze()

    return loss/len(val_iter), imputed_matrix


def evaluate(model, val_iter, scaler):
    model.eval()
    loss=0.0
    k=0
    with torch.no_grad():
        for i, (x,y,mask,target_mask) in enumerate(val_iter):

            x, mask = x.to(args.device), mask.to(args.device)
            y, target_mask = y.to(args.device), target_mask.to(args.device)

            x_hat=model(x,mask,k)

            if scaler is not None:
                x_hat = scaler.inverse_transform(x_hat)
                y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
            loss+=losses

    return loss/len(val_iter)


def recovBitGRAPH(ts_m=None, seq_len=24, sliding_windows=1, pred_len=0, kernel_size=1, kernel_set=[2,3,6,7], epochs=10, batch_size=32, subgraph_size=5, num_workers=0, tr_ratio=0.7, seed=0, norma=True, verbose=True, replicat=False):

    recov = np.copy(ts_m)
    m_mask = np.isnan(ts_m)

    if seq_len ==-1 or batch_size==-1:
        seq_len, batch_size = utils_imp.auto_seq_llms(data_x=ts_m, goal="seq", subset=True, high_limit=96, exception=True, b=True, verbose=verbose)

    ts_r, val_r = utils_imp.sets_splitter_based_on_training(tr_ratio, split=0.5, verbose=verbose)

    if sliding_windows == 0:
        multivariate = True
        ts_m = utils_imp.dataset_add_dimensionality(ts_m, seq_length=seq_len, three_dim=False, verbose=False)
    else:
        multivariate = False

    if replicat:
        seq_len = 24
        batch_size = 64
        epochs = 20
        args.mask_ratio = 0.2

    node_number = ts_m.shape[1]
    args.num_nodes = ts_m.shape[1]
    args.enc_in = ts_m.shape[1]
    args.dec_in = ts_m.shape[1]
    args.c_out = ts_m.shape[1]
    args.kernel_size = kernel_size
    args.seq_len = seq_len
    args.pred_len = seq_len
    args.epochs = epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args.subgraph_size = subgraph_size
    args.kernel_set = kernel_set
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.node_number = node_number
    args.ts_r = ts_r
    args.val_r = val_r
    args.verbose = verbose
    args.deep_verbose = False
    args.dim = ts_m.shape[0]
    if not replicat:
        args.mask_ratio = round(1 - tr_ratio, 1)
    args.norma = norma

    if verbose:
        print(f"\n(IMPUTATION) BitGRAPH\n\tMatrix Shape: ({ts_m.shape[0]}, {ts_m.shape[1]})\n\tseq_len: {args.seq_len}\n\tsliding_windows: {sliding_windows}\n\tkernel_size: {args.kernel_size}\n\tkernel_set: {args.kernel_set}\n\tbatch_size: {args.batch_size}\n\tepochs: {args.epochs}\n\tsubgraph_size: {args.subgraph_size}\n\tnum_workers: {num_workers}"
              f"\n\tpred_len: {args.pred_len}\n\tseed {seed}\n\tnode_number: {args.node_number}\n\tenc_in: {args.enc_in}\n\tdec_in: {args.dec_in}\n\tc_out: {args.c_out}\n\ttr_ratio: {tr_ratio}\n\tval_ratio: {args.val_r}\n\tts_ratio: {args.ts_r}")
        print(f"\ncall: bitgraph.impute(params={{'seq_len': {seq_len}, 'sliding_windows': {sliding_windows}, 'kernel_size': '{args.kernel_size}', 'kernel_set': {args.kernel_set}, 'epochs': {args.epochs}, 'batch_size': {args.batch_size}, 'subgraph_size': {args.subgraph_size}, 'num_workers': {num_workers}}})\n")

    model=Model(True, True, 2, node_number, args.kernel_set, device, predefined_A=None, dropout=0.3, subgraph_size=args.subgraph_size, node_dim=3, dilation_exponential=1, conv_channels=8, residual_channels=8, skip_channels=16, end_channels= 32, seq_length=args.seq_len, in_dim=1,out_len=args.pred_len, out_dim=1, layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True) #2 4 6
    if torch.cuda.is_available():
        model = model.cuda()

    imputed_matrix = train(ts_m, model, sliding_windows=sliding_windows, multivariate=multivariate, verbose=verbose)

    if multivariate:
        imputed_matrix = utils_imp.dataset_reverse_dimensionality(imputed_matrix, expected_n=recov.shape[0])
    else:
        imputed_matrix = utils_imp.reconstruction_windowd_based(preds=imputed_matrix, nbr_timestamps=recov.shape[0], verbose=verbose, deep_verbose=False)

    if verbose:
        print(f"\t{imputed_matrix.shape =}\n")

    recov[m_mask] = imputed_matrix[m_mask]

    return recov
