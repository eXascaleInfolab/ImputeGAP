import pandas as pd
import torch
from imputegap.wrapper.AlgoPython.MPIN.utils.regressor import MLPNet
import torch.optim as optim
import copy
import numpy as np
import random
from datetime import datetime
from imputegap.wrapper.AlgoPython.MPIN.utils.DynamicGNN import DynamicGCN, DynamicGAT, DynamicGraphSAGE, StaticGCN, StaticGraphSAGE, StaticGAT
from imputegap.wrapper.AlgoPython.MPIN.utils.load_dataset import load_imputegap_dataset, load_ICU_dataset, get_model_size
import imputegap.tools.utils as utils_imp


def knn_graph_torch(x, k, loop=False):
    device = x.device  # ensure all tensors are on the same device

    num_nodes = x.size(0)
    dist = torch.cdist(x, x, p=2)  # shape: [N, N]

    if not loop:
        dist.fill_diagonal_(float('inf'))

    knn_idx = dist.topk(k, largest=False).indices  # shape: [N, k]

    row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, k).flatten()
    col = knn_idx.flatten()
    edge_index = torch.stack([row, col], dim=0)  # shape: [2, N*k]

    return edge_index

def handle_parser(argv=None):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--incre_mode", type=str, default='alone')

    parser.add_argument("--window", type=int, default=6)
    parser.add_argument('--stream', type=float, default=1)

    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--site", type=str, default='KDM')
    parser.add_argument("--floor", type=str, default='F1')

    parser.add_argument('--base', type=str, default='SAGE')
    parser.add_argument("--prefix", type=str, default='testN')

    parser.add_argument('--state', type=str, default='true')
    parser.add_argument('--thre', type=float, default=0.25)

    parser.add_argument('--method', type=str, default='DMU')

    parser.add_argument("--num_of_iter", type=int, default=5)
    parser.add_argument("--out_channels", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dynamic", type=str, default='false')
    parser.add_argument("--dataset", type=str, default='ICU')

    args, _unknown = parser.parse_known_args(argv)

    return args




def recovMPIN(incomp_data, window=6, incre_mode="alone", base="SAGE", epochs=15, num_of_iter=5, k=10, seed=2021, tr_ratio=0.7, normalizer=True, verbose=True, deep_verbose=False, replicat=False):

    if window > len(incomp_data):
        print(f"(ERROR) The current seq_length {window} is not adapted to the contaminated matrix {len(incomp_data)} !")
        return incomp_data

    args = handle_parser(argv=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    torch.random.manual_seed(seed)
    random.seed(seed)

    if replicat:
        tr_ratio = 0.5
        window = 6
        epochs = 200
        incre_mode = "data+model"
        k = 10
        args.stream = 1

    out_channels = args.out_channels
    lr = args.lr
    weight_decay = args.weight_decay

    args.k = k
    args.epochs = epochs
    args.num_of_iter = num_of_iter
    args.window = window
    args.incre_mode = incre_mode
    args.base = base

    test_ratio, val_ratio = utils_imp.sets_splitter_based_on_training(tr_ratio)
    eval_ratio = round(test_ratio+val_ratio, 1)
    args.eval_ratio = eval_ratio

    if verbose:
        print(f"(IMPUTATION) MPIN\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\twindow: {args.window}\n\tincre_mode: {args.incre_mode}\n\tbase: {args.base}\n\tepochs: {args.epochs}\n\tnum_of_iter: {num_of_iter}\n\tk: {k}\n\tdevice: {device}\n\ttr_ratio: {tr_ratio}\n\teval_ratio: {args.eval_ratio}")
        print(f"\ncall: mpin.impute(params={{'window': {window}, 'incre_mode': '{incre_mode}', 'base': '{base}', 'epochs': {epochs}, 'num_of_iteration': {num_of_iter}, 'k': {k}}})\n")

    def data_transform(X, X_mask, eval_ratio=0.1, deep_verbose=False):
        eval_mask = np.zeros(X_mask.shape)
        rows, cols = np.where(X_mask==1)
        eval_row_index_index = random.sample(range(len(rows)),int(eval_ratio*len(rows)))
        eval_row_index = rows[eval_row_index_index]
        eval_col_index = cols[eval_row_index_index]
        X_mask[eval_row_index, eval_col_index] = 0
        eval_mask[eval_row_index, eval_col_index] = 1
        eval_X = copy.copy(X)
        X[eval_row_index, eval_col_index] = 0

        if deep_verbose:
            print(f"{eval_ratio}:\n\n{X = }")
            print(f"\n{X_mask = }\n")
            print(f"\n{eval_X = }")
            print(f"\n{eval_mask = }\n")

        return X, X_mask, eval_X, eval_mask

    if replicat:
        base_X = load_ICU_dataset(window=args.window, stream=args.stream, trans=True)
    else:
        base_X = load_imputegap_dataset(ts_m, seq_len=24, stream=args.stream, shuffle=False, verbose=verbose)

    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)
    if normalizer:
        mean_X = np.mean(base_X)
        std_X = np.std(base_X)
        base_X = (base_X - mean_X) / std_X

    if verbose:
        print(f"\tbase data shape: {base_X.shape = }, {base_X_mask.shape = }\n")
        #if normalizer:
        #    print(f"\tcurrent mean/deviation: {mean_X = }/{std_X = }\n")

    def build_GNN(in_channels, out_channels, k, base):
        if base == 'GAT':
            gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
        elif base == 'GCN':
            gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
        elif base == 'SAGE':
            gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)

        return gnn

    def build_GNN_static(in_channels, out_channels, k, base):
        if base == 'GAT':
            gnn = StaticGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
        elif base == 'GCN':
            gnn = StaticGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
        elif base == 'SAGE':
            gnn = StaticGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
        return gnn


    def get_window_data(start, end, ratio):
        X = base_X[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
        X_mask = base_X_mask[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
        return X, X_mask


    def window_imputation(start, end, sample_ratio, initial_state_dict=None, X_last=None, mask_last=None, mae_last=None, transfer=False, state=args.state):

        X, X_mask = get_window_data(start=start, end=end, ratio=sample_ratio)

        ori_X = copy.copy(X)
        feature_dim = ori_X.shape[1]
        ori_X_row = ori_X.shape[0]
        ori_X_mask = copy.copy(X_mask)

        all_mask = copy.copy(X_mask)
        all_X = copy.copy(X)

        if X_last:
            X_last = np.array(X_last)
            # eval_X = np.concatenate([X_last, X], axis=0)
            all_X = np.concatenate([X_last, X], axis=0)
            # eval_mask_last = np.zeros(shp_last)
            # eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
            all_mask = np.concatenate([mask_last, X_mask], axis=0)

            X_last = X_last.tolist()

        #print('all mask shp', all_mask.shape)
        #print('all X shp', all_X.shape)

        all_mask_ts = torch.FloatTensor(all_mask).to(device)

        # gram_matrix = all_mask_ts.matmul(all_mask_ts.transpose(1,0))

        gram_matrix = torch.mm(all_mask_ts, all_mask_ts.t())  # compute the gram product


        # gram_matrix = all_mask @ (all_mask.transpose())
        # print('gram_matrix shp', gram_matrix.shape)
        # print('gram_matrix', gram_matrix)

        gram_vec = gram_matrix.diagonal()
        #print('gram vec shp', gram_vec.shape)
        #print('gram vec', gram_vec)

        gram_row_sum = gram_matrix.sum(dim=0)

        #print('gram_row_sum shp', gram_row_sum.shape)

        #print('gram_row_sum', gram_row_sum)

        value_vec = gram_vec - (gram_row_sum - gram_vec)/(gram_matrix.shape[0]-1)

        #print('value_vec shp', value_vec.shape)
        #print('value_vec:', value_vec)

        # print('max min mean median vec values shp:', max(value_vec), min(value_vec), np.mean(value_vec), np.median(value_vec))

        keep_index = torch.where(value_vec > args.thre * (feature_dim-1))[0]
        keep_index = keep_index.data.cpu().numpy()
        # keep_index = torch.where(value_vec > np.quantile(value_vec, args.thre))

        keep_mask = all_mask[keep_index]

        keep_X = all_X[keep_index]

        X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=args.eval_ratio)

        if X_last:
            X_last = np.array(X_last)
            shp_last = X_last.shape
            eval_X = np.concatenate([X_last, eval_X], axis=0)
            X = np.concatenate([X_last, X], axis=0)
            eval_mask_last = np.zeros(shp_last)
            eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
            X_mask = np.concatenate([mask_last, X_mask], axis=0)

        in_channels = X.shape[1]

        X = torch.FloatTensor(X).to(device)
        X_mask = torch.LongTensor(X_mask).to(device)
        eval_X = torch.FloatTensor(eval_X).to(device)
        eval_mask = torch.LongTensor(eval_mask).to(device)

        # build model
        if args.dynamic == 'true':
            gnn = build_GNN(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
            gnn2 = build_GNN(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
        else:
            gnn = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
            gnn2 = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
            # gnn3 = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)

        model_list = [gnn, gnn2]
        regressor = MLPNet(out_channels, in_channels).to(device)

        if initial_state_dict != None:
            gnn.load_state_dict(initial_state_dict['gnn'])
            gnn2.load_state_dict(initial_state_dict['gnn2'])
            if not transfer:
                regressor.load_state_dict(initial_state_dict['regressor'])

        trainable_parameters = []
        for model in model_list:
            trainable_parameters.extend(list(model.parameters()))

        trainable_parameters.extend(list(regressor.parameters()))
        filter_fn = list(filter(lambda p: p.requires_grad, trainable_parameters))

        num_of_params = sum(p.numel() for p in filter_fn)

        model_size = get_model_size(gnn) + get_model_size(gnn2) + get_model_size(regressor)
        model_size = round(model_size, 6)

        if verbose:
            print(f"number of trainable parameters:{num_of_params}, for a model size of {model_size}")

        num_of_params = num_of_params/1e6

        opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

        graph_impute_layers = len(model_list)
        st = datetime.now()

        X_knn = copy.deepcopy(X)

        edge_index = knn_graph_torch(X_knn, args.k, loop=False)

        min_mae_error = 1e9
        min_mse_error = None
        min_mape_error = None
        opt_epoch = None
        opt_time = None
        best_X_imputed = None
        best_state_dict = None

        for pre_epoch in range(epochs):
            gnn.train()
            gnn2.train()
            regressor.train()
            opt.zero_grad()
            loss = 0
            X_imputed = copy.copy(X)

            # edge_index = None
            for i in range(graph_impute_layers):
                if args.dynamic == 'true':
                    X_emb = model_list[i](X_imputed)
                else:
                    X_emb, edge_index = model_list[i](X_imputed, edge_index)

                if deep_verbose:
                    print(i, 'X_emb shape:', X_emb.shape)
                    print(i, 'X_emd:', X_emb)

                # X_emb = F.relu(X_emb)
                pred = regressor(X_emb)
                X_imputed = X*X_mask + pred*(1 - X_mask)
                temp_loss = torch.sum(torch.abs(X - pred) * X_mask) / (torch.sum(X_mask) + 1e-5)
                # print('temp loss:', temp_loss.item())
                loss += temp_loss

            loss.backward()
            opt.step()
            train_loss = loss.item()

            if deep_verbose:
                print('epoch {n}: loss:'.format(n=pre_epoch), train_loss)

            # trans_X = X_imputed * std_f + mean_f
            trans_X = copy.copy(X_imputed)
            # trans_eval_X = eval_X * std_f + mean_f
            trans_eval_X = copy.copy(eval_X)
            #print('trans_X shape', trans_X.shape)
            #print('trans_eval_X shape', trans_eval_X.shape)

            epoch_state_dict = {'gnn': gnn.state_dict(), 'gnn2': gnn2.state_dict(),  'regressor': regressor.state_dict()}

            gnn.eval()
            gnn2.eval()
            regressor.eval()

            with torch.no_grad():

                mae_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask) / torch.sum(eval_mask)
                mse_error = torch.sum(((trans_X - trans_eval_X) ** 2) * eval_mask) / torch.sum(eval_mask)
                mape_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask) / (torch.sum(torch.abs(trans_eval_X) * eval_mask) + 1e-12)

                #print('{epcoh}:valid impute value samples:'.format(epcoh=pre_epoch), (trans_X[torch.where(eval_mask == 1)]))
                #print('valid true value samples:', (trans_eval_X[torch.where(eval_mask == 1)]))

                # mae_error_list.append(round(mae_error.item(), 6))
                # print('valid min impute error MAE:', min(mae_error_list))

                if mae_error.item() < min_mae_error:
                    opt_epoch = copy.copy(pre_epoch)
                    min_mae_error = round(mae_error.item(), 6)
                    #print('epoch:{epoch}: opt_mae_error'.format(epoch=pre_epoch), min_mae_error)

                    min_mse_error = round(mse_error.item(), 6)
                    min_mape_error = round(mape_error.item(), 6)

                    opt_time = (datetime.now()-st).total_seconds()/60
                    opt_time = round(opt_time, 6)
                    #print('\t{epoch}_opt time:'.format(epoch=pre_epoch), opt_time)

                    if deep_verbose:
                        print(f'\tvalid impute error MAE: {mae_error.item()} - MSE: {mse_error.item()} - MRE: {mape_error.item()}')

                    best_X_imputed = copy.copy(X_imputed)
                    best_X_imputed = best_X_imputed.data.cpu().numpy()[-ori_X_row:]
                    best_X_imputed = best_X_imputed*(1-ori_X_mask) + ori_X*ori_X_mask
                    best_state_dict = copy.copy(epoch_state_dict)

        if verbose:
            print(f'\tvalid impute error MAE: {mae_error.item()} - MSE: {mse_error.item()} - MRE: {mape_error.item()}')

        et = datetime.now()
        total_elapsed_time = round((et-st).total_seconds()/60, 6)
        results_list = [opt_epoch, min_mae_error, min_mse_error, min_mape_error, num_of_params, model_size, opt_time, total_elapsed_time]

        if mae_last and (min_mae_error > mae_last) and (state == 'true'):
            best_state_dict = copy.copy(initial_state_dict)

        if best_X_imputed is None:
            print(f"\n(ERR) The window chosen is not working for this dataset\n\tPlease decrease the value of the window variable (window={window}).\n")

        if verbose:
            print(f"imputed matrix shape for the window: {best_X_imputed.shape} - done!\n")

        return best_state_dict, keep_X.tolist(), keep_mask, results_list, min_mae_error, best_X_imputed

    #incre_mode = args.incre_mode # 'alone',  'data', 'state', 'state+transfer', 'data+state', 'data+state+transfer'
    prefix = args.prefix

    num_windows = args.window

    results_schema = ['opt_epoch', 'opt_mae', 'mse', 'mre', 'para', 'memo', 'opt_time', 'tot_time']

    num_of_iteration = args.num_of_iter
    iter_results_list = []

    for iteration in range(num_of_iteration):
        results_collect = []
        imputation = []
        for w in range(num_windows):
            if verbose:
                print(f'\niteration {iteration}: time window n°{w} ({incre_mode = })')
            if w == 0 :
                window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows)
            else:
                if incre_mode == 'alone':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows)
                elif incre_mode == 'data':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows, X_last=X_last, mask_last=mask_last)
                elif incre_mode == 'state':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows,initial_state_dict=window_best_state, mae_last=mae_last)
                elif incre_mode == 'state+transfer':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows,initial_state_dict=window_best_state, transfer=True, mae_last=mae_last)
                elif incre_mode == 'data+state':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, mae_last=mae_last)
                elif incre_mode == 'data+state+transfer':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, transfer=True, mae_last=mae_last)

            results_collect.append(window_results)
            best_X_imputed = best_X
            imputation.append(best_X_imputed)

        df = pd.DataFrame(results_collect, index=range(num_windows), columns=results_schema)
        imputed_matrix = np.vstack(imputation)
        iter_results_list.append(df)

    avg_df = sum(iter_results_list)/num_of_iteration

    if verbose:
        avg_df_with_mean = avg_df.copy()
        print(f"\n{imputed_matrix.shape = }\n")
        pd.set_option('display.max_columns', None)  # show all columns
        pd.set_option('display.width', None)  # don't wrap lines
        pd.set_option('display.precision', 2)  # control float precision
        pd.set_option('display.float_format', lambda x: f"{x:,.6f}")
        print("Averaged results over windows:")
        avg_df_with_mean.loc["mean"] = avg_df.mean(numeric_only=True)
        print(avg_df_with_mean.to_string())

    recov[m_mask] = imputed_matrix[m_mask]

    return np.array(recov)

    return recov





