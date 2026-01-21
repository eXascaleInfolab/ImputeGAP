from imputegap.wrapper.AlgoPython.NuwaTS.exp.exp_imputation import Exp_Imputation
import random
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import imputegap.tools.utils as utils_imp
from imputegap.wrapper.AlgoPython.TimesNet.utils.print_args import print_args


def handle_parser(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='NuwaTS')

    # basic config
    parser.add_argument('--task_name', type=str, default='imputation', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, denoise]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='NuwaTS', help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='imputegap_datasret', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # imputation task
    parser.add_argument('--test_mask_rate', type=float, default=0.8, help='test mask ratio')
    parser.add_argument('--mask_rate', type=float, default=0.125, help='test mask ratio')
    parser.add_argument('--max_iterations', type=int, default=10, help='max iterations')
    parser.add_argument('--max_optimization_iterations', type=int, default=10, help='max optimization iterations')
    parser.add_argument('--regularization_weight', type=float, default=0.05, help='regularization weight')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # patching
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--ln', type=int, default=0)
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--percent', type=int, default=5)

    # prefix tuning
    parser.add_argument('--prefix_tuning', action='store_true', help='', default=False)
    parser.add_argument('--prefix_tuningv2', action='store_true', help='', default=False)
    parser.add_argument('--continue_tuning', action='store_true', help='', default=False)
    parser.add_argument('--continue_tuningv2', action='store_true', help='', default=False)

    parser.add_argument('--frozen_lm', action='store_true', help='', default=False)
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--train_all_lm', action='store_true', help='', default=False)
    parser.add_argument('--use_llama', action='store_true', help='', default=False)
    parser.add_argument('--use_bert', action='store_true', help='', default=False)
    parser.add_argument('--alignment', action='store_true', help='', default=False)

    # contrastive
    parser.add_argument('--con_weight', type=float, default=0.01, help='')
    parser.add_argument('--patch_con', action='store_true', help='', default=False)
    parser.add_argument('--temporal_con', action='store_true', help='', default=False)
    parser.add_argument('--flatten_con', action='store_true', help='', default=False)
    parser.add_argument('--best_con_num', type=int, default=128)
    # output learnable token
    parser.add_argument('--seq_token', type=int, default=0)
    parser.add_argument('--word_prompt', action='store_true', help='', default=False)
    parser.add_argument('--cov_prompt', action='store_true', help='', default=False)
    parser.add_argument('--output_token', action='store_true', help='', default=False)

    # test
    parser.add_argument('--test_all', action='store_true', help='', default=False)

    # forecasting
    parser.add_argument('--is_forecasting', action='store_true', help='', default=False)
    parser.add_argument('--auto_regressive', action='store_true', help='', default=False)

    parser.add_argument('--origin_missrate', type=float, default=0, help='')

    args, _unknown = parser.parse_known_args(argv)

    return args



def recovLLMs(ts_m, model="NuwaTS", seq_len=-1, batch_size=-1, epochs=10, gpt_layers=6, num_workers=0, seed=2023, tr_ratio=0.7, verbose=True, deep_verbose=False, replicat=False, normalization=True, scaling=True, shuffle=True, strat="seq"):

    if seq_len == -2:
        return np.nan_to_num(ts_m, nan=0.0)

    miss = np.copy(ts_m)
    recov = np.copy(ts_m)
    m_mask = np.isnan(ts_m)

    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = handle_parser()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.ts_m = ts_m

    args.is_training = 1
    args.model_id = "imputegap"
    args.mask_rate = 0.1
    args.model = model
    args.data = "custom"
    args.features = "M"
    args.label_len = 0
    args.pred_len = 0
    args.stride = 1
    args.d_model = 768
    args.des = "Exp"
    args.itr = 1
    args.mlp = 1
    args.prefix_length = 1
    args.prefix_tuning = True
    args.cov_prompt = True
    args.learning_rate = 0.001

    args.seq_len = seq_len
    args.batch_size = batch_size
    args.gpt_layers = gpt_layers
    args.train_epochs = epochs
    args.verbose = verbose
    args.deep_verbose = deep_verbose
    args.num_workers = num_workers
    args.tr_ratio = tr_ratio
    args.normalization = normalization
    args.scaling = scaling
    args.shuffle = shuffle
    args.enc_in = ts_m.shape[1]
    args.dec_in = ts_m.shape[1]
    args.c_out = ts_m.shape[1]

    if seq_len ==-1 or batch_size==-1:
        args.seq_len, args.batch_size = utils_imp.auto_seq_llms(data_x=ts_m, goal=strat, subset=False, high_limit=32, verbose=verbose)
        args.patch_size = args.seq_len
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.patch_size = seq_len//3

    if args.patch_size <= 0:
        args.patch_size = 1

    if replicat:
        args.seq_len = 96
        args.batch_size = 16
        args.patch_size = 1
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.iter = 1
        args.data = "ETTh1"
        normalization = True
        scaling = True
        shuffle = True

    if verbose:
        print(f"(IMPUTATION) {model} (LLMs)\n\tMatrix: {miss.shape[0]}, {miss.shape[1]}\n\tseq_len: {args.seq_len}\n\tbatch_size: {args.batch_size}\n\tepochs: {args.train_epochs}\n\tenc_in: {args.enc_in}\n\tdec_in: {args.dec_in}\n\tc_out: {args.c_out}\n\tgpt_layers: {args.gpt_layers}\n\tnum_workers: {args.num_workers}\n\ttr_ratio: {args.tr_ratio}\n\tseed: {seed}\n\tverbose: {verbose}\n\tGPU: {args.use_gpu}\n")

        print('\nArgs in experiment:')
        print_args(args)

    Exp = Exp_Imputation

    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = 'imputegap_settings'

        if args.verbose:
            print(f'\n\n>>>>>>>training>>>>>>>>>>>>>>>>>>>>>>>>>>')

        m, path = exp.train(setting, verbose=args.verbose, deep_verbose=args.deep_verbose, tr_ratio=args.tr_ratio, normalization=normalization, scaling=scaling, shuffle=shuffle, replicat=replicat)

        if args.verbose:
            print(f'\n\n>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        _ = exp.test(setting, test=0, verbose=args.verbose, normalization=args.normalization, scaling=args.scaling, replicat=replicat, checkpoint=path)
        if args.verbose:
            print(f'\n\n>>>>>>>reconstruction<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        reconstruction = exp.test(setting, test=0, reconstruction=True, normalization=args.normalization, scaling=args.scaling, replicat=replicat, original_shape=ts_m.shape, verbose=args.verbose, checkpoint=path)
        reconstruction = np.array(reconstruction)
        torch.cuda.empty_cache()

        if np.isnan(reconstruction).any():
            print(f"\n(ERROR) Imputation error with {model}, the number of NaNs values injected with ImputeGAP filled all the sequence, please change the seq_len or the contamination percentage.\n")
            return recov

        if replicat:
            recov = reconstruction
        else:
            if np.isnan(recov).any():
                recov[m_mask] = reconstruction[m_mask]
                if verbose:
                    print(f"\n\nmodel checkpoint saved: {path}\n\nimputed matrix reconstructed successfully: {recov.shape = }\n")
            else:
                if verbose:
                    print(f"\n\n\nmodel checkpoint saved: {path}\n\n(INFO) no NaNs has been detected in the input matrix: {reconstruction.shape = }\n")
                recov = reconstruction

        return recov


