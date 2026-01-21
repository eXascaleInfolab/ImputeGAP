# ===============================================================================================================
# SOURCE: google.com/url?q=https://github.com/DAMO-DI-ML/One_Fits_All&sa=D&source=editors&ust=1763995357004903&usg=AOvVaw33kubL9FDsXc_ZL-M_onqA
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://dl.acm.org/doi/10.5555/3666122.3667999
# ===============================================================================================================

from imputegap.wrapper.AlgoPython.GPT4TS.data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, normalization=False, scaling=False, shuffle=False, reconstruction=False, verbose=True, tr_ratio=0.7, replicat=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    shuffle_flag = shuffle


    if args.data == 'm4':
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        percent=percent,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        ts_m = args.ts_m,
        reconstruction=reconstruction,
        tr_ratio=tr_ratio,
        verbose = verbose,
        replicat=replicat,
        normalizer=normalization,
        scale=scaling
    )

    batch_size = args.batch_size
    #print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader, len(data_set)
