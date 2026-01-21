# ===============================================================================================================
# SOURCE: https://github.com/thuml/TimesNet
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://iclr.cc/virtual/2023/poster/11976
# ===============================================================================================================

from imputegap.wrapper.AlgoPython.TimesNet.data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, normalization=False, scaling=False, shuffle=False, reconstruction=False, verbose=True, tr_ratio=0.7, replicat=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = shuffle

    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        ts_m=args.ts_m,
        reconstruction=reconstruction,
        tr_ratio=tr_ratio,
        verbose=verbose,
        replicat=replicat,
        normalizer=normalization,
        scale=scaling
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader, len(data_set)
