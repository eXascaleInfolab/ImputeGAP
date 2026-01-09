from imputegap.wrapper.AlgoPython.NuwaTS.data_provider.data_loader_imputation import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from imputegap.wrapper.AlgoPython.NuwaTS.data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from imputegap.wrapper.AlgoPython.NuwaTS.data_provider.data_fusion import load_dataset_dataloader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, normalization=False, scaling=False, shuffle=False, reconstruction=False, verbose=True, tr_ratio=0.7, replicat=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    drop_last = True
    batch_size = args.batch_size  # bsz=1 for evaluation
    freq = args.freq
    shuffle_flag = shuffle

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        ts_m=args.ts_m,
        reconstruction=reconstruction,
        tr_ratio=tr_ratio,
        verbose=verbose,
        replicat=replicat,
        scale=scaling
    )

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    return data_set, data_loader, len(data_set)
