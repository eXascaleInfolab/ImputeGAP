from imputegap.wrapper.AlgoPython.Moment.momentfm.utils.utils import control_randomness
from imputegap.wrapper.AlgoPython.Moment.momentfm.models.moment import MOMENTPipeline
from imputegap.tools import utils
import torch

from imputegap.wrapper.AlgoPython.Moment.momentfm.data.informer_dataset import InformerDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from imputegap.wrapper.AlgoPython.Moment.momentfm.utils.masking import Masking
from tqdm import tqdm
import imputegap.tools.utils as utils_imp


def init_model(model_size="small", seed=0, verbose=True):
    control_randomness(seed=seed)  # Set random seeds for PyTorch, Numpy etc.

    if model_size == "small":
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={'task_name': 'reconstruction'}
        )
    elif model_size == "medium":
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-base",
            model_kwargs={'task_name': 'reconstruction'}
        )
    else:
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={'task_name': 'reconstruction'}  # For imputation, we will load MOMENT in `reconstruction` mode
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )

    model.init()
    if verbose:
        print(model)

    # Number of parameters in the encoder
    num_params = sum(p.numel() for p in model.encoder.parameters())
    if verbose:
        print(f"\nNumber of parameters: {num_params}\n")

    return model


def recovery_moment(incomp_data, model="zero-shot", model_size="small", use_mean=True, seq_len=-1, sliding_windows=1, seed=26, verbose=False, replicat=False):
    if model == "zero-shot":
        return recov_moment_zeroshot(incomp_data=incomp_data, model=model, model_size=model_size, use_mean=use_mean, seq_len=seq_len, sliding_windows=sliding_windows, verbose=verbose, replicat=replicat)
    else:
        return recov_moment_train(incomp_data=incomp_data, model=model, model_size=model_size, use_mean=use_mean, seq_len=seq_len, sliding_windows=sliding_windows, verbose=verbose, replicat=replicat)


def recov_moment_zeroshot(incomp_data, model="zero-shot", model_size="small", use_mean=True, seq_len=-1, sliding_windows=1, seed=26, verbose=False, replicat=False):

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)
    size_limitation = False

    if seq_len ==-1:
        seq_len, batch_size, size_limitation, taux_nans, tr_ratio = utils_imp.compute_variables_timesnetlike(ts_m=ts_m, strat="seq", tr_ratio=1, size_limitation=size_limitation, model=model, high=32,verbose=verbose)
        old = seq_len
        seq_len = (seq_len // 8) * 8
        if verbose:
            print(f"(INFO) Adapation of the sequence length to the model patch_len ({old}//8)*8) : {seq_len}.")

    taux_nans = utils_imp.max_consecutive_nans_per_col_loop(ts_m).max(initial=0)

    if seq_len < taux_nans and not use_mean:
        #use_mean = True
        if verbose:
            print(f"\n\n(INFO) Imputation adaptation with {model}, the number of NaNs values injected with ImputeGAP filled all the sequence, please change the seq_len or the contamination percentage. \n\tCurrently, the number of NaNs might reach {taux_nans} for a sequence and the size of seq_len is {seq_len}.\n(INFO) NaNs sequence is put to mean {use_mean = }\n")

    if verbose:
        print(f"(IMPUTATION) MOMENT\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tmodel: {model}\n\tmodel_size: {model_size}\n\tuse_mean: {use_mean}\n\tseq_len: {seq_len}\n\tsliding_windows: {sliding_windows}\n\tseed: {seed}\n\tverbose: {verbose}\n\tmax consecutive nans: {taux_nans}\n")
        print(f"\ncall: moment.impute(params={{'model': {model}, 'model_size': {model_size}, 'use_mean': {use_mean}, 'seq_len': {seq_len}, 'sliding_windows': {sliding_windows}}})\n\n")

    model = init_model(model_size=model_size, seed=seed, verbose=verbose)

    x = utils.window_truncation(ts_m, seq_len=seq_len, stride=sliding_windows, info="moments", verbose=False, deep_verbose=False)
    x = torch.as_tensor(x)  # converts numpy -> torch (keeps dtype if possible)
    x_enc = x.permute(0, 2, 1).contiguous()   # [B, C, L]

    if use_mean:
        # Fill NaNs in x (model also nan_to_num later, but do it early to be safe)
        mean = torch.nanmean(x_enc)
        mean = torch.nan_to_num(mean, nan=0.0)  # if everything is NaN
        x_enc = torch.where(torch.isnan(x_enc), mean, x_enc)

    output = model(x_enc=x_enc)

    if verbose:
        print(f"\nreconstruction...\n")

    recovery = output.reconstruction.permute(0, 2, 1).contiguous()  # [B, L, C]

    recovery_matrix = utils.reconstruction_window_based(preds=recovery, nbr_timestamps=incomp_data.shape[0], sliding_windows=sliding_windows, verbose=False, deep_verbose=False)
    recovery_matrix = recovery_matrix.detach().cpu().numpy()

    if verbose:
        print(f"sanity check : {np.isnan(recovery_matrix).any() = }")
        print(f"{recovery_matrix.shape = }")


    recov[m_mask] = recovery_matrix[m_mask]

    return recov, model



def make_mask(input_mask, mask_ratio):
    mask = torch.ones_like(input_mask)

    available_idx = torch.where(input_mask == 1)[0]
    n = int(len(available_idx) * mask_ratio)

    if n > 0:
        chosen = available_idx[torch.randperm(len(available_idx))[:n]]
        mask[chosen] = 0

    return mask.long()

def recov_moment_train(incomp_data, model="train", model_size="small", use_mean=True, seq_len=-1, sliding_windows=1, seed=26, verbose=False, replicat=False):

    ts_m = np.copy(incomp_data)
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)
    size_limitation = False

    if seq_len == -1:
        seq_len, batch_size, size_limitation, taux_nans, tr_ratio = utils_imp.compute_variables_timesnetlike(ts_m=ts_m, strat="seq", tr_ratio=0.7, size_limitation=size_limitation, model=model, high=32, verbose=verbose, consecutive=True)

        if seq_len == -10:
            seq_len = 128
            use_mean = True

        old = seq_len
        seq_len = utils_imp.max_consecutive_nans_per_col_loop(ts_m).max(initial=0) +9
        seq_len = (seq_len // 8) * 8
        if verbose:
            print(f"(INFO) Adapation of the sequence length to the model patch_len ({old}//8)*8) : {seq_len}.")
        mask_ratio = 0.1

    taux_nans = utils_imp.max_consecutive_nans_per_col_loop(ts_m).max(initial=0)
    if seq_len < taux_nans and not use_mean:
        mask_ratio = 0.01
        if verbose:
            print(f"\n\n(INFO) Imputation adaptation with {model}, the number of NaNs values injected with ImputeGAP filled all the sequence, please change the seq_len or the contamination percentage. \n\tCurrently, the number of NaNs might reach {taux_nans} for a sequence and the size of seq_len is {seq_len}.\n(INFO) NaNs sequence is put to mean {use_mean = }\n")

    if verbose:
        print(f"(IMPUTATION) MOMENT\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tmodel: {model}\n\tmodel_size: {model_size}\n\tuse_mean: {use_mean}\n\tseq_len: {seq_len}\n\tsliding_windows: {sliding_windows}\t\n\tseed: {seed}\n\tverbose: {verbose}\n\tmax consecutive nans: {taux_nans}\n")
        print(f"\ncall: moment.impute(params={{'model': {model}, 'model_size': {model_size}, 'use_mean': {use_mean}, 'seq_len': {seq_len}, 'sliding_windows': {sliding_windows}}})\n\n")

    model = init_model(model_size=model_size, seed=seed, verbose=False)

    #x = utils.window_truncation(ts_m, seq_len=seq_len, stride=sliding_windows, info="moments", verbose=False, deep_verbose=False)
    #x = torch.as_tensor(x)  # converts numpy -> torch (keeps dtype if possible)
    #x_enc = x.permute(0, 2, 1).contiguous()  # [B, C, L]

    if use_mean:
        col_means = np.nanmean(ts_m, axis=0)  # mean per column, ignoring NaNs
        inds = np.where(np.isnan(ts_m))  # locations of NaNs
        ts_m[inds] = np.take(col_means, inds[1])  # fill with the mean of that column

    test_dataset = InformerDataset(
        data_split='test',
        task_name='imputation',
        data_stride_len=1,
        ts_m=ts_m,
        seq_len=seq_len)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    mask_generator = Masking(mask_ratio=mask_ratio)  # Mask 25% of patches randomly

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).float()

    trues, preds, masks = [], [], []
    with torch.no_grad():
        for batch_x, batch_masks, input_mask in tqdm(test_dataloader, total=len(test_dataloader)):
            trues.append(batch_x.numpy())

            batch_x = batch_x.to(device).float()
            n_channels = batch_x.shape[1]

            # Reshape to [batch_size * n_channels, 1, window_size]
            batch_x = batch_x.reshape((-1, 1, seq_len))
            input_mask = input_mask.reshape((-1, 1, seq_len)).squeeze()

            batch_masks = batch_masks.to(device).long()
            input_mask = input_mask.to(device).long()

            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
            #mask = mask_generator.generate_mask(x=batch_x, input_mask=input_mask).to(device).long()

            masks_ = []
            for inp in input_mask:
                mask = make_mask(inp, mask_ratio)
                masks_.append(mask)
            mask = torch.stack(masks_).to(device)

            #for a, b,x, c in zip(batch_x, batch_masks, mask, input_mask):
            #    print(f"batch_x____:{a}")
            #    print(f"batch_masks:{b}")
            #    print(f"mask_______:{x}")
            #    print(f"input_mask_:{c}")
            #    print("\n\n\n")

            # mask = batch_mask
            #output = model(x_enc=batch_x, input_mask=batch_mask, mask=mask)  # [batch_size, n_channels, window_size]
            batch_x = torch.nan_to_num(batch_x, nan=0.0)
            output = model(x_enc=batch_x, input_mask=input_mask, mask=mask)

            reconstruction = output.reconstruction.detach().cpu().numpy()
            input_mask = input_mask.detach().squeeze().cpu().numpy()

            # Reshape back to [batch_size, n_channels, window_size]
            reconstruction = reconstruction.reshape((-1, n_channels, seq_len))
            input_mask = input_mask.reshape((-1, n_channels, seq_len))

            preds.append(reconstruction)
            masks.append(input_mask)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    masks = np.concatenate(masks)

    if verbose:
        print(f"Shapes: preds={preds.shape} | trues={trues.shape} | masks={masks.shape}")

    recovery = np.transpose(preds, (0, 2, 1))  # (B, L, C)
    recovery = np.ascontiguousarray(recovery)

    recovery_matrix = utils.reconstruction_window_based(preds=recovery, nbr_timestamps=incomp_data.shape[0], sliding_windows=sliding_windows, verbose=False, deep_verbose=False)
    recovery_matrix = recovery_matrix.detach().cpu().numpy()

    if verbose:
        print(f"sanity check : {np.isnan(recovery_matrix).any() = }")
        print(f"{recovery_matrix.shape = }")

    recov[m_mask] = recovery_matrix[m_mask]

    return recov, model

