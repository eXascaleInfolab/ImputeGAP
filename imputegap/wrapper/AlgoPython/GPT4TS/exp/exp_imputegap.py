# ===============================================================================================================
# SOURCE: google.com/url?q=https://github.com/DAMO-DI-ML/One_Fits_All&sa=D&source=editors&ust=1763995357004903&usg=AOvVaw33kubL9FDsXc_ZL-M_onqA
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://dl.acm.org/doi/10.5555/3666122.3667999
# ===============================================================================================================

from imputegap.wrapper.AlgoPython.GPT4TS.data_provider.data_factory import data_provider
from imputegap.wrapper.AlgoPython.GPT4TS.exp.exp_basic import Exp_Basic
from imputegap.wrapper.AlgoPython.GPT4TS.utils.tools import EarlyStopping, adjust_learning_rate, visual
from imputegap.wrapper.AlgoPython.GPT4TS.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import imputegap.tools.utils as utils_imp

warnings.filterwarnings('ignore')


class Exp_ImputeGAP(Exp_Basic):
    def __init__(self, args):
        super(Exp_ImputeGAP, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, normalization=False, scaling=False, shuffle=False, reconstruction=False, verbose=True, tr_ratio=0.7, replicat=False):
        data_set, data_loader, l = data_provider(self.args, flag, normalization=normalization, scaling=scaling, shuffle=shuffle, reconstruction=reconstruction, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        return data_set, data_loader, l

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, normalization=False, verbose=False):
        total_loss = []
        d_tqdm = not verbose

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in tqdm(enumerate(vali_loader), disable=d_tqdm):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask[imputegap_mask == 0] = 1

                inp = batch_x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, batch_x_mark, None, None, mask, normalization)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting, verbose, deep_verbose, normalization=False, scaling=False, shuffle=False, tr_ratio=0.7, replicat=False):
        train_data, train_loader, len_tr = self._get_data(flag='train', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        vali_data, vali_loader, len_val = self._get_data(flag='val', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        test_data, test_loader, len_ts = self._get_data(flag='test', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio,  replicat=replicat)
        recon_data, recon_loader, len_recon = self._get_data(flag='train', scaling=scaling, shuffle=shuffle, reconstruction=True, verbose=verbose, replicat=replicat)

        if verbose:
            print(f"\nresults of the pre-processing :\n\t{len(train_loader) = } - {len_tr = }\n\t{len(vali_loader) = } - {len_val = }\n\t{len(test_loader) = } - {len_ts = }\n\t{len(recon_loader) = } - {len_recon = }\n")

        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(here, "imputegap_assets/models/checkpoints")
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=verbose)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            if verbose:
                print(f"\n\n---gpt4ts---epoch:{epoch+1}/{self.args.train_epochs}---imputegap----------------------------------------------------------------------------------------------------------")
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            d_tqdm = not verbose
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in tqdm(enumerate(train_loader), disable=d_tqdm):

                v = ((i + 1) % 250 == 0) # display

                if verbose and (i == 0 or v):
                    print(f"training checkup: {batch_x.shape = } - {imputegap_mask.shape = }- {i + 1}/{len(train_loader) + 1}")

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask[imputegap_mask == 0] = 1

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask, normalization)

                if verbose and (i == 0 or v):
                    print(f"\tbatch NaNs: {torch.isnan(batch_x).any()} | imp NaNs: {torch.isnan(inp).any()} | imp NaNs: {torch.isnan(outputs).any()} | mask with 0: {(imputegap_mask == 0).any()}")

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if v:
                    if verbose:
                        print("\n\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if verbose:
                print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, normalization, verbose=verbose)
            test_loss = self.vali(test_data, test_loader, criterion, normalization, verbose=verbose)

            if verbose:
                print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                if verbose:
                    print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args, verbose=verbose)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, path

    def test(self, setting, test=0, normalization=False, scaling=False, shuffle=False, reconstruction=False, original_shape=None, verbose=True, replicat=False, checkpoint=None, deep_verbose=False):

        if not reconstruction:
            test_data, test_loader, _ = self._get_data(flag='test', scaling=scaling, shuffle=shuffle, verbose=verbose, replicat=replicat)
        else:
            test_data, test_loader, _ = self._get_data(flag='train', scaling=scaling, shuffle=shuffle, reconstruction=reconstruction, verbose=verbose, replicat=replicat)

        if test:
            self.model.load_state_dict(torch.load(os.path.join(checkpoint, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = checkpoint
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        d_tqdm = not verbose

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in tqdm(enumerate(test_loader), disable=d_tqdm):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained

                mask[imputegap_mask == 0] = 0

                inp = batch_x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, batch_x_mark, None, None, mask, normalization)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0 and deep_verbose:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])

        if verbose:
            print('\nmse:{}, mae:{}, rmse:{}, shape:{}'.format(mse, mae, rmse, preds.shape))

        if original_shape is not None:
            recons = utils_imp.reconstruction_window_based(preds=preds, nbr_timestamps=original_shape[0], verbose=verbose, deep_verbose=False)
            return recons

        return None
