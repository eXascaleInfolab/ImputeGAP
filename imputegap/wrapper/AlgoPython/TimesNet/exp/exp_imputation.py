# ===============================================================================================================
# SOURCE: https://github.com/thuml/TimesNet
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://iclr.cc/virtual/2023/poster/11976
# ===============================================================================================================

from imputegap.wrapper.AlgoPython.TimesNet.data_provider.data_factory import data_provider
from imputegap.wrapper.AlgoPython.TimesNet.exp.exp_basic import Exp_Basic
from imputegap.wrapper.AlgoPython.TimesNet.utils.tools import EarlyStopping, adjust_learning_rate, visual
from imputegap.wrapper.AlgoPython.TimesNet.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import imputegap.tools.utils as utils_imp

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, normalization=False, scaling=False, shuffle=False, reconstruction=False, verbose=True, tr_ratio=0.7, replicat=False):
        data_set, data_loader, l = data_provider(self.args, flag, normalization=normalization, scaling=scaling, shuffle=shuffle, reconstruction=reconstruction, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        return data_set, data_loader, l

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, normalization=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask[imputegap_mask == 0] = 1
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask, normalization)

                if torch.all(imputegap_mask == 0):
                    print("WARNING !")

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach()
                true = batch_x.detach()
                mask = mask.detach()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, verbose=True, deep_verbose=False, normalization=False, scaling=False, shuffle=False, tr_ratio=0.7, replicat=False):
        train_data, train_loader, len_tr = self._get_data(flag='train', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        vali_data, vali_loader, len_va = self._get_data(flag='val', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        test_data, test_loader, len_ts = self._get_data(flag='test', normalization=normalization, scaling=scaling, shuffle=shuffle, verbose=verbose, tr_ratio=tr_ratio, replicat=replicat)
        recon_data, recon_loader, len_recon = self._get_data(flag='train', scaling=scaling, shuffle=shuffle, reconstruction=True, verbose=verbose, replicat=replicat)

        if verbose:
            print(f"\nresults of the pre-processing :\n\t{len(train_loader) = } - {len_tr = }\n\t{len(vali_loader) = } - {len_va = }\n\t{len(test_loader) = } - {len_ts = }\n\t{len(recon_loader) = } - {len_recon = }\n")

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
                print(f"\n\n---timesnet---epoch:{epoch + 1}/{self.args.train_epochs}---imputegap------------------------------------------------------------------------------------------------")

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in enumerate(train_loader):

                v = ((i + 1) % 250 == 0)  # display

                if verbose and v:
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

                if verbose and v:
                    print(f"\tbatch NaNs: {torch.isnan(batch_x).any()} | imp NaNs: {torch.isnan(inp).any()} | imp NaNs: {torch.isnan(outputs).any()} | mask with 0: {(imputegap_mask == 0).any()}")

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if v:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if verbose:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, normalization)
            test_loss = self.vali(test_data, test_loader, criterion, normalization)

            if verbose:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(checkpoint, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, imputegap_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask[imputegap_mask == 0] = 0

                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask, normalization)
                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS 
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0 and deep_verbose:
                    folder_path = './test_results/' + setting + '/'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        if verbose:
            print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        if verbose:
            print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        if original_shape is not None:
            recons = utils_imp.reconstruction_window_based(preds=preds, nbr_timestamps=original_shape[0], verbose=verbose, deep_verbose=False)
            return recons

        return None
