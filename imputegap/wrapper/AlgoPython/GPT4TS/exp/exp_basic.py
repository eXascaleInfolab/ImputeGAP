# ===============================================================================================================
# SOURCE: google.com/url?q=https://github.com/DAMO-DI-ML/One_Fits_All&sa=D&source=editors&ust=1763995357004903&usg=AOvVaw33kubL9FDsXc_ZL-M_onqA
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://dl.acm.org/doi/10.5555/3666122.3667999
# ===============================================================================================================

import os
import torch
from imputegap.wrapper.AlgoPython.GPT4TS.models import GPT4TS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            # 'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            # 'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            'GPT4TS': GPT4TS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
