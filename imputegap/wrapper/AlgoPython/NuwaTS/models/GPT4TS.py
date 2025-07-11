# ===============================================================================================================
# SOURCE: https://github.com/Chengyui/NuwaTS/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/pdf/2405.15317
# ===============================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from imputegap.wrapper.AlgoPython.NuwaTS.layers.Embed import DataEmbedding


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed,
                                           configs.freq,
                                           configs.dropout)

        config = GPT2Config.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True, attn_implementation="eager" )

        self.gpt2 = self.gpt2 = GPT2Model.from_pretrained('gpt2', config=config)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.predict_linear = nn.Linear(self.patch_size, configs.enc_in)
            self.ln = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out)
        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True)
        if self.task_name == 'anomaly_detection':
            self.ln_proj = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(
                configs.d_ff,
                configs.c_out,
                bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(0.1)
            self.ln_proj = nn.LayerNorm(configs.d_model * self.patch_num)
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape

        # Compute normalization counts
        nor = torch.sum(mask == 1, dim=1)  # shape: [B, D]
        nor = torch.clamp(nor, min=1.0)  # avoid divide-by-zero, but preserve actual counts as much as possible

        # Compute masked mean
        means = torch.sum(x_enc, dim=1) / nor
        means = means.unsqueeze(1).detach()

        # Center data
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)

        # Compute standard deviation over valid values
        var = torch.sum(x_enc * x_enc, dim=1) / nor
        stdev = torch.sqrt(torch.clamp(var, min=1e-5))  # protect against sqrt(0)
        stdev = stdev.unsqueeze(1).detach()

        # Normalize
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        #dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        #dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        T = dec_out.shape[1]  # actual time length of model output
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, T, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, T, -1)

        return dec_out
