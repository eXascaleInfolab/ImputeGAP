# ===============================================================================================================
# SOURCE: https://github.com/jsyoon0823/GAIN
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
# ===============================================================================================================


# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from imputegap.wrapper.AlgoPython.GAIN.data_loader import data_loader
from imputegap.wrapper.AlgoPython.GAIN.gain import gain
from imputegap.wrapper.AlgoPython.GAIN.utils import rmse_loss


def handle_parser(argv=None):
  import argparse

  parser = argparse.ArgumentParser(description='GAIN')

  parser.add_argument('--data_name', choices=['letter', 'spam'], default='spam', type=str)
  parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
  parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
  parser.add_argument('--hint_rate', help='hint probability', default=0.9, type=float)
  parser.add_argument('--alpha', help='hyperparameter', default=100, type=float)
  parser.add_argument('--iterations', help='number of training interations', default=10000, type=int)

  args, _unknown = parser.parse_known_args(argv)

  return args


def recovGAIN (ts_m, batch_size=128, iterations=10000, alpha=100, hint_rate=0.9, tr_ratio=0.8, verbose=False, replicat=False):
  '''Main function for UCI letter and spam datasets.

  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''

  recov = np.copy(ts_m)
  m_mask = np.isnan(ts_m)
  input_data = np.copy(ts_m)

  args = handle_parser()

  if verbose:
    print(f"(IMPUTATION) GAIN\n\tMatrix: {input_data.shape[0]}, {input_data.shape[1]}\n\tbatch_size: {batch_size}\n\tepochs: {iterations}\n\talpha: {alpha}\n\thint_rate: {hint_rate}\n\ttr_ratio: {tr_ratio}\n")
    print(f"call: gain.impute(params={{'batch_size': {batch_size}, 'epochs': {iterations}, 'alpha': {alpha}, 'hint_rate': {hint_rate}}})\n")

    if batch_size > input_data.shape[0]:
      old_batch_size = batch_size
      batch_size = input_data.shape[0]//2
      print(f"\n(ERROR): {old_batch_size} > {input_data.shape[0]}, in order to train the model, reduction of the batch_size: {batch_size}.")

  inc = 1
  tries, max_tries = 0, 10
  while np.any(recov) and tries < max_tries:

    if iterations < 50:
      break

    iterations = iterations // inc

    miss_rate = 1-tr_ratio

    gain_parameters = {'batch_size': batch_size, 'hint_rate': hint_rate, 'alpha': alpha, 'iterations': iterations}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name="data_name", miss_rate=miss_rate, ts=input_data, inject_nan=False)

    # Impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters, verbose=verbose)

    if replicat:
      model_rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

    recov[m_mask] = imputed_data_x[m_mask]

    inc = inc+1

    if not np.isnan(imputed_data_x).any():
        break

  if replicat:
    print(f"\n\tmodel output RMSE: {model_rmse}\n")

  return recov
