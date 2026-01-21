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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from imputegap.wrapper.AlgoPython.GAIN.utils import binary_sampler


def data_loader(data_name, miss_rate, ts=None, inject_nan=True):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if ts is None:
    if data_name in ['letter', 'spam']:
      file_name = 'data/'+data_name+'.csv'
      data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
      print(f"letter/spam {data_x.shape = }")
  else:
    data_x = ts

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  if inject_nan:
    mask = (~np.isnan(data_x)).astype(float)  # imputegap
    data_m = binary_sampler(1-miss_rate, no, dim).astype(float) # author
    data_m = data_m * mask # injection
  else:
    data_m = binary_sampler(1 - miss_rate, no, dim).astype(float)  # author

  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m
