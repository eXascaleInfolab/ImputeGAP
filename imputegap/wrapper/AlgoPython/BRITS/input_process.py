# ===============================================================================================================
# SOURCE: https://github.com/caow13/BRITS
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://papers.nips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
# ===============================================================================================================



# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import ujson as json

# ensure output dir exists
os.makedirs('./json', exist_ok=True)
fs = open('./json/json.json', 'w')  # single open

patient_ids = []
for filename in os.listdir('./raw'):
    # the patient data in PhysioNet contains 6-digits
    match = re.search(r'\d{6}', filename)
    if match:
        id_ = match.group()
        patient_ids.append(id_)

out = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 35 attributes which contain enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

# mean and std of 35 attributes
mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
                 147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
                 141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
                 23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
                 119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
                 119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
                 191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])

std = np.array(
    [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
     7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
     1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
     0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
     3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
     9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
     133.96778334724377])

def to_time_bin(x):
    h, m = map(int, x.split(':'))
    return h

def parse_data(x):
    # build Parameter->Value dict
    x = x.set_index('Parameter')['Value'].to_dict()
    values = []
    for attr in attributes:
        values.append(x[attr] if attr in x else np.nan)   # <- no has_key
    return values

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]
    deltas = []
    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35))
        else:
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1])
    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)
    # only used in GRU-D
    forwards = (
        pd.DataFrame(values)
          .fillna(method='ffill')
          .fillna(0.0)
          .to_numpy()                              # <- was .as_matrix()
    )
    rec = {
        'values': np.nan_to_num(values).tolist(),
        'masks': masks.astype('int32').tolist(),
        'evals': np.nan_to_num(evals).tolist(),    # imputation ground-truth
        'eval_masks': eval_masks.astype('int32').tolist(),
        'forwards': forwards.tolist(),
        'deltas': deltas.tolist(),
    }
    return rec

def parse_id(id_):
    data = pd.read_csv(f'./raw/{id_}.txt')
    # accumulate the records within one hour
    data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))

    evals = []
    # merge all the metrics within one hour
    for h in range(48):
        evals.append(parse_data(data[data['Time'] == h]))

    evals = (np.array(evals) - mean) / std

    shp = evals.shape
    flat = evals.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    rng = np.random.default_rng()
    indices = np.where(~np.isnan(flat))[0]
    if indices.size:
        indices = rng.choice(indices, indices.size // 10, replace=False)
    values = flat.copy()
    if indices.size:
        values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(flat))

    evals = flat.reshape(shp)
    values = values.reshape(shp)
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    pid = int(id_)
    if pid not in out.index:
        raise KeyError(f"RecordID {pid} not found in Outcomes-a.txt")
    label = out.loc[pid]

    rec = {'label': int(label)}
    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    fs.write(json.dumps(rec) + '\n')

print(f"Found {len(patient_ids)} patient files in ./raw")
for id_ in patient_ids:
    print(f'Processing patient {id_}')
    try:
        parse_id(id_)
    except Exception as e:
        print(e)
        continue

fs.close()
