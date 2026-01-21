# ===============================================================================================================
# SOURCE: https://github.com/wangliang-cs/hkmf-t?tab=readme-ov-file
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://ieeexplore.ieee.org/document/8979178
# ===============================================================================================================

# Copyright (c) [2021] [wlicsnju]
# [HKMF-T] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2. 
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2 
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
# See the Mulan PSL v2 for more details.  
import datetime
import logging
import os
import fire
import zipfile

import pandas as pd


BSD_inner_file = 'day.csv'
MVCD_inner_file = 'Motor_Vehicle_Crashes.csv'
EPCD_inner_file = 'household_power_consumption.txt'

MVCD_state = {
    'Western_New_York': {'NIAGARA', 'ERIE', 'CHAUTAUQUA', 'CATTARAUGUS', 'ALLEGANY', },
    'Finger_Lakes': {'ORLEANS', 'GENESEE', 'WYOMING', 'MONROE', 'LIVINGSTON', 'WAYNE', 'ONTARIO', 'YATES', 'SENECA', },
    'Southern_Tier': {'STEUBEN', 'SCHUYLER', 'CHEMUNG', 'TOMPKINS', 'TIOGA', 'CHENANGO', 'BROOME', 'DELAWARE', },
    'Central_New_York': {'CORTLAND', 'CAYUGA', 'ONONDAGA', 'OSWEGO', 'MADISON', },
    'North_Country': {'ST. LAWRENCE', 'LEWIS', 'JEFFERSON', 'HAMILTON', 'ESSEX', 'CLINTON', 'FRANKLIN', },
    'Mohawk_Vally': {'ONEIDA', 'HERKIMER', 'FULTON', 'MONTGOMERY', 'OTSEGO', 'SCHOHARIE', },
    'Capital_District': {'ALBANY', 'COLUMBIA', 'GREENE', 'WARREN', 'WASHINGTON', 'SARATOGA', 'SCHENECTADY', 'RENSSELAER', },
    'Hudson_Valley': {'SULLIVAN', 'ULSTER', 'DUTCHESS', 'ORANGE', 'PUTNAM', 'ROCKLAND', 'WESTCHESTER', },
    'New_York_City': {'NEW YORK', 'BRONX', 'QUEENS', 'KINGS', 'RICHMOND', },
    'Long_Island': {'NASSAU', 'SUFFOLK', },
}


def bsd_generate(raw_file: str, out_dataset: str):
    data_name = 'cnt'
    tag_name = 'weathersit'
    if not os.path.exists(raw_file):
        logging.warning(f'BSD raw file {raw_file} do not exists, skip.')
    with zipfile.ZipFile(raw_file, 'r', zipfile.ZIP_DEFLATED) as zf:
        fp = zf.open(BSD_inner_file)
        df = pd.read_csv(fp, index_col=0)
        ds_df = df[['dteday', data_name, tag_name]]
        ds_df = ds_df.rename(columns={data_name: 'data_0',
                                      tag_name: 'tag_0'})
        ds_df.set_index('dteday', inplace=True)
        ds_df.index.name = 'index'
        ds_df.sort_index()
        ds_df.to_csv(out_dataset, index=True)
        logging.info(f'succ save to {out_dataset}')


def mvcd_generate(raw_file: str, out_dataset_template: str):
    tag_name = 'Weather Conditions'
    if not os.path.exists(raw_file):
        logging.warning(f'MVCD raw file {raw_file} do not exists, skip.')
    with zipfile.ZipFile(raw_file, 'r', zipfile.ZIP_DEFLATED) as zf:
        fp = zf.open(MVCD_inner_file)
        df = pd.read_csv(fp)
        df['Date'] = df['Date'].apply(lambda x: datetime.datetime(int(x.split('/')[2]),
                                                                  int(x.split('/')[0]),
                                                                  int(x.split('/')[1])))
        sds_df = df[['Date']].groupby('Date').size().reset_index(name='data_0')
        sds_df.set_index('Date', inplace=True)
        sds_df['tag_0'] = df[['Date', tag_name]].groupby('Date').agg(lambda x: x.value_counts().index[0])
        sds_df.index.name = 'index'
        sds_df.sort_index()
        sds_df.replace('Sleet/Hail/Freezing Rain', 'Snow', inplace=True)
        sds_df.to_csv(out_dataset_template, index=True)
        logging.info(f'Uni-d succ save to {out_dataset_template}')


def epcd_generate(raw_file: str, out_dataset: str):
    data_names = ['Global_active_power']
    tag_name = 'Date'
    if not os.path.exists(raw_file):
        logging.warning(f'EPCD raw file {raw_file} do not exists, skip.')
    with zipfile.ZipFile(raw_file, 'r', zipfile.ZIP_DEFLATED) as zf:
        fp = zf.open(EPCD_inner_file)
        df = pd.read_csv(fp, sep=';')
        for n in data_names:
            df = df[df[n] != '?']
            df[n] = df[n].astype(float)
        df[tag_name] = df[tag_name].apply(lambda x: datetime.datetime(int(x.split('/')[2]),
                                                                      int(x.split('/')[1]),
                                                                      int(x.split('/')[0])))
        if len(data_names) != 1:
            df['data_0'] = df[data_names].sum(axis=1)
        else:
            df['data_0'] = df[data_names[0]]
        ds_df = df[[tag_name, 'data_0']]
        ds_df = ds_df.groupby(tag_name).sum()
        ds_df.reset_index(inplace=True)
        ds_df['tag_0'] = ds_df[tag_name].apply(lambda x: x.weekday())
        ds_df.set_index(tag_name, inplace=True)
        ds_df.index.name = 'index'
        ds_df.sort_index()
        ds_df.to_csv(out_dataset, index=True)
        logging.info(f'succ save to {out_dataset}')


def main(bsd_raw_file: str = 'raw_data/Bike-Sharing-Dataset.zip',
         mvcd_raw_file: str = 'raw_data/Motor_Vehicle_Crashes.zip',
         epcd_raw_file: str = 'raw_data/household_power_consumption.zip',
         bsd_out_dataset: str = 'dataset/BSD.csv',
         mvcd_out_dataset: str = 'dataset/MVCD.csv',
         epcd_out_dataset: str = 'dataset/EPCD.csv',
         ):
    bsd_generate(bsd_raw_file, bsd_out_dataset)
    mvcd_generate(mvcd_raw_file, mvcd_out_dataset)
    epcd_generate(epcd_raw_file, epcd_out_dataset)


if __name__ == '__main__':
    fire.Fire(main)
