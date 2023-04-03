import pandas as pd
import numpy as np
import pickle

from utils import txt_to_df

from tqdm import tqdm

import os
import argparse


parser = argparse.ArgumentParser()

## User must enter path to nni data directory 
parser.add_argument("data_path")

## User must enter path to destination directory 
parser.add_argument("save_path")

## Optional arguments
parser.add_argument('-t', '--type', 
                    choices=['pickle', 'csv'], 
                    default = 'csv') 

args = parser.parse_args()

data_dir = args.data_path
save_dir = args.save_path
save_type = args.type
    
    
## Indexes for data of interest    
IDXS = [0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,24,25,27,28,29,30,31]

## Get path to data folder
filepath = os.path.join(data_dir, os.listdir(data_dir)[0])

## Open file in read mode
data = open(filepath,"r+")

## Get each line of .txt
lines=data.readlines()

## Get column headers from .txt
header = [e.split(' =')[0] for i,e in enumerate(lines) if i in IDXS]

## Convert from .txt to pandas DataFrame
df = txt_to_df(data_dir, header, IDXS)


if save_type == "pickle":
    save_pkl=save_dir
    df.to_pickle(save_pkl)
elif save_type == "csv":
    save_csv=save_dir
    df.to_csv(save_csv, encoding='utf-8', index=False)


