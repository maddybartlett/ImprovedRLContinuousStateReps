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

args = parser.parse_args()

data_dir = args.data_path
save_dir = args.save_path

## Create empty list for storing data
allData=[]
## Counter
i=0

## For each file in the data folder
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    ## Load the data as a numpy array
    arr=np.load(filepath, allow_pickle=True)
    
    vals=[]
    ## In the first loop, get the column headers
    if i==0:
        header = arr.files
        df = pd.DataFrame(header)
    
    ## Every loop, get the values
    for item in arr.files:
        vals.append(arr[item])
    
    ## Add values to list
    allData.append(vals)
    ## increase counter
    i+=1
    
## Create pandas data frame from headers and list of data
df = pd.DataFrame(allData, columns=header)

## Save as csv
df.to_csv(save_dir)