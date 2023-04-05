import sys
sys.path.insert(0, '..\network')
from tqdm import tqdm

## For running trial
from trial_ratbox import ACTrial
import rlnet as net
from rlnet.utils import get_ac_output, plot_policy

## For plotting
import matplotlib.pyplot as plt
import numpy as np
import random

trials = 500
learnTrials = trials+1
ac = ACTrial()

data_dir_ = '..\\data\\ratbox50_hexssp_x10'
seeds = random.sample(range(1, 100), 10)

for seed in tqdm(seeds):
    results = ac.run(trials=trials,
                     steps=200,
                     env='RatBox-blocks-v1',
                     steering='compass',
                     gifs=False,
                     rep="HexSSP", 
                     n_rotates=7,
                     n_scales=6,
                     length_scale=91.775829,
                     rule="TD0iG",
                     eps=0.360414,
                     dynamic_epsilon=True,
                     lr=0.206206, 
                     act_dis=0.820566,
                     state_dis=0.857552,
                     learnTrials=learnTrials,
                     specify_encoder_samples=True,
                     active_prop=0.156189,
                     verbose=False,
                     data_dir=data_dir_,
                     data_format='npz',
                     seed=seed)
    
    