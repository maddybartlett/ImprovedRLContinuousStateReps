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

data_dir_ = '..\\data\\ratbox50_discrete10_x10'

seeds = random.sample(range(1, 100), 10)

for seed in tqdm(seeds):
    results = ac.run(trials=trials,
                     steps=200,
                     env='RatBox-blocks-v1',
                     steering='compass',
                     gifs=False,
                     rep="OneHot",
                     rep_ranges=(10,10,10),
                     rule="TD0iG",
                     eps=0.439516,
                     dynamic_epsilon=True,
                     lr=0.300605, 
                     act_dis=0.947206,
                     state_dis=0.966626,
                     learnTrials=learnTrials,
                     specify_encoder_samples=False,
                     verbose=False,
                     data_dir=data_dir_,
                     data_format='npz',
                     seed=seed)