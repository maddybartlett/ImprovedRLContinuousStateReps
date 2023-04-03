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

data_dir_ = '..\\data\\ratbox100_discrete8_x10'

seeds = random.sample(range(1, 100), 10)

for seed in tqdm(seeds):
    results = ac.run(trials=trials,
                     steps=200,
                     env='RatBox-blocks-v0',
                     steering='compass',
                     gifs=False,
                     rep="OneHot",
                     rep_ranges=(8,8,8),
                     rule="TD0n",
                     eps=0.580777,
                     dynamic_epsilon=True,
                     lr=0.247154, 
                     act_dis=0.949027,
                     state_dis=0.841157,
                     learnTrials=learnTrials,
                     specify_encoder_samples=False,
                     verbose=False,
                     data_dir=data_dir_,
                     data_format='npz',
                     seed=seed)