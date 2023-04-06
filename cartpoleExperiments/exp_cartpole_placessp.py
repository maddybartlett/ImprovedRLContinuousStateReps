import numpy as np
import sys,os
import time
import nni

sys.path.append(os.path.join(os.path.dirname(__file__),'../trials'))
from trial_cartpole import ACTrial

# specify where to save the data locally
data_dir_ = '../../data-ks/iccm23-git'

trials = 1000
learnTrials = trials
ac = ACTrial()

def main(args):
    out = ac.run(trials = trials,
            steps = 500,
            env = 'CartPole-v1',
            n_done = 1,
            
            rep_ = 'PlaceSSP',
            specify_encoder_samples = False,
            
            # hyperparameters for NNI optimization
            length_scale = args['length_scale'],
            n_rotates = args['n_rotates'],
            state_neurons = args['n_neurons'],
            active_prop = args['active_prop'],
            lr = args['lr'],
            eps = args['epsilon'],

            # specified hyperparameters & options
            rule = "TD0",
            dynamic_epsilon = True,
            act_dis = 0.9,
            state_dis = 0.99,
            learnTrials = learnTrials,            
            
            verbose = False,
            data_dir = data_dir_,
        )

    score = out['terminal_reward']

    nni.report_final_result( score )

if __name__ == '__main__':
    params = nni.get_next_parameter()