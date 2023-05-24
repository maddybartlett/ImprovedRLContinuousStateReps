import matplotlib.pyplot as plt
import numpy as np
import pyperclip
import nengo
import sys,os

sys.path.insert(1,'../network')
from trial_cartpole import ACTrial
import rlnet as net

trials = 1000
learnTrials = trials
ac = ACTrial()

data_dir_ = '../../all-experiments/cartpole/data-raw/cartpole-debug'

for i in range(10):

    # pre_comment_ = input('Enter notes about change made for this trial: ')
    pre_comment_ = 'run {}'.format(i)

    # return parameters and results
    metadata = ac.run(seed = i,
                trials = trials,
                
                ### environment parameters ###
                steps = 500,
                env = 'CartPole-v1',
                n_done = 1,
                gifs = False,
                
                ### model-specific parameters ###
                
                # Uncomment  next 4 lines for Discrete rep, 3 bins
                # rep_ = 'Discrete',
                # n_bins = 3,
                # eps = 0.556489,                              # from NNI
                # lr = 0.208818,                               # from NNI
                
                # Uncomment next 4 lines for Discrete rep, 7 bins
                # rep_ = 'Discrete',
                # n_bins = 7,
                # eps = 0.307233,                              # from NNI
                # lr = 0.170501,                               # from NNI

                # Uncomment next 4 lines for Discrete rep, 11 bins
                # rep_ = 'Discrete',
                # n_bins = 11,
                # eps = 0.448556,                              # from NNI
                # lr = 0.085822,                               # from NNI                

                # Uncomment next 4 lines for Discrete rep, 15 bins
                # rep_ = 'Discrete',
                # n_bins = 15,
                # eps = 0.293466,                              # from NNI
                # lr = 0.264277,                               # from NNI   

                # Uncomment next 4 lines for Discrete rep, 19 bins
                rep_ = 'Discrete',
                n_bins = 19,
                eps = 0.259453,                              # from NNI
                lr = 0.283112,                               # from NNI   
                
                # Uncomment next 9 lines for HexSSP rep
                # rep_ = 'PlaceSSP',                             # todo -- fix this in the trial file
                # specify_encoder_samples = False,
                # neuron_type = nengo.RectifiedLinear(),
                # length_scale = 0.527579,                       # from NNI
                # n_rotates = 7,                                 # from NNI
                # state_neurons = 4096,                          # from NNI
                # active_prop = 0.242728,                        # from NNI
                # eps = 0.221581,                                # from NNI
                # lr = 0.195843,                                 # from NNI
                
                ###
                
                ### common model parameters
                rule = "TD0",               # kinda lambda
                dynamic_epsilon = True,     # epsilon decays if performance is improving
                act_dis = 0.9,              # beta
                state_dis = 0.99,           # gamma
                learnTrials = learnTrials,
                ###

                ### data saving specifications
                verbose = False,
                data_dir = data_dir_,
                pre_comment = pre_comment_,
                ###
            )
            
    pyperclip.copy(metadata['trial_ID'])
    print(metadata['trial_ID'])
    print('episodes to learn: ', metadata['episodes_to_learn'])
    print('terminal reward, learning: ', metadata['terminal_reward_learning'])
    print('terminal reward: ', metadata['terminal_reward'])
    print('dimensionality: ', metadata['dimensionality'])

    # post_comment = input('Enter observations for this trial: ')
    post_comment = 'run {}'.format(i+1)
    with open(os.path.join(data_dir_,'{}.txt'.format(metadata['trial_ID'])),'a') as data_file:
        data_file.write('post_comment = ' + post_comment)
    
    