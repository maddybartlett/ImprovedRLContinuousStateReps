from trial_ratbox import ACTrial

import sys
sys.path.insert(0, '../network')

import rlnet as net
import numpy as np
import nni

trials=500
learnTrials=trials+1
ac = ACTrial()

def main(args):
    results = ac.run(trials=trials,
                 steps=200,
                 env='RatBox-blocks-v0',
                 steering='compass',
                 gifs=False,
                 rep="OneHot", 
                 rep_ranges=(6,6,6), 
                 rule="TD0n",
                 eps=args['epsilon'],
                 dynamic_epsilon=True,
                 lr=args['lr'], 
                 act_dis=args['act_dis'],
                 state_dis=args['state_dis'],
                 learnTrials=learnTrials,
                 specify_encoder_samples=False,
                 verbose=False,
                 data_dir='..\\data\\nn_rawtxt\\nni_ratbox100_discrete6')


    score = [val for index,val in enumerate(results['roll_mean'][0][-1:])]
    nni.report_final_result(score[0])
                                       
if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)