import os
import numpy as np
import pandas as pd
from tqdm import tqdm


## EXTRACT DATA FROM TXT FILES ## 
def txt_to_df(folder, header, idxs):
    allVals = []

    for filename in tqdm(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        with open(filepath) as f:
            lines = f.readlines()

        vals = [e.split(' =')[1] for i,e in enumerate(lines) if i in idxs]

        ## Change the episodes from strings to lists of arrays
        str_val = vals[-3].split('[')[1]
        str_val = str_val.split(']')[0]
        str_list = np.fromstring(str_val, dtype=float, sep=',')
        vals[-3] = str_list

        ## rewards
        str_val = vals[-2][2:-1].split('], [')
        str_list = [np.fromstring(x, dtype=float, sep=',') for x in str_val]
        vals[-2] = str_list[1:]

        ## values
        str_val = vals[-1][2:-1].split(']), array([')
        str_val[0] = str_val[0].split('[array([')[1]
        str_val[-1] = str_val[-1].split('])]')[0]
        str_list = [np.fromstring(x, dtype=float, sep=',') for x in str_val]
        vals[-1] = str_list[1:]

        allVals.append(vals)
    
    df = pd.DataFrame(allVals, columns=header)
    
    ## Remove '\n' from all cells
    df = df.replace(r'\n', '', regex=True) 

    ## Change data types from strings to ints or floats
    df = df.astype({'seed': 'int', 'trials': 'int', 'steps': 'int', 
                    'env_dt': 'float', 'n_rotates': 'int', 'n_scales': 'int',
                    'length_scale': 'float', 'eps': 'float', 'lr': 'float',
                    'act_dis': 'float', 'state_dis': 'float',
                    'active_prop': 'float', 'ssp_dim': 'int',}, errors='ignore')
            
    return df

## CALCULATE THE ROLLING MEAN REWARD AND ADD IT TO DATAFRAME AS A COLUMN ##
def add_roll_mean(df):
    #from dataframe, get the rewards for each trial
    rewards_over_eps = df['episodes']
    
    #create empty lists for recording new values
    Roll_mean=[]
    lst=[]

    #for each trial
    for i in range(len(rewards_over_eps)):
        #get rewards for trial i
        x = rewards_over_eps[i]
        
        #convert array to dataframe
        array_df = pd.DataFrame(x)

        #calculate rolling mean and add to one of the lists
        Roll_mean.append(array_df[array_df.columns[0]].rolling(100).mean())
        
        #copy the rolling mean data and convert to an array
        a = np.asarray(Roll_mean[i]).copy()
        #add the new array to the second list
        lst.append(a)
    
    #create new column with rolling mean data
    df['roll_mean'] = lst
    
    #return dataframe
    return df

## CALCULATE THE NUMBER OF TRIALS TO REACH THE GOAL ROLLING MEAN ##
def trials_to_goal(df, goal):
    ## create list for storing trial indices
    goal_reached = []
    
    ## for each value in roll_mean, record the index of values > goal
    for i in range(len(df)):
        a=[index for index,val in enumerate(df['roll_mean'][i]) if val > goal]
        
        ## if experiment never reached goal, put max number of runs in list
        if len(a) < 1:
            goal_reached.append(100000)
        ## otherwise, add the index to the list for plotting
        else:
            goal_reached.append(a[0])
            
    df['goal_reached'] = goal_reached
        
    ## return list 
    return df

## RETRIEVE THE BEST PERFORMING PARAMETER SETS ##
def get_best(df, params_list):
    ## Set a threshold for getting the quickest experiments
    top_5per = int(len(df)*0.05)
    
    ## Generate new dataframe containing only the experiments that achieved the goal rolling mean 
    ## quicker than the threshold number of trials
    best = df.nsmallest(top_5per, ['goal_reached'])
    best = best.reset_index()
        
    ## return new data frame
    return best

## CONVERT STRING DATA IN EPISODES, REWARDS AND VALUES COLUMNS TO ARRAYS ##
def str_to_arr(df, col_names=[None]):
    for i in range(len(df)):
        if 'episodes' in col_names:
            str_array = df['episodes'][i].replace("\n", "")
            str_array = str_array.replace("[", "")
            str_array = str_array.replace("]", "")
            df['episodes'][i] = np.fromstring(str_array, sep=' ')

        if 'rewards' in col_names:
            str_arr = df['rewards'][i].replace("\n", "")
            str_arr = str_arr.replace("[array(", "")
            str_arr = str_arr.replace("), array(", "")
            str_arr = str_arr[1:-3]

            rew_list = []
            for i in range(len(str_arr.split(']['))):
                rew_list.append(np.fromstring(str_arr.split('][')[0], sep=','))

            df['rewards'][i] = rew_list

        if 'values' in col_names:
            str_arr = df['values'][0].replace("\n", "")
            str_arr = str_arr.replace("[array(", "")
            str_arr = str_arr.replace("), array(", "")
            str_arr= str_arr[1:-3]

            val_list = []
            for i in range(len(str_arr.split(']['))):
                val_list.append(np.fromstring(str_arr.split('][')[0], sep=','))

            df['values'][i] = val_list
       
    return df