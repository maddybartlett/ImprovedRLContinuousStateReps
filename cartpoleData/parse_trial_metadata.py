import pandas as pd
import numpy as np
import sys,os

data_dir = os.path.join(os.path.dirname(__file__),'raw')
files = [ f for f in os.listdir(data_dir) if '.txt' in f ]

# create an empty dataframe to hold the metadata
col_labels = []
with open(os.path.join(data_dir,files[0])) as MetadataFile:
    for line in MetadataFile.readlines():
        line_info = line.split(' ')
        if len(line_info) == 3:           
            col_labels.append(line_info[0])
out_df = pd.DataFrame( columns = col_labels )

# read in each file and save as a row in the dataframe
float_cols = ['env_dt','length_scale','eps','lr','act_dis','state_dis','active_prop','terminal_reward','build_time','total_time','avg_trial_time']
int_cols = ['seed','trials','n_done','n_reset','n_bins','n_rotates','learnTrials','state_neurons','dimensionality','episodes_to_learn']
for file in files:
    temp_data = { }
    with open(os.path.join(data_dir,file)) as MetadataFile:
        for line in MetadataFile.readlines():
            line_info = line.split(' ')
            if len(line_info) == 3:
                col = line_info[0]
                if col in float_cols:
                    try:
                        value = float(line_info[-1].strip('\n'))
                    except:
                        value = np.nan
                elif col in int_cols:
                    try:
                        value = int(line_info[-1].strip('\n'))
                    except:
                        value = np.nan
                else:
                    value = line_info[-1].strip("'\n'")
                temp_data[col] = value
    temp_df = pd.DataFrame( data = temp_data, index = [0] )
    out_df = pd.concat( [out_df,temp_df], axis = 0, ignore_index = True )
    
# save the summary data
out_dir = 'processed'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_df.to_csv(os.path.join(out_dir,'metadata-summary.csv'))
