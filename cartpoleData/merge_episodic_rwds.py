import matplotlib.pyplot as plt
import pandas as pd
import sys,os

data_dir = os.path.join(os.path.dirname(__file__),'raw')
mddf = pd.read_csv(os.path.join(os.path.dirname(__file__),'processed/metadata-summary.csv'),index_col = 0)

print(mddf.sample(20))

# empty dictionary to hold trialIDs
trialIDs = {}

# get trial IDs associated with Hex, Discrete-3, d7, d11, d15, d19
for rep in ['PlaceSSP','Discrete']:
    
    # it's called PlaceSSP in the data but it's really just a population of random Hex
    if rep == 'PlaceSSP':
        IDs = mddf[ mddf['rep_'] == rep ]['trial_ID'].tolist()
        trialIDs['HexSSP'] = IDs
    
    elif rep == 'Discrete':
        subdf = mddf[ mddf['rep_'] == rep ]
        for n_bins in subdf['n_bins'].unique():
            IDs = subdf[subdf['n_bins'] == n_bins]['trial_ID'].tolist()
            trialIDs['{}_{:02d}'.format(rep,n_bins)] = IDs

out_df = pd.DataFrame( index = range(1000) )
for rep,IDs in trialIDs.items():
    for seed,ID in enumerate(IDs):
        
        rdf = pd.read_csv(os.path.join('raw','data',ID,'rewards.csv'),index_col=0)
        rtotal = rdf.sum(axis=0).reset_index(drop=True)
        
        out_df['{}-{}'.format(rep,seed)] = rtotal
out_df.to_csv(os.path.join('processed','all-episodic-rewards.csv'))

print(out_df.head())
print(out_df.tail())