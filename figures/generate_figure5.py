import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os

from scipy import stats

# visualization scheme
discr_colors = {
        3:  '#EAEF98',
        7:  '#54D078',
        11: '#1E64B1',
        15: '#6E0092',
        19: '#730002',
}

plt.style.use('plot_style.mplstyle')
figsize_ = (8,4.3)

fig,(ax1,ax2) = plt.subplots(1,2,figsize = figsize_)

### plot the mean episodic reward across the 10 seeds
df = pd.read_csv(os.path.join('../cartpoleData/processed','all-episodic-rewards.csv'),index_col=0)
for rep in sorted( set([c.split('-')[0] for c in df.columns]) ):
    print(rep)
    rep_cols = [ col for col in df.columns if rep in col ]
    print(rep,len(rep_cols))
    if 'HexSSP' in rep:
        label_ = 'HexSSP'
        color_ = 'k'
    elif 'Discrete' in rep:
        rep_info = rep.split('_')
        n_bins = int(rep_info[-1])
        label_ = '{} bins'.format(n_bins)
        color_ = discr_colors[n_bins]
    
    roll_mean = df[rep_cols].rolling(100,min_periods=0).mean().mean(axis=1)
    roll_sem = df[rep_cols].rolling(100,min_periods=0).mean().sem(axis=1)
    ax1.plot(df.index,roll_mean,label = label_,color = color_,linewidth=2.5)
    ax1.fill_between(df.index,roll_mean-roll_sem,roll_mean+roll_sem,color = color_,alpha=.3)
    
ax1.axhline(500,color='k',linestyle='--')
ax1.xaxis.set_ticks_position('none')
ax1.yaxis.set_ticks_position('none')
ax1.set_ylim(10,550)
ax1.set_xlim(0,1000)
ax1.set_xlabel(r'Epsiodes $\to$')
ax1.set_ylabel('Rolling Mean Reward')
ax1.legend(loc = 'upper center',bbox_to_anchor=(0.5,1.25),ncol=3,frameon=False)
#ax1.grid()

### plot spread of terminal rewards observed within the 10 seeds
expt_name = 'cartpole-repeats'
data_dir = os.path.join(os.path.dirname(__file__),'../cartpoleData/processed')
mddf = pd.read_csv(os.path.join(data_dir,'metadata-summary.csv'.format(expt_name)),index_col = 0)

# plot data for discrete representation
discr_mddf = mddf[mddf['rep_'] == 'Discrete']
discretizations = sorted(discr_mddf['n_bins'].unique())

vdf = pd.DataFrame( index = range(10) )
for discr in discretizations:
    rws = discr_mddf[discr_mddf['n_bins'] == discr]['terminal_reward'].reset_index(drop=True)
    vdf[discr] = rws
    
ax2.scatter(discretizations,vdf.mean(axis=0))
for discr in discretizations:
    ys = vdf[discr]
    ci95 = stats.t.interval(0.95, len(ys)-1, loc = np.mean(ys), scale = stats.sem(ys))
    print(discr,ci95)
    ax2.plot([discr,discr],[ci95[0],ci95[1]],color='tab:blue',linewidth=2.5)

# plot data for HexSSPs
trws = mddf[mddf['rep_'] == 'PlaceSSP']['terminal_reward']

ci95 = stats.t.interval(0.95, len(trws)-1, loc = np.mean(trws), scale = stats.sem(trws))
print('hex',ci95)
ax2.axhspan( ci95[0],ci95[1], color = 'dimgray' )
ax2.set_ylabel('Terminal Reward')
ax2.set_ylim(10,550)
ax2.xaxis.set_ticks_position('none')
ax2.yaxis.set_ticks_position('none')
ax2.set_xlabel('Discretization (# bins per state)')
ax2.set_xticks( [3,7,11,15,19] )

for ax in (ax1,ax2):
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig('figure5_cartpole_discrete_vs_hex.pdf', bbox_inches='tight')

plt.show()
