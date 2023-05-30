### Visualise and Explore Data 

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import iqr
import seaborn as sns

# Load dataframes
data_location = '..//ratboxData//ratbox100_hexssp_x10.csv'
df_hex100 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox50_hexssp_x10.csv'
df_hex50 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox100_discrete6_x10.csv'
df_discrete6100 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox50_discrete6_x10.csv'
df_discrete650 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox100_discrete8_x10.csv'
df_discrete8100 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox50_discrete8_x10.csv'
df_discrete850 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox100_discrete10_x10.csv'
df_discrete10100 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox50_discrete10_x10.csv'
df_discrete1050 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox100_discrete12_x10.csv'
df_discrete12100 = pd.read_csv(data_location)

data_location = '..//ratboxData//ratbox50_discrete12_x10.csv'
df_discrete1250 = pd.read_csv(data_location)


# Convert relevant data from string to array inside dataframes
DATAFRAMES = [df_hex100, df_hex50, df_discrete6100, df_discrete650, df_discrete8100, 
             df_discrete850, df_discrete10100, df_discrete1050, df_discrete12100,
             df_discrete1250]

for df in DATAFRAMES:
    for i in range(10):
        str_array = df['roll_mean'][i].replace("]", "")
        str_array = str_array.replace("[", "")
        str_array = str_array.replace("\n", " ")
        df['roll_mean'][i] = np.fromstring(str_array, sep=' ')
        
        str_array = df['episodes'][i].replace('\n', "")
        str_array = str_array.replace('[', "")
        str_array = str_array.replace(']', "")
        df['episodes'][i] = np.fromstring(str_array, sep=' ')


# Set formatting values
axisLabelSize=16
titleSize=20
tickSize=14
legendtitleSize=14
legendSize=12

COLORS = ['#ffdcc0', '#eaef98', '#97e074', '#54d078', '#37c1b4', '#1e64b1',
          '#1e08a2', '#6e0092', '#820052', '#730002']

import scipy
### Compare Performance with HexSSP vs Discrete Representation
mean_rollmean_discrete6100 = np.vstack(df_discrete6100['roll_mean'])[:,99:].mean(axis=0)
sem_rollmean_discrete6100 = scipy.stats.sem( np.vstack(df_discrete6100['roll_mean'])[:,99:], axis=0, ddof=1)

mean_rollmean_discrete8100 = np.vstack(df_discrete8100['roll_mean'])[:,99:].mean(axis=0)
sem_rollmean_discrete8100 = scipy.stats.sem( np.vstack(df_discrete8100['roll_mean'])[:,99:], axis=0, ddof=1)

mean_rollmean_discrete10100 = np.vstack(df_discrete10100['roll_mean'])[:,99:].mean(axis=0)
sem_rollmean_discrete10100 = scipy.stats.sem( np.vstack(df_discrete10100['roll_mean'])[:,99:], axis=0, ddof=1)

mean_rollmean_discrete12100 = np.vstack(df_discrete12100['roll_mean'])[:,99:].mean(axis=0)
sem_rollmean_discrete12100 = scipy.stats.sem( np.vstack(df_discrete12100['roll_mean'])[:,99:], axis=0, ddof=1)

mean_rollmean_hex100 = np.vstack(df_hex100['roll_mean'])[:,99:].mean(axis=0)
sem_rollmean_hex100 = scipy.stats.sem( np.vstack(df_hex100['roll_mean'])[:,99:], axis=0, ddof=1)

figsize_ = (8,4.3)

fig,(ax1,ax2) = plt.subplots(1,2,figsize = figsize_)

ax1.plot(mean_rollmean_discrete6100, label='6 bins', color=COLORS[0], linewidth=2.5)
#ax1.fill_between(np.arange(0,401), idr_90_discrete6100, idr_10_discrete6100, alpha=0.3, color=COLORS[0])
ax1.fill_between(np.arange(0,401), mean_rollmean_discrete6100-sem_rollmean_discrete6100, mean_rollmean_discrete6100+sem_rollmean_discrete6100,alpha=0.3, color=COLORS[0])

ax1.plot(mean_rollmean_discrete8100, label='8 bins', color=COLORS[2], linewidth=2.5)
#ax1.fill_between(np.arange(0,401), idr_90_discrete8100, idr_10_discrete8100, alpha=0.3, color=COLORS[2])
ax1.fill_between(np.arange(0,401), mean_rollmean_discrete8100-sem_rollmean_discrete8100, mean_rollmean_discrete8100+sem_rollmean_discrete8100,alpha=0.3, color=COLORS[2])

ax1.plot(mean_rollmean_discrete10100, label='10 bins', color=COLORS[4], linewidth=2.5)
#ax1.fill_between(np.arange(0,401), idr_90_discrete10100, idr_10_discrete10100, alpha=0.3, color=COLORS[4])
ax1.fill_between(np.arange(0,401), mean_rollmean_discrete10100-sem_rollmean_discrete10100, mean_rollmean_discrete10100+sem_rollmean_discrete10100,alpha=0.3, color=COLORS[4])

ax1.plot(mean_rollmean_discrete12100, label='12 bins', color=COLORS[6], linewidth=2.5)
#ax1.fill_between(np.arange(0,401), idr_90_discrete12100, idr_10_discrete12100, alpha=0.3, color=COLORS[6])
ax1.fill_between(np.arange(0,401), mean_rollmean_discrete12100-sem_rollmean_discrete12100, mean_rollmean_discrete12100+sem_rollmean_discrete12100,alpha=0.3, color=COLORS[6])

ax1.plot(mean_rollmean_hex100, label='Hex SSP', color=COLORS[8], linewidth=2.5)
#ax1.fill_between(np.arange(0,401), idr_90_hex100, idr_10_hex100, alpha=0.3, color=COLORS[8])
ax1.fill_between(np.arange(0,401), mean_rollmean_hex100-sem_rollmean_hex100, mean_rollmean_hex100+sem_rollmean_hex100,alpha=0.3, color=COLORS[8])

ax1.set_ylabel('Rolling Mean Reward', fontsize=axisLabelSize)
ax1.set_xlabel(r'Episodes $\to$', fontsize=axisLabelSize)
ax1.tick_params(labelsize=tickSize)

#plt.legend(bbox_to_anchor=(1.25, 0.0),title='Representation', title_fontsize=legendtitleSize, 
           #fontsize=legendSize, loc='lower right')
ax1.legend(loc = 'upper center',bbox_to_anchor=(0.5,1.25),ncol=3,frameon=False)
ax1.set_xlim(-10, 405)

### Explore How Changing the Agent's Speed Affected Performance

# Create combined dataframes for each speed condition
term_rew =[np.vstack(df_discrete6100['episodes']).T[-1], 
           np.vstack(df_discrete8100['episodes']).T[-1], np.vstack(df_discrete10100['episodes']).T[-1], 
           np.vstack(df_discrete12100['episodes']).T[-1],
           np.vstack(df_discrete650['episodes']).T[-1], 
           np.vstack(df_discrete850['episodes']).T[-1], np.vstack(df_discrete1050['episodes']).T[-1], 
           np.vstack(df_discrete1250['episodes']).T[-1]]

rep = [['6']*10, ['8']*10, ['10']*10, ['12']*10, ['6']*10, ['8']*10, ['10']*10, ['12']*10]
speed = [['10,000']*40, ['5,000']*40]

combined_all = pd.DataFrame({'terminal reward': np.hstack(term_rew),
                            'representation': np.hstack(rep),
                            'pixels per sec': np.hstack(speed)})


# Plot the results as a point plot with confidence intervals as error bars
# fig, ax1 = plt.subplots(1, 1, figsize=(5,4))

sns.set_style('white')
sns.pointplot(data=combined_all, x='representation', y='terminal reward', hue="pixels per sec", 
              ci=95, palette=[COLORS[8], COLORS[5]], join=False)
sns.despine()

mu_hex100 = np.vstack(df_hex100['episodes']).mean(axis=0)[-1]
sigma_hex100 = np.vstack(df_hex100['episodes']).std(axis=0)[-1]
ci_hex100 = stats.norm.interval(0.95, loc=mu_hex100, scale=sigma_hex100)

mu_hex50 = np.vstack(df_hex50['episodes']).mean(axis=0)[-1]
sigma_hex50 = np.vstack(df_hex50['episodes']).std(axis=0)[-1]
ci_hex50 = stats.norm.interval(0.95, loc=mu_hex50, scale=sigma_hex50)

l1 = ax2.axhline(y=mu_hex100, color=COLORS[8], linestyle='-', linewidth=2.5)
plt.fill_between(np.arange(-.5,4.5), ci_hex100[0], ci_hex100[1], alpha=0.3, color=COLORS[8])

l2 = ax2.axhline(y=mu_hex50, color=COLORS[5], linestyle='-', linewidth=2.5)
plt.fill_between(np.arange(-.5,4.5), ci_hex50[0], ci_hex50[1], alpha=0.3, color=COLORS[5])

ax2.set_ylabel('Terminal Reward', fontsize=axisLabelSize)
ax2.set_xlabel('Discretization (# bins per state)', fontsize=axisLabelSize)

ax2.tick_params(labelsize=tickSize)

m1 = mlines.Line2D([], [], color=COLORS[8], marker='.', linestyle='None',
                          markersize=9, label='10,000 pix/sec')
m2 = mlines.Line2D([], [], color=COLORS[5], marker='.', linestyle='None',
                          markersize=9, label='5,000 pix/sec')
l1 = mlines.Line2D([], [], color=COLORS[8], marker='.', linestyle='-',
                          markersize=1, label='HexSSP')
l2 = mlines.Line2D([], [], color=COLORS[5], marker='.', linestyle='-',
                          markersize=1, label='HexSSP')

ax2.legend(handles=[m1, m2, (l1,l2)], labels=['10,000 pix/sec', '5,000 pix/sec', 'HexSSP'], 
          fontsize=legendSize, handler_map={tuple: HandlerTuple(ndivide=None)},
          bbox_to_anchor=(0.9, 1.2), ncol=2, frameon=False,)

ax2.set_xlim(-0.5, 3.5)

fig.tight_layout()
fig.savefig("figure4_ratbox_discrete_vs_hex.pdf", bbox_inches="tight")
plt.show()