import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import levene

window = 50

# Grab all files in subfolder HexPlace (this will contain reward.csv, reward(1).csv, etc)
ssp_files = [f for f in listdir("../cartpoleData/baseline-comparison/HexSSP")]
all_mean_rews = []
all_ts = []
for i, name in enumerate(ssp_files):
    ssp_data = np.genfromtxt("../cartpoleData/baseline-comparison/HexSSP/" + name, delimiter=',')[1:]
    ssp_data[np.isnan(ssp_data)] = 0
    # Epsiodic rewards
    ep_rews = np.sum(ssp_data[:,1:],axis=0)
    # computing number of timesteps
    timesteps = np.cumsum(ep_rews)
    # Moving average
    ma_ep_rews = np.convolve(ep_rews, np.repeat(1.0, window) / window, 'same')
    all_mean_rews.append(ma_ep_rews)
    all_ts.append(timesteps)

# Problem: in both the HexSSP and baseline data rewards are sampled at different
# timesteps, so we cannot simply take the mean of them all.
# To get around the uneven sampling times, I use cubic splines to interpolate 
# the data at a common set of timesteps / x-axis values. 
ssp_ts = np.arange(np.max([np.min(t) for t in all_ts]), np.min([np.max(t) for t in all_ts]), 50)
ssp_int_rews = np.zeros((len(ssp_files),len(ssp_ts)))
for i in range(len(ssp_files)):
    cs = CubicSpline(all_ts[i], all_mean_rews[i])
    ssp_int_rews[i,:] = cs(ssp_ts)
# Median (or mean) and upper & lower bounds of intervals
#ssp_mean = np.mean(ssp_int_rews, axis=0)
ssp_mean = np.percentile(ssp_int_rews, 50, axis=0, interpolation = 'midpoint')
ssp_min = np.percentile(ssp_int_rews, 10, axis=0, interpolation = 'midpoint')
ssp_max =  np.percentile(ssp_int_rews, 90, axis=0, interpolation = 'midpoint')

# The baseline data
baseline_ts = np.load('../cartpoleData/baseline-comparison/baseline/baseline_num_timesteps.npy', allow_pickle=True)
baseline_rs = np.load('../cartpoleData/baseline-comparison/baseline/baseline_rewards.npy', allow_pickle=True)

# I ran 20 extra trials and saved with a different filename, if the 
# cartpolae_a2c.py is simply run with 20 trials this code is not needed

baseline2_ts = np.load('../cartpoleData/baseline-comparison/baseline/baseline_num_timesteps_set2.npy', allow_pickle=True)
baseline2_rs = np.load('../cartpoleData/baseline-comparison/baseline/baseline_rewards_set2.npy', allow_pickle=True)
baseline_ts= np.concatenate((baseline_ts,baseline2_ts))
baseline_rs= np.concatenate((baseline_rs,baseline2_rs))

# Again usnig interpolate  due to unequal sampling 
bl_ts = np.arange(np.max([np.min(t) for t in baseline_ts]), np.min([np.max(t) for t in baseline_ts]), 50)
bl_int_rews = np.zeros((len(baseline_ts),len(bl_ts)))
for i in range(len(baseline_ts)):
    cs = CubicSpline(baseline_ts[i], baseline_rs[i])
    bl_int_rews[i,:] = cs(bl_ts)

#bl_mean = np.mean(bl_int_rews, axis=0)
bl_mean = np.percentile(bl_int_rews, 50, axis=0, interpolation = 'midpoint')
bl_min = np.percentile(bl_int_rews, 10, axis=0, interpolation = 'midpoint')
bl_max = np.percentile(bl_int_rews, 90, axis=0, interpolation = 'midpoint')


fig,ax = plt.subplots(1,1,figsize=(3.5, 2.5))
ax.plot(bl_ts, bl_mean, c="red", linewidth=1.5, label='Baseline A2C')
ax.fill_between(bl_ts, bl_min, bl_max, color="red", alpha=.15)

ax.plot(ssp_ts, ssp_mean, c="blue", linewidth=1.5, label='HexSSP')
ax.fill_between(ssp_ts, ssp_min, ssp_max, color="blue", alpha=.15)

ax.set_xlabel('Number of Timesteps')
ax.set_ylabel('Episodic Reward')
ax.legend(loc='lower right')
ax.set_xlim([0,bl_ts[-1]])
ax.set_xticks([0,50000,100000,150000])

for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
plt.savefig('figure6_cartpole_ssp_vs_baseline.pdf')
plt.show()

### Investigate
ssp_end_vals = [all_mean_rews[i][np.where(all_ts[i]> bl_ts[-1])[0][0]] for i in range(len(all_ts))]
bl_end_vals = [baseline_rs[i][np.where(baseline_ts[i]> bl_ts[-1])[0][0]] for i in range(len(baseline_rs))]
print(np.var(ssp_end_vals))
print(np.var(bl_end_vals))
print(levene(ssp_end_vals,bl_end_vals, center='median'))
