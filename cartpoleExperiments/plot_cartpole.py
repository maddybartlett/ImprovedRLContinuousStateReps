import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import levene

window = 50

# Grab all files in subfolder HexPlace (this will contain reward.csv, reward(1).csv, etc)
ssp_files = [f for f in listdir("./HexPlace")]
all_mean_rews = []
all_ts = []
for i, name in enumerate(ssp_files):
    ssp_data = np.genfromtxt("./HexPlace/" + name, delimiter=',')[1:]
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
baseline_ts = np.load('../cartpoleData/baseline_num_timesteps.npy', allow_pickle=True)
baseline_rs = np.load('../cartpoleData/baseline_rewards.npy', allow_pickle=True)


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


fig=plt.figure(figsize=(3.37566, 2))
plt.plot(bl_ts, bl_mean, c="red", linewidth=1.5, label='Baseline A2C')
plt.fill_between(bl_ts, bl_min, bl_max, color="red", alpha=.15)

plt.plot(ssp_ts, ssp_mean, c="blue", linewidth=1.5, label='HexSSP')
plt.fill_between(ssp_ts, ssp_min, ssp_max, color="blue", alpha=.15)

plt.xlabel('Number of Timesteps')
plt.ylabel('Episodic Reward')
plt.legend(loc='lower right')
plt.xlim([0,bl_ts[-1]])
utils.save(fig,'cartpole_ssp_vs_baseline.pdf')



ssp_end_vals = [all_mean_rews[i][np.where(all_ts[i]> bl_ts[-1])[0][0]] for i in range(len(all_ts))]
bl_end_vals = [baseline_rs[i][np.where(baseline_ts[i]> bl_ts[-1])[0][0]] for i in range(len(baseline_rs))]
print(np.var(ssp_end_vals))
print(np.var(bl_end_vals))
print(levene(ssp_end_vals,bl_end_vals, center='median'))
