import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

x_s = []
y_s = []
seeds=np.arange(20)
for seed in seeds:
    log_dir = "./cartpole_data" + str(seed) + "/"
    os.makedirs(log_dir, exist_ok=True)
    
    set_random_seed(seed)
    
    env = gym.make("CartPole-v1")
    env = Monitor(env, log_dir)
    
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=172000, log_interval=1)
    model.save("cartpole_all_data/a2c_cartpole" + str(seed))
    
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y_ma = moving_average(y, window=50)
    
    fig = plt.figure()
    plt.plot(x, y, color='grey' )
    plt.plot(x[len(x) - len(y_ma):], y_ma, color='blue' )
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title('Cartpole: seed ' + str(seed))
    plt.show()
    
    x_s.append(x[len(x) - len(y_ma):])
    y_s.append(y_ma)


np.save('../cartpoleData/baseline_num_timesteps.npy', np.array(x_s, dtype=object), allow_pickle=True)
np.save('../cartpoleData/baseline_rewards.npy', np.array(y_s, dtype=object), allow_pickle=True)


