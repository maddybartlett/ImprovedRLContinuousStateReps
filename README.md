# Improving Reinforcement Learning with Biologically Motivated Continuous State Representations

Repository to accompany [Bartlett, Simone, Dumont, Furlong, Eliasmith, Orchard & Stewart (2022)]() "Improving Reinforcement Learning with Biologically Motivated Continuous State Representations" ICCM Paper.

## Advantage Actor Critic (A2C) Network: 

Across all experiments the same A2C network was used. 

<p align="center">
<img src="https://github.com/maddybartlett/ImprovedRLContinuousStateReps/blob/main/figures/a2c.png" width="300"/>
</p>

The network was written in Python and [Nengo](https://www.nengo.ai/) using the Neural Engineering Framework ([NEF](http://compneuro.uwaterloo.ca/research/nef/overview-of-the-nef.html)). 

It is a shallow (single hidden layer), Advantage Actor-Critic network which takes as input the agent's state, most recent action and most recent reward. 
A single hidden layer is used to contain the state representation. This layer constitutes either a one-hot vector where discrete representations are used, or a population of rectified linear neurons which encode the continuous SSP representation. 
[Temporal Difference](http://incompleteideas.net/book/RLbook2020.pdf) (TD) learning rules are implemented as a node within the network and used to update the weights mapping the state representation to the state value (critic) and the policy or action values (actor). 

In these experiments two TD learning rules were used. <br>
**Ratbox** used an alternative version of TD(0) where the policy was formulated as an isotropic Gaussian distribution over the action vector (`TD0iG`). <br>
**CartPole**, in contrast, used the standard TD(0) learning rule. 

## Ratbox Experiments

### Task:

**[Ratbox Blocks Environment](https://github.com/maddybartlett/Ratbox)**: Agent has to navigate through a 2D world containing obstacles in order to reach a goal location. Agent starts in the top left-hand corner and the goal is located in the bottom right-hand corner. Agent receives a reward of $100$ for reaching the goal (discounted according to the number of steps the agent took to reach the goal) and a penalty of $-0.5$ whenever it bumps into an obstacle. Agent uses the *compass* steering model. 

<p align="center">
<img src="https://github.com/maddybartlett/ImprovedRLContinuousStateReps/blob/main/figures/blocksroom.png" width="300"/>
</p>

In each learning trial or episode the agent has 200 timesteps in which to reach the goal. If the goal is not reached the agent receives a reward of $0$ and is returned to the start location to try again. In all experiments the agent was given 500 learning trials in which to learn to solve the task. 

### Study Design:

$5 \times 2$ study design.
Comparison of 5 state representation methods (1 continuous and 4 tabular): 

1) Hexagonal Spatial Semantic Pointers (HexSSPs) for continuous space representation
2) Tabular with 6 bins per dimension
3) Tabular with 8 bins per dimension
4) Tabular with 10 bins per dimension
5) Tabular with 12 bins per dimension 

### Procedure:

**NNI Experiments**

To reproduce the NNI hyperparameter optimization you will need to have NNI version 2.6.1 installed. 
```
pip install nni==2.6.1
```

The first stage of this experiment was to conduct hyperparameter optimisation using Microsoft's [Neural Network Intelligence (NNI)](https://nni.readthedocs.io/en/stable/index.html#). 
5 separate NNI optimization experiments were run to find the optimal parameters for the network when using each of the 5 representation methods. 

The experiment and configuration files for these optimization experiments can be found in `ratboxExperiments/nni_exps`. In order to run the NNI optimizations yourself simply download these files, the network folder and the `trial_ratbox.py` file then following these steps:

1) open a command prompt/terminal in the nni_exps directory
2) enter the following command to start the NNI optimization experiment, replacing `CONFIG_FILE.YML` with the chosen configuration file (e.g. `config_ratbox_discrete6`) 
```
nnictl create --config CONFIG_FILE.YML
```

Make sure to create the data directory and (if necessary) change the `data_dir` variable in each of the *exp* files before running. 

Hyperparameter optimization was performed using the Annealing algorithm and the network was optimized to maximize the reward averaged over the last 100 trials. The maximum number of NNI trials was set to 100. Data was saved in *.txt* format and converted to CSVs and Pickle files for data exploration. 

**10 random seeds**

The second stage of this study involved running the optimized networks with 10 random seeds in order to assess the network's behaviour when solving the task. 

Converting the NNI *.txt* data files to *.csvs* or *.pkl* can be done using the cleaning scripts either in jupyter notebook (`cleanNNIData.ipynb`) or from the command prompt/terminal by running:

```
python clean_nni.py [PATH_TO_DATA] [PATH_TO_SAVE_DIRECTORY]
```

The *getBestParams.ipynb* notebook can then be used to identify the hyperparameters which produced the best performance for each network. One set of hyperparameter values was chosen from the top 5% of NNI experiments and used for the rest of the experiments. The table below shows the chosen hyperparameter values.

| Parameters | HexSSPs | 6 bins | 8 bins | 10 bins | 12 bins |
| ---------- | ------- | ------ | ------ | ------- | ------- |
| learning rate | $0.206206$ | $0.300605$ | $0.247154$ | $0.300605$ | $0.300605$ |
| action value discount | $0.820566$ | $0.947206$ | $0.949027$ | $0.947206$ | $0.947206$ |
| state value discount | $0.857552$ | $0.966626$ | $0.841157$ | $0.966626$ | $0.966626$ |
| epsilon | $0.360414$ | $0.439516$ | $0.580777$ | $0.439516$ | $0.439516$ |
| proportion of active neurons | $0.156189$ |  |  |  |  |
| rotations of $V$ | $7$ |  |  |  |  |
| scalings of $V$ | $6$ |  |  |  |  |
| length scale of representation | $91.775829$ |  |  |  |  |

The A2C network using each representation was run 10 times with 10 random seeds and the results collected as npz files and then converted to csv files. 

In order to replicate these experiments you need to use the files in `ratboxExperiments/run10`. 
Simply open a command prompt/terminal in that directory and then enter the following, replacing `EXP.PY` with the relevant file name (e.g. `run_ratbox100_discrete6_exp.py`)

```
python EXP.PY
```

Make sure to create the data directory and (if necessary) change the `data_dir_` variable in each of these experiment files before running. 

**Changing Agent Speed**

The final stage in these experiments was to change the agent's maximum speed from 10,000 pixels per second (100 pixels per timestep) to 5,000 pixels per second (50 pixels per timestep). The same network configurations were then run 10 times with 10 random seeds. The networks were **not** re-optimized for the new task but maintained the same hyperparameter values as shown in the table above. 

**Data Exploration**

Converting the 10-seeds *.npz* data files to *.csvs* can be done using the cleaning scripts either in jupyter notebook (`cleanNPZData.ipynb`) or from the command prompt/terminal by running:

```
python clean_npz.py [PATH_TO_DATA] [PATH_TO_SAVE_DIRECTORY]
```
Make sure that you navigate to the directory containing these scripts before running them, and that you replace [PATH_TO] with the actual paths to the relevant locations. 

The *exploreData.ipynb* notebook can then be used to create plots of the data and provide a descriptive analysis of the results. 
