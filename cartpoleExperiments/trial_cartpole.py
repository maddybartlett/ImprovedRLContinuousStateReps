import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
import warnings
import random
import pytry
import nengo
import time

import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from rlnet.utils import softmax, rend, save_gifs, next_power_of_2
import rlnet as net

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class ACTrial(pytry.Trial):
    ## PARAMETERS ##
    def params(self):
        ## Task Parameters 
        self.param('Number of learning trials', trials = 1000)
        self.param('Number of time steps per trial', steps = 500)
        self.param('Number of time steps on done', n_done = 1)
        self.param('Number of time steps on reset', n_reset = 1)
        self.param('Task or Environment', env = 'MountainCar-v0')
        self.param('Duration of task time step', env_dt = 0.001)
        
        ## Gif Parameters
        self.param('Create render gifs', gifs = False)
        
        ## Representation
        self.param('Method for representing the state', rep_ = 'Normal')
        self.param('Discretization of the representation', n_bins = 100)
        self.param('Normalize state', normalize_state = False)
        self.param('Length scale',length_scale = 1.)
        self.param('Number of rotations',n_rotates = 5)
        self.param('Specified encoder sampling',specify_encoder_samples = False)
        self.param('Neuron type', neuron_type = nengo.RectifiedLinear())
        
        ## Rule Parameters
        self.param('Learning rule', rule = "TD0")
        self.param('Epsilon for epsilon-greedy', eps = 100)
        self.param('Dynamic Epsilon',dynamic_epsilon = False)
        self.param('Learning rate', lr = 0.001)
        self.param('Action value discount', act_dis = 0.9)
        self.param('State value discount', state_dis = 0.9)
        self.param('n for TD(n)', n = None)
        self.param('Lambda for TD(lambda)', lambd = None)
        self.param('Number of trials with learning', learnTrials = None)
        
        ## Network Parameters
        self.param('Number of neurons in state ensemble', state_neurons = None)
        self.param('Proportion of neurons active', active_prop = None)
        self.param('Comments prior to running trial', pre_comment = 'N/A')
        
    def evaluate(self, param):
        total_start = time.time()
        build_start = time.time()
        
        ## INITIALIZATION ##
        ## Task Parameters       
        trials = param.trials
        steps = param.steps
        env_dt = param.env_dt
        
        # Environment
        if param.gifs == True:
            self.env = gym.make(param.env,render_mode = "rgb_array")
            self.gifs_dir = os.path.join(os.path.dirname(__file__),param.data_dir,'gifs',param.data_filename,'./')
        
        self.env = gym.make(param.env)
        self.env._max_episode_steps = 500
        
        gif_trials = [ t for t in range(param.trials) if t % (trials/4) == 0 ]
        
        ## Representation - turn this into a wrapper?
        if param.normalize_state == True:
            # check these factors
            print('computing state scaling factors')
            print(self.env.observation_space.low)
            print(self.env.observation_space.high)
            
            low = self.env.observation_space.low
            low[1] = -10            # cart velocity and pole angular velocity are otherwise (-inf,inf)
            low[3] = -10
        
            high = self.env.observation_space.high
            high[1] = 10
            high[3] = 10            
            self.state_scale = np.array([4.8, 10., 0.418, 10.])
            domain_bounds_ = np.array([[-1,-1,-1,-1],[1,1,1,1]]).T
        else:
            
            low = self.env.observation_space.low
            low[1] = -10            # cart velocity and pole angular velocity are otherwise (-inf,inf)
            low[3] = -10
        
            high = self.env.observation_space.high
            high[1] = 10
            high[3] = 10
            
            domain_bounds_ = np.array( [ low, high ] ).T
        
        if param.rep_ == 'HexSSP' or param.rep_ == 'PlaceSSP':
            state_size = len(self.env.observation_space.high)
            rep = net.representations.SSPRep(state_size,length_scale = param.length_scale,
                                                n_rotates = param.n_rotates, domain_bounds = domain_bounds_)
            print('state size: ',state_size)
            ssp_dim = rep.size_out
            print('SSP dim: ', ssp_dim)
        elif param.rep_ == 'Normal':
            rep = net.representations.NormalRep(self.env)
            rep.upper = high
            rep.lower = low
            rep.ranges = rep.upper - rep.lower
            state_size = rep.size_out
        elif param.rep_ == 'Discrete':
            rep = net.representations.OneHotRepCP( (param.n_bins,param.n_bins,param.n_bins,param.n_bins))
            state_size = rep.size_out
        
        ## Rule Parameters
        rule = getattr(net.rules, param.rule)
        n_actions = self.env.action_space.n
        # print(n_actions)
 
        eps = param.eps
        lr = param.lr
        act_dis = param.act_dis
        state_dis = param.state_dis
        n = param.n
        lambd = param.lambd
        learnTrials = param.learnTrials
        
        ## Network Parameters
        state_neurons = param.state_neurons
        active_prop = param.active_prop
        neuron_type = param.neuron_type
        
        if state_neurons != None:
            n_neurons = next_power_of_2(state_neurons)
            if n_neurons != state_neurons:
                print('# neurons must be a power of 2, requested: {}, actual: {}'.format(state_neurons,n_neurons))
        else:
            n_neurons = None
            
        if param.specify_encoder_samples == True:
            if param.rep_ == 'HexSSP':
                encoders = rep.make_encoders(n_neurons,type = 'grid')
            elif param.rep_ == 'PlaceSSP':
                encoders = []
                domain_samples = []
                for n in range(n_neurons):
                    # get a sample from the domain
                    x = np.random.uniform( low, high, size = None )
                    domain_samples.append(x)
                    
                    # project the state variable to SSP space
                    e = rep.map(x)
                    encoders.append(e)
                encoders = np.array(encoders)
                samples = np.array(domain_samples)
                print(samples)
                
        else:
            encoders = nengo.Default
        
        ## Network
        self.net = net.networks.ActorCritic(representation = rep,
                                              rule = rule(n_actions = n_actions,
                                                        lr = lr, 
                                                        act_dis = act_dis, 
                                                        state_dis = state_dis,
                                                        n = n,
                                                        lambd = lambd,
                                                        env_dt = env_dt,
                                                        learnTrials = learnTrials),
                                              state_neurons = n_neurons,
                                              active_prop = active_prop, 
                                              encoders = encoders,
                                              neuron_type = neuron_type,
                                              radius = np.sqrt(len(high))
                                              )
        build_end = time.time()
        ### LEARNING ###
        
        ## DATA SAVING ##
        Ep_rewards = []
        rdata = {}      # rewards
        sdata = {}      # states
        vdata = {}      # values    
        adata = {}      # actions
        
        self.data_dir = os.path.join(os.path.dirname(__file__),param.data_dir, param.data_filename,'./')
        
        trials_start = time.time()
        for trial in tqdm(range(trials)):
            
            trial_str = 'trial{}'.format(trial)
            
            # ### DATA SAVING ###
            rs = []             # rewards
            
            # states 
            cps = []            # cart position
            cvs = []            # cart velocity
            pas = []            # pole angle
            pvs = []            # pole angular velocity
            
            # values
            vs = []
            
            # actions
            acts = []             
            
            if (param.gifs == True) & (trial in gif_trials):
                
                if not os.path.exists(self.gifs_dir):
                    os.mkdir(self.gifs_dir)
                    
                # remember to close the figure down when done
                fig = plt.figure( figsize = (8.,5.) )
                ims = []

            ## Start environment
            reset_obj = self.env.reset()[0]
            # print('state size: ', state_size)
            update_state = rep.get_state(reset_obj[:state_size],self.env )
            if param.normalize_state == True:
                update_state /= self.state_scale
            
            value, action_logits = self.net.step(update_state, 0, 0, reset = True) ## state and action values

            ## Each time step
            for step in range(steps):
            
                ### CORE TRAINING LOOP ###
                p = np.random.random()
                
                ## Force exploration using epsilon greedy
                if trial < learnTrials and p < eps:
                    action_logits = np.random.uniform(0, n_actions, n_actions)

                ## convert action_logits to action_choice
                #action_choice = np.dot(softmax(action_logits), [-1,1])
                action_choice = np.argmax(softmax(action_logits))
                
                ## Do the action   
                obs, reward, done, t, info = self.env.step(action_choice)   
                if trial > learnTrials:
                    reward = 0

                ## Collect agent location and direction
                state = obs[:state_size]
                
                ## Get new state
                current_state = rep.get_state(state,self.env)
                # print(state,rep.map(state))
                
                if param.normalize_state:
                    # print('before normalization: ', current_state)
                    current_state /= self.state_scale
                    # print('after normalization: ', current_state)
                
                ## For the continuous action spaces, learn about the probability distribution across action primitives
                action_learn = softmax(action_logits*1)             
                
                ## Update state and max weighted action values 
                value, action_logits = self.net.step(current_state, action_learn, reward)

                rs.append( reward )

                cps.append(current_state[0])
                cvs.append(current_state[1])
                pas.append(current_state[2])
                pvs.append(current_state[3])
                
                vs.append(value[0])
                
                # es.append( current_state[0] - state[0] )      # discretization error

                ## On Finish
                ## When using multi-step back-ups, once the agent has reached the goal you 
                ## need it to sit there for n time steps and continue doing updates
                if done:
                    done_counter = param.n_done
                    if n is not None:
                        for j in range(n):
                            ## Update state and action values 
                            reward = 0
                            value, action_logits = self.net.step(current_state, action_learn, reward)
                            
                            rs.append(reward) ## save reward
                            
                            cps.append(current_state[0])
                            cvs.append(current_state[1])
                            pas.append(current_state[2])
                            pvs.append(current_state[3])
                            
                            vs.append(value[0])
                    else:
                        while(done_counter > 0):
                            reward = 0
                            value, action_logits = self.net.step(current_state, action_learn, reward)
                            
                            rs.append(reward) ## save reward
                            
                            cps.append(current_state[0])
                            cvs.append(current_state[1])
                            pas.append(current_state[2])
                            pvs.append(current_state[3])
                            
                            vs.append(value[0])
                            done_counter -= 1
                    break
                
                ### DATA SAVING ###
                rdata[trial_str] = rs
                vdata[trial_str] = vs
                
                for label,data in zip( ['cp','cv','pa','pv'], [cps,cvs,pas,pvs] ):
                    sdata['{}-{}'.format(label,trial_str)] = data

                ## Store images for gifs
                if param.gifs == True:
                    if trial in gif_trials:
                        im = rend(self.env)
                        ims.append([im])
                        
            if param.gifs == True:
                if trial in gif_trials:
                    save_gifs(fig, ims, trial, self.gifs_dir)
                plt.close(fig)
            
            Ep_rewards.append(np.sum(rs)) ## Store total reward for episode         
            # feel like this should be in the agent not in the main script
            
            if param.dynamic_epsilon == True:
                if np.mean(Ep_rewards[trial-10:trial]) > np.mean(Ep_rewards[trial-20:trial-10]) + np.std(Ep_rewards[trial-20:trial-10]):
                    eps -= 0.001
                # elif np.mean(Ep_rewards[trial-10:trial]) < np.mean(Ep_rewards[trial-20:trial-10])- np.std(Ep_rewards[trial-20:trial-10]):
                #     eps += 0.001
        
        trials_end = time.time()
        total_end = time.time()
        
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            
        rdf = pd.DataFrame( dict( [ (k,pd.Series(r)) for k,r in rdata.items() ] ) )
        rdf.to_csv(os.path.join(self.data_dir,'rewards.csv'))
            
        sdf = pd.DataFrame( dict( [ (k,pd.Series(s)) for k,s in sdata.items() ] ) )
        sdf.to_csv(os.path.join(self.data_dir,'states.csv'))            

        vdf = pd.DataFrame( dict( [ (k,pd.Series(v)) for k,v in vdata.items() ] ) )
        vdf.to_csv(os.path.join(self.data_dir,'values.csv'))            
            
        self.env.close()

        ## Save parameters, data/return path
        trial_ID = param.data_filename
        
        # dimensionality of the representation
        rep_dim = rep.size_out
        
        reward_rolling_mean = rdf.sum(axis = 0).rolling(100).mean()
        
        # the average reward the agent received in the last 100 epsiodes
        terminal_reward_learning = reward_rolling_mean[learnTrials-1]
        terminal_reward = reward_rolling_mean[-1]
        
        # the number of episodes to reach an average reward of 50. (as observed in the last 100 episodes)
        episodes_to_learn = next(itertools.chain(iter(i for i,v in enumerate(reward_rolling_mean) if v > 195.), [-1]))
        if episodes_to_learn == -1:
            episodes_to_learn = np.nan
        
        # return an empty dict to save file data
        return {    'dimensionality'            : rep_dim,
                    'terminal_reward_learning'  : terminal_reward_learning,
                    'terminal_reward'           : terminal_reward,
                    'episodes_to_learn'         : episodes_to_learn,
                    'trial_ID'                  : trial_ID,
                    'build_time'                : build_end - build_start,
                    'total_time'                : total_end - total_start,
                    'avg_trial_time'            : ( trials_end - trials_start ) / trials
                }