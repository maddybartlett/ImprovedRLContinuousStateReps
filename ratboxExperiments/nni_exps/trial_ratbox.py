from tqdm import tqdm #progress bar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import sys
sys.path.insert(0, '..\\..\\network')

import rlnet as net
from rlnet.utils import softmax, rend, save_gifs, get_ac_output, next_power_of_2

import gymnasium as gym
import ratbox
from ratbox.wrappers import ConvertSkidWrapper as CSW
import pytry
import nengo

import nni

class ACTrial(pytry.Trial):
    ## PARAMETERS ##
    def params(self):
        ## Task Parameters 
        self.param('Number of learning trials', trials=1000),
        self.param('Number of time steps per trial', steps=500),
        self.param('Task or Environment', env='RatBox-empty-v0'),
        self.param('Steering model for ratbox', steering='discrete'),
        self.param('Duration of task time step', env_dt=0.001),
        
        ## Gif Parameters
        self.param('Create render gifs', gifs=False),
        self.param('Directory for saving gifs', gif_dir='./gifs/'),
        
        ## Representation
        self.param('Method for representing the state', rep="HexSSP"),
        self.param('Number of rotates for hex SSPs', n_rotates=4),
        self.param('Number of scales for hex SSPs', n_scales=1),
        self.param('Length scale for hex SSPs', length_scale=1),
        self.param('Ranges for One Hot representation', rep_ranges=(1,1,1)),
        
        ## Rule Parameters
        self.param('Learning rule', rule="TD0"),
        self.param('Epsilon for epsilon-greedy', eps=100),
        self.param('Dynamic epsilon', dynamic_epsilon=False),
        self.param('Learning rate', lr=0.001),
        self.param('Action value discount', act_dis=0.8),
        self.param('State value discount', state_dis=0.9),
        self.param('n for TD(n)', n=None),
        self.param('Lambda for TD(lambda)', lambd=None),
        self.param('Number of trials with learning', learnTrials=None),
        
        ## Network Parameters
        self.param('Number of neurons in state ensemble', state_neurons=None),
        self.param('Sample grid encoders', specify_encoder_samples=False),
        self.param('Proportion of neurons active', active_prop=0.1),
        self.param('Type of neuron in state ensemble', state_neuron_type=nengo.RectifiedLinear()),
        
    def evaluate(self, param):
        ## INITIALISE THINGS ##
        ## Task Parameters       
        trials=param.trials
        steps=param.steps
        env_dt=param.env_dt
        
        ## Gif Parameters
        #screen=param.screen
        gifs=param.gifs
        
        ## Environment
        self.env=gym.make(param.env, render_mode="rgb_array", steering=param.steering, dt=env_dt)
        n_actions=self.env.n_actions
        if param.steering == "skidsteer":
            self.env = CSW(self.env)
            n_actions = 3
        
        ## Representation
        if param.rep == "HexSSP":
            self.rep=net.representations.SSPRep(3, n_scales=param.n_scales, n_rotates=param.n_rotates, length_scale=param.length_scale)
        elif param.rep == "OneHot":
            self.rep=net.representations.OneHotRep(param.rep_ranges)
       
        
        ## Rule Parameters
        rule=getattr(net.rules, param.rule)
        eps=param.eps
        lr=param.lr
        act_dis=param.act_dis
        state_dis=param.state_dis
        n=param.n
        lambd=param.lambd
        if param.learnTrials != None:
            learnTrials=param.learnTrials
        else:
            learnTrials=trials+1
        
        ## Network Parameters
        state_neurons=param.state_neurons
        ssp_dim=None
        ## Set state neurons according to the number of scales and rotates being used in the representation
        if param.rep == "HexSSP":
            ssp_dim = 8*param.n_scales*param.n_rotates 
            state_neurons = 10 * ssp_dim
            
        if state_neurons != None:
            n_neurons = next_power_of_2(state_neurons)
            if n_neurons != state_neurons:
                print('# neurons must be a power of 2, requested: {}, actual: {}'.format(state_neurons,n_neurons))
                
        
        active_prop=param.active_prop
        state_neuron_type=param.state_neuron_type
        
        if param.specify_encoder_samples == True:
            encoders = self.rep.make_encoders(state_neurons)
        else:
            encoders = nengo.Default
        
        ## Network
        self.net = net.networks.ActorCritic(representation=self.rep,
                                              rule=rule(n_actions=n_actions,
                                                        lr=lr, 
                                                        act_dis=act_dis, 
                                                        state_dis=state_dis,
                                                        n=n,
                                                        lambd=lambd,
                                                        env_dt=env_dt,
                                                        learnTrials=learnTrials),
                                              state_neurons=state_neurons,
                                              active_prop=active_prop, 
                                              encoders=encoders,
                                              neuron_type=state_neuron_type)
        ## Data lists
        Ep_rewards=[]
        Rewards=[]
        Values=[] 
        Roll_mean=[]
        Policy=[]
        Actions=[]
        States=[]
        
        ## LEARNING ##
        ## Each learning trial
        for trial in tqdm(range(trials)):
            
            if gifs==True:
                ## Figure for gifs
                fig = plt.figure(figsize=(8.0, 5.0))
                ims=[] ##render storage
                
            ## temp data storage
            rs=[] #rewards
            vs=[] #state values
            act=[]
            sts=[]
            
            ## Start environment
            update_state = self.rep.get_state(self.env.reset()[0][:3], self.env)
            value, action_logits = self.net.step(update_state, 0, 0, reset=True) ##get state and action values
            sts.append(update_state)
            
            if gifs==True:
                if trial % (trials/4) == 0 or trial == trials-1:
                    im = rend(self.env)
                    ims.append([im])

            ## Each time step
            for step in range(steps):
                p = np.random.random()
                
                ## Force exploration using epsilon greedy
                if trial < learnTrials and p < eps:
                    action_logits = np.random.uniform(0, 6, n_actions)
                   
                ## Collect the action values to be given to the steering model in the environment
                ## these will be used to choose the next action
                action_choice = action_logits
                act.append(action_choice.copy())
                
                ## For the discrete action space, only learn about the final action choice
                action_learn = softmax(action_logits)

                ## Store images for gifs
                if gifs==True:
                    if trial % (trials/4) == 0 or trial == trials-1:
                        im = rend(self.env)
                        ims.append([im])
                        
                ## Do the action   
                blah, reward, done, t, info = self.env.step(action_choice)
                #if trial == trials-1:
                #    print(action_choice)
                ## Collect agent location and direction
                obs = blah[:3]

                ## Get new state
                current_state = self.rep.get_state(obs, self.env)
                sts.append(current_state)

                ## Update state and max weighted action values 
                value, action_logits = self.net.step(current_state, action_learn, reward)

                ## Store data
                rs.append(reward) ##save reward
                vs.append(value.copy()) ##save state value
                
                ## On Finish
                ## When using mutli-step back-ups, once the agent has reached the goal you 
                ## need it to sit there for n time steps and continue doing updates
                if done:
                    if n is not None:
                        for j in range(n):
                            ##Update state and action values 
                            reward = 0
                            value, action_logits = self.net.step(current_state, action_learn, reward)

                            vs.append(value.copy()) ##save state value
                            rs.append(reward) ##save reward
                    break
                    
            ## Collect data
            Ep_rewards.append(np.sum(rs)) ##Store total reward for episode
            Rewards.append(rs) ##Store all rewards in episode
            Values.append(vs) ##Store all values in episode 
            Actions.append(act)
            States.append(sts)
            
            if trial > 100 and param.dynamic_epsilon==True:
                if np.mean(Ep_rewards[trial-10:trial]) > 0.5:
                    eps -= 0.02
                
            
            ## Save gifs
            if trial % (trials/4) == 0 or trial == trials-1:                   
                policy, vals, pts  = get_ac_output(self.net, self.env, n_grid=10, n_ranges=param.rep_ranges)
                Policy.append(np.asarray([policy, vals, pts]))

                if gifs==True:
                    save_gifs(fig, ims, trial, param.gif_dir)                 
                
        #convert list of rewards per episode to dataframe    
        rewards_over_eps = pd.DataFrame(Ep_rewards)
        #calculate a rolling average reward across previous 100 episodes
        Roll_mean.append(rewards_over_eps[rewards_over_eps.columns[0]].rolling(100).mean())
        
        self.env.close()

        ## Return data as dictionary
        return dict(
            n_neurons=state_neurons,
            ssp_dim=ssp_dim,
            episodes=Ep_rewards,
            rewards = Rewards,
            values = Values,
            roll_mean = Roll_mean,
            policy=Policy,
            actions=Actions,
            states=States,
            rgb=self.env.render(),
            )