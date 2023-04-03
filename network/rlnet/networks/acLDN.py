import nengo
import numpy as np
from scipy.special import legendre
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from .ldn import LDN
from rlnet.utils import sparsity_to_x_intercept
   
## Actor Critic Network with LDN memories for rewards and values
## 2 types of reward decoder - one is for discrete time, one for continuous time
class ActorCriticLDN(object):
    ''' Nengo model implementing an Actor-Critic network.
    Single-layer network
    State value and reward are stored in ldn memories
    Inputs: state, action, reward and reset
    Outputs: updated state value, action values for actions available in current state, value of current state
    
    (Note: value of current state is fed back into the network's own value memory)
    
    Example of Usage:
        >> rep = NormalRep((8,8,4))
        >> ac = ActorCriticLDN(rep, 
                 ActorCriticTD0(n_actions=3, lr=0.1, act_dis=0.9, state_dis=0.95),
                 state_neurons=1000, 
                 active_prop=0.1,
                 theta=0.001,
                 q_r=3,
                 q_v=3,
                 continuous=True,
                 neuron_type=nengo.RectifiedLinear()
                )
    
    '''
    def __init__(self, representation, rule, state_neurons=None, active_prop=None, theta = 0.1, q_r = 10, q_v = 10,
                 continuous = False, ens_neurons=nengo.RectifiedLinear(), 
                 report_ldn = False, **ensemble_args):
        self.representation = representation
        ## set dim = size of state representation
        dim = representation.size_out
        ## create empty array for action values being updated
        self.update_action = np.zeros(rule.n_actions) 

        self.state_neurons = state_neurons
        self.continuous = continuous
        
        ## empty arrays for state value and action values
        self.state_value = np.zeros(1)
        self.action_values = np.zeros(rule.n_actions)
        
        n = rule.n
        ## add 10% to length of LDN window to increase accuracy at the extremes
        theta = theta + (theta * 0.1)
        state_dis = rule.state_dis
        
        self.env_dt = rule.env_dt
        self.report_ldn = report_ldn
        
               
        ## Create nengo model
        self.model = nengo.Network()
        with self.model:
            
            ## empty array for state
            ## size = size of state representation + number of actions + reward + whether env was reset
            self.state = np.zeros(dim+rule.n_actions+2)  
            ## create nengo node for containing state
            self.state_node = nengo.Node(lambda t: self.state)
            
            ## create nengo node for containing the learning rule
            ## input size in is state_neurons (state) + n_actions + reward + env.reset
            self.rule = nengo.Node(rule, size_in=state_neurons+rule.n_actions+3)                
            ## create ensemble for containing the state representation
            self.ensemble = nengo.Ensemble(n_neurons=self.state_neurons, dimensions=dim,
                                           neuron_type=ens_neurons,
                                           intercepts = nengo.dists.Choice([sparsity_to_x_intercept(dim, active_prop)]),
                                           **ensemble_args
                                          )
            ##connect the state representation to the ensemble
            nengo.Connection(self.state_node[:dim], self.ensemble, synapse=None)

        
            ## LDN MEMORIES FOR REWARDS AND VALUES
            ## create LDNs for rewards and state values
            ldn_r = LDN(theta=theta, q=q_r, size_in=1)
            ldn_v = LDN(theta=theta, q=q_v, size_in=1)
            ## create memory nodes which perform LDN transformation
            self.reward_memory = nengo.Node(ldn_r)
            self.value_memory = nengo.Node(ldn_v)
            ## feed the reward from the state node into the LDN Reward memory               
            nengo.Connection(self.state_node[-2], self.reward_memory, synapse=None) 

            ## Create reward decoders
            ## Decoders for discrete time -- TD(n) with LDN memories
            if self.continuous == False:
                reward_decoders = np.sum([np.exp(-state_dis*theta*(1-dth))* ldn_r.get_weights_for_delays(dth) 
                                      for dth in np.linspace(0,1, n)],axis=0)
                
            ## Decoders for continuous time -- TD(theta)  
            elif self.continuous == True:               
                reward_decoders = np.zeros(q_r)
                for i in range(q_r):
                    intgrand = lambda x: (state_dis**(theta*(1-(x))))*legendre(i)(2*x-1)
                    reward_decoders[i]=integrate.quad(intgrand, 0,1)[0]
                    
                reward_decoders =  np.kron(np.eye(1), reward_decoders.reshape(1, -1))
            
            ## Value memory decoders just fetch the value from n time steps ago
            value_decoders = ldn_v.get_weights_for_delays(1)

            
            ## connect the state ensemble to the rule node
            nengo.Connection(self.ensemble.neurons, self.rule[:state_neurons], synapse=None)
            ## connect the actions, reward, reset and value-to-be-updated to the rule node
            nengo.Connection(self.state_node[dim:-2], self.rule[state_neurons:-3], synapse=None) ##actions
            nengo.Connection(self.reward_memory, self.rule[-3], transform = reward_decoders, synapse=None) ##reward
            nengo.Connection(self.state_node[-1], self.rule[-2], synapse=None) ##reset
            nengo.Connection(self.value_memory, self.rule[-1], transform = value_decoders, synapse=None) ##update value
            
            ## Decoded Reward node - this node is used to probe the values decoded out from the reward memory
            if self.report_ldn == True:
                self.rdec = nengo.Node(None, size_in=1)
                nengo.Connection(self.reward_memory, self.rdec, transform=reward_decoders)
                self.vdec = nengo.Node(None, size_in=1)
                nengo.Connection(self.value_memory, self.vdec, transform=value_decoders)

            ## create node for containing the updated state value
            self.value_node = nengo.Node(self.value_node_func, size_in=1)
            
            ## create node for containing the updated action values
            self.action_values_node = nengo.Node(self.action_values_node_func, size_in=rule.n_actions)
            
            ## send first output from rule node (updated state value) to the state value node
            nengo.Connection(self.rule[0], self.value_node, synapse=None)
            ## send updated action values from rule node to action value node
            nengo.Connection(self.rule[1:-1], self.action_values_node, synapse=None)
            
            ## send the current state value to the value memory
            nengo.Connection(self.rule[-1], self.value_memory, synapse=0)
            
            ## Probes
            if self.report_ldn == True:
                self.p_rldn = nengo.Probe(self.reward_memory) ##reward memory activity
                self.p_rdec = nengo.Probe(self.rdec) ##decoded reward memory
                self.p_vldn = nengo.Probe(self.value_memory) ##value memory activity
                self.p_vdec = nengo.Probe(self.vdec) ##decoded value memory
            
        ##run model
        self.sim = nengo.Simulator(self.model)
        
    def step(self, state, update_action, reward, reset=False, report=False):
        '''Function for running the model for one time step.
        
        Inputs: agent's state, chosen action, reward
        Outputs: state value, action values'''

        ## set update_action to an array of 0's with one value for each action
        self.update_action[:] = 0
        ## set the update_action value at the position of the chosen action to 1
        self.update_action[update_action] = 1
        
        ## create state variable containing state representation,
        ## update_action array, reward, and whether or not the env was reset
        self.state[:] = np.concatenate([
            self.representation.map(state),
            self.update_action,
            [reward, reset],]
            )
        
        ## run model for one step
        if self.env_dt != None:
            for i in range(int(self.env_dt/0.001)):
                self.sim.step()
        else:
            self.sim.step()
            
        ## Fetch probe data
        #self.a = self.sim.data[self.p_ldn] ##reward memory activity
        #self.d = self.sim.data[self.p_dec] ##decoded reward memory
        
        ## return the updated state and action values from the model
        ## these are the values returned by the learning rule
        if report == True:
            ## Fetch probe data
            self.rldn_in = self.sim.data[self.p_rldn] ##reward memory activity
            self.rldn_out = self.sim.data[self.p_rdec] ##decoded reward memory
            self.vldn_in = self.sim.data[self.p_vldn] ##value memory activity
            self.vldn_out = self.sim.data[self.p_vdec] ##decoded value memory
            
            return self.state_value, self.action_values, self.rldn_in, self.rldn_out, self.vldn_in, self.vldn_out
        else:
            return self.state_value, self.action_values

    
    def get_tuning(self):
        
        plt.plot(*nengo.utils.ensemble.tuning_curves(self.ensemble, self.sim))
    
    def get_policy(self):
        ''''''
        ## create coordinate matrix from the dimensions of the state space
        X, Y, Z = np.meshgrid(np.arange(8), np.arange(8), np.arange(4))
        #X, Y = np.meshgrid(np.arange(50), np.arange(50))
        ## array of coordinates in state space
        pts = np.array([X, Y, Z])
        #pts = np.array([X, Y])
        
        ## flatten into 2D array
        pts = pts.reshape(3, -1)
        ## translate array into chosen state representation of entire state space
        X = [self.representation.map(x).copy() for x in pts.T]
        
        ## if using ensemble, calculate the tuning curves of the ensemble
        if self.state_neurons is not None:
            _, A = nengo.utils.ensemble.tuning_curves(self.ensemble, self.sim, inputs=X)
            ## reshape the activities of the ensemble neurons and assign to X
            X = A.reshape((8,8,4,-1))
            #X = A.reshape((50,50,-1))
            
        ## if not using ensemble, just reshape the state representationof the entire state space
        else:
            X = np.array(X).reshape((8,8,4,-1))
            #X = np.array(X).reshape((50,50,-1))
            
        ## get weight matrix
        w = self.sim.signals[self.sim.model.sig[self.rule.output]['_state_w']]
        
        ## Calculate policy by calculating dot product of state space and weight matrix
        V = X.dot(w.T)
        
        ## return policy
        return V 
    
    ## function for state value node
    def value_node_func(self, t, x):
        ## identity function
        self.state_value[:] = x
        
    ## function for action value node
    def action_values_node_func(self, t, x):
        ## identity function
        self.action_values[:] = x