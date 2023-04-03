import nengo
import numpy as np
import matplotlib.pyplot as plt

from rlnet.utils import sparsity_to_x_intercept
  
## Actor-Critic without LDNs ##
class ActorCritic(object):
    ''' Nengo model implementing an Actor-Critic network.
    Single-layer network
    Inputs: state, action, reward and reset
    Outputs: updated state value, action values for actions available in current state
    
    Example of Usage:
        >> rep = NormalRep((8,8,4))
        >> ac = ActorCritic(rep, 
                 ActorCriticTD0(n_actions=3, alpha=0.1, beta=0.9, gamma=0.95),state_neurons
                 state_neurons=1000, 
                 neuron_type=nengo.RectifiedLinear()
                 intercepts=nengo.dists.Uniform(0.01, 0.5)
                )
    '''
    def __init__(self, representation, rule, state_neurons=None, active_prop=None, neuron_type=nengo.RectifiedLinear(),
                 **ensemble_args):

        self.representation = representation
        ## set dim = size of state representation
        dim = representation.size_out
        ## create empty array for action values being updated
        self.update_action = np.zeros(rule.n_actions) 
        ## ensemble
        self.state_neurons = state_neurons
        self.active_prop = active_prop
        self.neuron_type = neuron_type
        
        ## empty arrays for state value and action values
        self.state_value = np.zeros(1)
        self.action_values = np.zeros(rule.n_actions)
        
        ## Create nengo model
        self.model = nengo.Network()
        with self.model:
            
            ## empty array for state
            ## size = size of state representation + number of actions + reward + whether env was reset
            self.state = np.zeros(dim+rule.n_actions+2)  
            ## create nengo node for containing state
            self.state_node = nengo.Node(lambda t: self.state)
            
            ## if we're not using a neuron ensemble to contain the state representation
            if state_neurons == None:
                ## create nengo node for containing the learning rule
                ## input size in is dim (state) + n_actions + reward + env.reset
                self.rule = nengo.Node(rule, size_in=dim+rule.n_actions+2)
                ## connect the state node to the rule node
                nengo.Connection(self.state_node, self.rule, synapse=None)
                
            ## if we are using a neuron ensemble
            else:
                ## create nengo node for containing the learning rule
                ## input size in is state_neurons (state) + n_actions + reward + env.reset
                self.rule = nengo.Node(rule, size_in=self.state_neurons+rule.n_actions+2)                
                ## create ensemble for containing the state representation
                self.ensemble = nengo.Ensemble(n_neurons=state_neurons, dimensions=dim,
                                               neuron_type=self.neuron_type,
                                               intercepts=nengo.dists.Choice([sparsity_to_x_intercept(dim, self.active_prop)]),
                                               **ensemble_args
                                              )
                ##connect the state representation to the ensemble
                nengo.Connection(self.state_node[:dim], self.ensemble, synapse=None)
                ##connect the state ensemble to the rule node
                nengo.Connection(self.ensemble.neurons, self.rule[:state_neurons], synapse=None)
                ##connect the state representation to the rule node
                nengo.Connection(self.state_node[dim:], self.rule[state_neurons:], synapse=None)
            
            ## create node for containing the updated state value
            self.value_node = nengo.Node(self.value_node_func, size_in=1)
            
            ## create node for containing the updated action values
            self.action_values_node = nengo.Node(self.action_values_node_func, size_in=rule.n_actions)
            
            ## send first output from rule node (updated state value) to the state value node
            nengo.Connection(self.rule[0], self.value_node, synapse=None)
            ## send updated action values from rule node to action value node
            nengo.Connection(self.rule[1:], self.action_values_node, synapse=None)
            
        ## run model
        self.sim = nengo.Simulator(self.model)
        
    def step(self, state, update_action, reward, reset=False):
        '''Function for running the model for one time step.
        
        Inputs: agent's state, chosen action, reward
        Outputs: state value, action values'''
        
        if type(update_action)==int or type(update_action)==np.int64 or len(update_action)==1:
            ## set update_action to an array of 0's with one value for each action
            self.update_action[:] = 0
            ## set the update_action value at the position of the chosen action to 1
            self.update_action[update_action] = 1
        else:
            self.update_action = update_action
                    
        ## create state variable containing state representation,
        ## update_action array, reward, and whether or not the env was reset
        self.state[:] = np.concatenate([
            self.representation.map(state),
            self.update_action,
            [reward, reset],]
            )
        
        ## run model for one step
        self.sim.step()
        
        ## return the updated state and action values from the model
        ## these are the values returned by the learning rule
        #w = self.sim.signals[self.sim.model.sig[self.rule.output]['_state_w']]
        return self.state_value, self.action_values#, w[0]

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