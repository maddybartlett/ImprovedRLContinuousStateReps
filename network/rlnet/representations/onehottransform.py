import numpy as np

class OneHotRepTransformND(object):
    '''Create one-hot representation. 
    I.e. the state is represented as a list of 0's and a 1. 
    This method works with MiniGrid using the provided wrapper.
    '''
    
    def __init__(self, n_bins, env):

        self.n_bins = n_bins
        self.state_size = len(env.observation_space.low)
        
        self.result = np.zeros( (n_bins,self.state_size) )
        self.size_out = len(self.result)
        
        self.obs_low = env.observation_space.low
        self.discrete_step_size = ( env.observation_space.high - env.observation_space.low ) / self.size_out

    def map(self, state):
        '''
        
        '''        

        for i,v in enumerate(state):
            
            index = int( ( v - self.obs_low[i]) / self.discrete_step_size[i] )

            self.result[:,i] = 0
            self.result[index,i] = 1

        return self.result

    def get_state(self, state):   
        '''
        Recovers the value of the closest bin to the given state.
        '''
        
        discrete_state = np.zeros_like(state)
        for i,v in enumerate(state):
            dv = v - ( ( v - self.obs_low[i] ) % self.discrete_step_size[i] - self.discrete_step_size[i]/2) 
            discrete_state[i] = dv
        
        return discrete_state

class OneHotRepTransform1D(object):
    '''Create one-hot representation. 
    I.e. the state is represented as a list of 0's and a 1. 
    This method works with MiniGrid using the provided wrapper.
    '''
    def __init__(self, n_bins, env):
    
        self.n_bins = n_bins
        self.factors = np.array([np.prod(n_bins[x+1:]) for x in range(len(n_bins))], dtype=int)
        self.result = np.zeros(np.prod(n_bins))
        self.size_out = len(self.result)
        
        self.obs_low = env.observation_space.low
        self.discrete_step_size = ( env.observation_space.high - env.observation_space.low ) / self.size_out
                
        self.states = np.linspace(env.observation_space.low,env.observation_space.high,self.size_out)

    def map(self, state):
        '''
        
        '''        

        index = 0
        for i,v in enumerate(state):
            
            index += int( ( v - self.obs_low) * self.factors[i] / self.discrete_step_size )
        
        self.result[:] = 0
        self.result[index] = 1
        
        return self.result

    def get_state(self, state):   
        '''
        Recovers the value of the closest bin to the given state.
        '''
        
        discrete_state = state - ( ( state - self.obs_low ) % self.discrete_step_size - self.discrete_step_size/2) 
        
        return discrete_state