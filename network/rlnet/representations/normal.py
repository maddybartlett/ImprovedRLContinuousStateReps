import numpy as np

class NormalRep(object):
    '''Create look-up tables of size: 
    (size_state_dim_1, size_state_dim_2, ..., size_state_dim_n).
    
    Inputs: agent's state
    Params: environment
    Outputs: representation of agent's state
    
    Examples of Usage:
    Initialize the representation:
        >> env = gym.make('MountainCar-v0')
        >> representation = NormalRep(env)
                    
    Translate agent state into representation:
        >> state = (0,7,3)  
        >> representation.map(state) 
        ans = array([-1.  ,  0.75,  0.5 ])
    '''
    def __init__(self, env):
        ##Set environment
        self.env = env
            
        self.upper = self.env.observation_space.high
        self.lower = self.env.observation_space.low
        
        self.ranges = self.upper - self.lower
        self.size_out = len(self.ranges) 
        
    ## Normalize the state into values between -1 and 1   
    def map(self, state):
        state = state + abs(self.lower)
        norm_state = (state/self.ranges-0.5)*2
        
        #Check that the state has been normalised properly
        if np.any(norm_state > 1) or np.any(norm_state < -1) :
            print ("Representation Error: State values outside of normalisation bounds (-1, 1): ", state)
        
        return norm_state
    
    def get_state(self, state, env):
        discrete_state = state            
        return discrete_state