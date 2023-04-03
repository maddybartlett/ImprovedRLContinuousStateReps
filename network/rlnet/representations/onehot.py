import numpy as np

class OneHotRep(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1. 
    This method works with MiniGrid using the provided wrapper.
    '''
    def __init__(self, ranges):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)

    def map(self, state):
        index = 0
        ##step 3
        for i,v in enumerate(state):
            index += v*self.factors[i]
        self.result[:] = 0
        ##step 4
        #print(f'checking index {index}')
        if index%1 == 0:
            index=int(index)
        self.result[index] = 1
        return self.result    
    
    def get_state(self, state, env):       
        discrete_state = state
        
        if env.name == "RatBox":
            state_space_high = np.asarray([env.width, env.height, 360])
            state_space_low = np.asarray([0, 0, 0])
            
            discrete_obs_size = np.array(self.ranges)

            discrete_obs_size = np.array(discrete_obs_size, dtype='int')
            discrete_obs_win_size = (state_space_high - state_space_low)/discrete_obs_size

            discrete_state = (state - state_space_low)/discrete_obs_win_size
            discrete_state = tuple(discrete_state.astype(np.int))
            
        
        return discrete_state