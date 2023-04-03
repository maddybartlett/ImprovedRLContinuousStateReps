import numpy as np
import gymnasium as gym
from rlnet.sspspace import HexagonalSSPSpace, RandomSSPSpace_orig


class SSPRep(object):
    '''Create state representation with grid-cells'''
    
    def __init__(self, N, n_scales=8, n_rotates=4, ssp_dim=256, hex=True, length_scale=1, scale=[1.0, 1.0, 1.0],domain_bounds = None):
        if hex:
            self.ssp_space = HexagonalSSPSpace(N,
                                   domain_bounds=domain_bounds, scale_min=0.5, 
                                   scale_max = 2, n_scales=n_scales, n_rotates=n_rotates,
                                   length_scale=length_scale)
        else:
            self.ssp_space = RandomSSPSpace_orig(N,ssp_dim = ssp_dim,
                                   domain_bounds=None, 
                                    scale=scale)
        self.size_out = self.ssp_space.ssp_dim
       
    
    def map(self, state):   
        ssppos = self.ssp_space.encode(state)
        return ssppos.reshape(-1)
        
    def unmap(self,SSP,num_samples_ = 10):
        state = self.ssp_space.decode(SSP,num_samples = num_samples_,method='from-set')
        return state.reshape(-1)

    def make_encoders(self, n_neurons):
        return self.ssp_space.sample_grid_encoders(n_neurons)
    
    def get_state(self, state, env):
        return state