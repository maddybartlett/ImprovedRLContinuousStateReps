import numpy as np
from rlnet.sspspace import HexagonalSSPSpace, RandomSSPSpace
import nengo_spa as spa

class VSARep(object):
    '''Create state representation with SSPs and SPs'''
    
    def __init__(self, N, n_scales=8, n_rotates=4, ssp_dim=193, hex=True, length_scale=1):
        if hex:
            self.ssp_space = HexagonalSSPSpace(N,
                                   domain_bounds=None, scale_min=0.5, 
                                   scale_max = 2, n_scales=n_scales, n_rotates=n_rotates,
                                   length_scale=length_scale)
        else:
            self.ssp_space = RandomSSPSpace(N,ssp_dim = ssp_dim,
                                   domain_bounds=None, 
                                    length_scale=length_scale)
        self.size_out = self.ssp_space.ssp_dim*2
        
        self.vocab = spa.Vocabulary(dimensions=self.ssp_space.ssp_dim,
                                    pointer_gen=np.random.RandomState(1))
        obj_names = ['unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'agent']
        obj_type = ['I', 'I', 'I', 'I', 'DOOR', 'KEY', 'BALL', 'BOX', 'GOAL', 'LAVA', 'I']
        sp_names = list(filter(lambda a: a != 'I', obj_type))
        sp_names.append('HAS_KEY')
        #sp_names.append('NO_OBJ')
        
        self.obj_to_sp_map = dict(zip(obj_names, obj_type))
        self.vocab.add('I', self.vocab.algebra.identity_element(self.ssp_space.ssp_dim))

        self.vocab.populate(".normalized();".join(sp_names))
        
    def map(self, state):
        objs_in_view, dists, has_key, pos = state
        ssppos = self.ssp_space.encode(np.array(pos))
        if has_key:
            ssppos=self.ssp_space.bind(ssppos, self.vocab['HAS_KEY'].v)
        objvec = np.zeros(self.ssp_space.ssp_dim)
        if len(objs_in_view) > 0 :
            sspdists = self.ssp_space.encode(np.hstack([np.array(dists), np.zeros((len(objs_in_view),1))]))
            for i, name in enumerate(objs_in_view):
                objvec+= self.ssp_space.bind(sspdists[i,:], self.vocab[self.obj_to_sp_map[name]].v).reshape(-1)
        # else:
        #     objvec = self.vocab['NO_OBJ'].v
        return np.hstack([ssppos.reshape(-1),objvec])
    
    def make_encoders(self, n_neurons):
        return np.vstack([self.ssp_space.sample_grid_encoders(n_neurons),
                          self.ssp_space.sample_grid_encoders(n_neurons)])
    
    def get_state(self, state, env):
        return state

