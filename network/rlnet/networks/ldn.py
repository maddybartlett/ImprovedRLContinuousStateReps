import nengo
import numpy as np
import scipy.special
from scipy.special import legendre
import scipy.integrate as integrate

## Define a nengo.Process that implements an LDN.  This can be placed inside a
##  nengo.Node
class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q              # number of internal state dimensions per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs
        
        # Do Aaron's math to generate the matrices
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        Q = np.arange(q, dtype=np.float64)
        R = (2*Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)
    
        self.A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        self.B = (-1.)**Q[:, None] * R

        super().__init__(default_size_in=size_in, default_size_out=q*size_in)
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state=np.zeros((self.q, self.size_in))
        
        # Handle the fact that we're discretizing the time step
        #  see https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A*dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)
        
        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None,:])
            return state.T.flatten()
        return step_legendre

    def get_weights_for_delays(self, r):
        '''compute the weights needed to extract the value at time r
        from the network (r=0 is right now, r=1 is theta seconds ago)'''
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q,-1).T
