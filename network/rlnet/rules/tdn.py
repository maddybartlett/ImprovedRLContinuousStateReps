import nengo
import numpy as np

## TD(n) Learning Rule ##
class TDn(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(n) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, lr = 0.1, act_dis = 0.9, state_dis = 0.95, 
                 n=1, lambd=None, env_dt=None, learnTrials=None):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.lr = lr ##learning rate
        self.act_dis = act_dis ##discount factor for action values
        self.state_dis = state_dis ##discount factor for state values
        self.n = n ##value of n - the number of steps between the current state and the state being updated
        self.env_dt = env_dt ##number of environment steps per simulation step
        self.learnTrials = learnTrials ##number of trials where agent learns
        
        self.step_count = 0
        self.count = -1
        
        if self.env_dt != None:
            self.lr = lr/(self.env_dt/0.001)
            
        ## Input = reward + state representation + action values for current state
        ## Output = action values for current state + state value
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1)  
        
    def make_state(self, shape_in, shape_out, dt, dtype=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the state being updated (update_state_rep),
        the initial value of the state (0) (update_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ## set dim = length of each row in matrix/look-up table 
        ## where a nengo ensemble is used to store the state representation, 
        ## this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        ## return the state dictionary
        return dict(update_state_rep=np.zeros(dim),
                    update_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state):
        '''Create function that advances the process forward one time step.'''
        ## set dim = length of each row in matrix/look-up table 
        ## where a nengo ensemble is used to store the state representation, 
        ## this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        
        state_memory = [] ##list for storing the last n states
        value_memory = [] ##list for storing the last n values
        reward_memory = [] ##list for storing the last n rewards
        action_memory = [] ##list for storing the last n chosen actions
        
        ## One time step
        def step_TDn(t, x, state=state):
            ''' Function for performing TD(n) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state value, updated action values'''
            
            n = int(self.n)
                
            current_state_rep = x[:dim] ##get the state representation of current state
            update_state_rep = state['update_state_rep'] ##get representation of the state being updated
            last_action = x[dim:-2] ##get the action that was taken
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset

            ## get the dot product of the weight matrix and state representation
            ## results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep[:])

            ## get the value of the current state
            current_state_value = result_values[0]

            ## Do the TD(n) update
            ## if the environemt has just been reset, set the count to 0 and empty the memory lists
            if reset:
                self.step_count = 0
                state_memory.clear()
                value_memory.clear()
                reward_memory.clear()
                action_memory.clear()
                
                ## Increase trial count for tracking learning trials
                if self.env_dt != None:
                    self.count += 1/(self.env_dt/0.001)
                else:
                    self.count += 1

            if self.learnTrials is None or self.count < self.learnTrials:
                ## if the environment has NOT just been reset
                if not reset: 
                    ## add the most recent reward and action to the memory lists
                    reward_memory.append(reward)
                    action_memory.append(last_action)

                    ## Start updating after n steps
                    if self.step_count>=n:
                        ## calculate td error term
                        Rs = self.state_dis**np.arange(n)*reward_memory[:]
                        target = np.sum(Rs) + ((self.state_dis**n)*current_state_value)
                        td_error = target - state['update_value']

                        ## calculate a scaling factor
                        ## this scaling factor allows us to switch from updating
                        ## a look-up table to updating weights 
                        scale = np.sum(update_state_rep**2)
                        if scale != 0:
                            scale = 1.0 / scale

                        ## update the state value
                        state['w'][0] += self.lr*td_error*update_state_rep*scale

                        ## update the action values
                        ## multiply the entire state represention by (action value * beta * tderror)
                        ## scale these values and then update the weight matrix/look-up table
                        dw = np.outer(action_memory[0]*self.act_dis*td_error, update_state_rep)
                        state['w'][1:] += dw*scale

                        ## delete the first value in each memory list 
                        state_memory.pop(0)
                        value_memory.pop(0)
                        reward_memory.pop(0)
                        action_memory.pop(0)

            ## increase step count by 1 
            self.step_count+=1
            ## add the most recent state and value to the memories
            state_memory.append(current_state_rep.copy())
            value_memory.append(current_state_value.copy())

            ## calculate the updated value for update state and add it to the result_values array
            result_values[0] = state['w'].dot(state['update_state_rep'][:])[0]
            ## change the state to be updated to the new first value in the state memory   
            state['update_state_rep'][:] = state_memory[0][:]
            ## change the value being updated to the new first value in the value memory   
            state['update_value'] = value_memory[0]

            ## return updated state value for update state and action values for current state
            return result_values 
        return step_TDn