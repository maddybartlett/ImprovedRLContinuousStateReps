import minigrid
from minigrid.wrappers import *
import gymnasium as gym
import matplotlib.pyplot as plt

class location_wrapper(gym.Env):
    '''Wrapper for the mini-grid environment.
    This allows us to access the agent's state in the same way as we would with 
    other gym environments.
    Returns the agent's location and direciton as observation.
    '''
    def __init__(self, env, dist=None, render_mode=None):
        self.dist = dist
        self.orig_env = gym.make(env, render_mode = render_mode) #call minigrid environment
        self.action_space = spaces.Discrete(3) #self.orig_env.action_space #call minigrid actionspace
        self.n_actions = 3
        self.orig_env.reset() #reset environment
        
        self.min_x_pos = 0
        self.max_x_pos = self.orig_env.width-1
        self.min_y_pos = 0
        self.max_y_pos = self.orig_env.height-1
        self.min_dir = 0
        self.max_dir = 3

        self.low = np.array([self.min_x_pos, self.min_y_pos, self.min_dir], dtype=np.float32)
        self.high = np.array([self.max_x_pos, self.max_y_pos, self.max_dir], dtype=np.float32)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
    def render(self):
        #plt.imshow(orig_env.render('rgb_array'))
        return self.orig_env.render()

    def reset(self):
        obs = self.orig_env.reset() #reset environment
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        return pos #return state
    
    def get_goal(self):
        for grid in self.orig_env.grid.grid:
            if grid is not None and grid.type == "goal":
                return(grid.cur_pos)
        
    def step(self, a):
        obs, reward, terminated, truncated, info = self.orig_env.step(a) #do step in environment
        done = terminated or truncated
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
            
        #return state, reward, done and info
        return pos, reward, done, info
    
class visual_wrapper(gym.Env):
    '''Wrapper for the mini-grid environment.
    This allows us to access the agent's state in the same way as we would with 
    other gym environments.
    '''
    def __init__(self, env, dist=None, render_mode=None):
        self.dist = dist
        self.orig_env = gym.make(env, render_mode = render_mode) #call minigrid environment
        self.action_space = spaces.Discrete(3) #self.orig_env.action_space #call minigrid actionspace
        self.n_actions = 3
        self.orig_env.reset() #reset environment
        
        self.min_x_pos = 0
        self.max_x_pos = self.orig_env.width-1
        self.min_y_pos = 0
        self.max_y_pos = self.orig_env.height-1
        self.min_dir = 0
        self.max_dir = 3

        self.low = np.array([self.min_x_pos, self.min_y_pos, self.min_dir], dtype=np.float32)
        self.high = np.array([self.max_x_pos, self.max_y_pos, self.max_dir], dtype=np.float32)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
    def render(self):
        #plt.imshow(orig_env.render('rgb_array'))
        return self.orig_env.render()
                                                            
    def get_obj_info(self, obs):
        '''
        OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,}
        '''
        
        ## Agent's position in visual field
        agent = (3,6)

        ## Depending on env and wrapper, the obs object can take slightly different shapes
        try:
            ## Locate key
            key_pos = np.where(obs['image'][:,:,0]==5)
            ## Locate door
            door_pos = np.where(obs['image'][:,:,0]==4)
            ## Locate goal
            goal_pos = np.where(obs['image'][:,:,0]==8)

        except TypeError:
            ## Locate key
            key_pos = np.where(obs[0]['image'][:,:,0]==5)
            ## Locate door
            door_pos = np.where(obs[0]['image'][:,:,0]==4)
            ## Locate goal
            goal_pos = np.where(obs[0]['image'][:,:,0]==8)


        ## Create dictionaries with object positions and distances to objects
        objs_dict = {'key': key_pos, 'door': door_pos, 'goal': goal_pos}

        ## Calculate distance to each item if visible, add this to dictionary
        objs_in_view = []
        dists = []
        for key in objs_dict:
            if objs_dict[key][0].size is not 0:
                objs_in_view.append(key)
                itemSimple = (objs_dict[key][0][0], objs_dict[key][1][0])
                dists.append(np.asarray(agent) - np.asarray(itemSimple))

        

        try:
            if obs[-1]['has_key']:
                has_key = 1
            else:
                has_key = 0
        except TypeError:
            if obs[0][-1]['has_key']:
                has_key = 1
            else:
                has_key = 0
        except KeyError:
            has_key = 0


        return (objs_in_view, dists, has_key)

    def reset(self):
        obs = self.orig_env.reset() #reset environment
        objs_in_view, dists, has_key = self.get_obj_info(obs)
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        return objs_in_view, dists, has_key, pos #return state
    
    def get_goal(self):
        for grid in self.orig_env.grid.grid:
            if grid is not None and grid.type == "goal":
                return(grid.cur_pos)
        
    def step(self, a):
        obs, reward, terminated, truncated, info = self.orig_env.step(a) #do step in environment
        done = terminated or truncated
        #get agent's state (x position, y position, direction)
        pos = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        
        objs_in_view, dists, has_key = self.get_obj_info(obs)
            
        #return state, reward, done and info
        return objs_in_view, dists, has_key, pos, reward, done, info