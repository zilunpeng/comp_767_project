'''
Implementation of the training Lavaland environment
Note that this is the training environment, so there is no lava
Each cell has a fixed type, which is exactly same as Figure 2 (left) of the paper
Proxy rewards are specified by the user
'''

import gym
import numpy as np
from gym.spaces import Discrete

class Simple_training_lavaland(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = Discrete(4)
        self.define_cell_type()

    # Agent has 4 types of actions
    # 0 = Up    1 = Down    2 = Left    3 = Right
    # Position is a tuple where first number = row & second number = column
    # This method performs action and updates total reward received so far. Terminate if reaching the terminal
    def step(self, action):
        if action == 0:
            self.current_pos = (self.current_pos[0]-1, self.current_pos[1])
        elif action == 1:
            self.current_pos[0] = (self.current_pos[0]+1, self.current_pos[1])
        elif action == 2:
            self.current_pos[1] = (self.current_pos[0], self.current_pos[1]-1)
        elif action == 3:
            self.current_pos[1] = (self.current_pos[0], self.current_pos[1]+1)

        cell_type = self.land[self.current_pos]
        self.episode_tot_reward += self.proxy_rewards[cell_type]

        if cell_type == 3:
            done = True
        else:
            done = False

        return done


    def reset(self,proxy_rewards):
        self.proxy_rewards = proxy_rewards
        self.current_pos = (6, 1) # same with what's on the paper
        self.episode_tot_reward = 0

    def render(self, mode='human'):
        return 0

    # define cell types
    # 0 = dirt   1 = grass
    def define_cell_type(self):
        self.land = np.zeros((10,10))
        self.land[0:4, 2:8] = 1
        self.land[4:7, 3:7] = 1
        self.land[7:10, 4:6] = 1
        self.land[5,8] = 3