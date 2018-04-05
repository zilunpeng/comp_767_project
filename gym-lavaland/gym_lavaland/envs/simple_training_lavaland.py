'''
Implementation of the training Lavaland environment
Note that this is the training environment, so there is no lava
Each cell has a fixed type, which is exactly same as Figure 2 (left) of the paper
Proxy rewards are specified by the user
'''

import gym
import numpy as np
from gym.spaces import Discrete
import sys

class Simple_training_lavaland(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.define_cell_type()
        self.proxy_rewards = 1
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))


        return {'cur_state':self.cur_state, 'prev_state':self.prev_state}, reward, done, {}


    def reset(self):
        return 0

    def render(self, mode='human'):
        return 0

    # define cell types
    # 0 = dirt   1 = grass
    def define_cell_type(self):
        self.land = np.zeros((10,10))
        self.land[0:3, 2:7] = 1
        self.land[4:6, 3:6] = 1
        self.land[7:9, 4:5] = 1