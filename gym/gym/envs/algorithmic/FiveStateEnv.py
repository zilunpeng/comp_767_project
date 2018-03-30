import gym
from gym.spaces import Discrete
import numpy as np
import sys

class FiveStateEnv(gym.Env):

    Transistion = [np.array([0.42, 0.13, 0.14, 0.03, 0.28]),
                   np.array([0.25, 0.09, 0.16, 0.35, 0.15]),
                   np.array([0.08, 0.2, 0.33, 0.17, 0.22]),
                   np.array([0.36, 0.05, 0, 0.52, 0.07]),
                   np.array([0.17, 0.24, 0.19, 0.18, 0.22])]

    Reward = [[104.66, 29.69, 82.36, 37.49, 68.82],
              [75.86, 29.24, 100.37, 0.31, 35.99],
              [57.68, 65.66, 56.95, 100.44, 47.63],
              [96.23, 14.01, 0.88, 89.77, 66.77],
              [70.35, 23.69, 73.41, 70.70, 85.41]]

    reward_scaling_factor = 1

    def __init__(self):
        self.num_states = 5;
        self.reset()

    def step(self, action):
        self.prev_state = self.cur_state
        self.cur_state = np.random.choice(self.num_states, p=FiveStateEnv.Transistion[self.prev_state])
        reward = self.reward_scaling_factor*FiveStateEnv.Reward[self.prev_state][self.cur_state]
        done = False
        self.time = self.time+1
        self.episode_total_reward = self.episode_total_reward+reward

        return {'cur_state':self.cur_state, 'prev_state':self.prev_state}, reward, done, {}


    def reset(self):
        self.cur_state = np.random.randint(5)
        self.time = 0
        self.episode_total_reward = 0
        return self.cur_state

    def render(self, mode='human'):
        outfile = sys.stdout
        inp = "time step: %d\n" % self.time
        outfile.write(inp)
        inp = "at state %d and arrive at state %d\n" %(self.prev_state, self.cur_state)
        outfile.write(inp)
        outfile.write("total reward received so far is %d\n\n" %self.episode_total_reward)