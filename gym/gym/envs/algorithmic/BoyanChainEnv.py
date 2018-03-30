import gym
from gym.spaces import Discrete
import sys

class BoyanChainEnv(gym.Env):

    def __init__(self):
        self.num_states = 13;
        self.action_space = Discrete(2);
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = action+1
        self.action = action
        self.prev_state = self.cur_state
        self.cur_state = self.prev_state-action
        if self.cur_state == 0:
            reward = -2
            done = True
        else:
            reward = -3
            done = False
        self.episode_total_reward = self.episode_total_reward+reward

        return {'cur_state':self.cur_state, 'prev_state':self.prev_state}, reward, done, {}


    def reset(self):
        self.cur_state = 12
        self.time = 0
        self.episode_total_reward = 0

    def render(self, mode='human'):
        outfile = sys.stdout
        inp = "time step: %d\n" % self.time
        outfile.write(inp)
        inp = "took action %d at state %d and arrive at state %d\n" %(self.action, self.prev_state, self.cur_state)
        outfile.write(inp)
        outfile.write("total reward received so far is %d\n\n" %self.episode_total_reward)