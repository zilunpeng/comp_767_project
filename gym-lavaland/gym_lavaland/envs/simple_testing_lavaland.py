# with lava
import gym
import numpy as np
from gym.spaces import Discrete

class Simple_testing_lavaland(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.define_cell_type()

    # Agent has 4 types of actions
    # 0 = Up    1 = Down    2 = Left    3 = Right
    # Position is a tuple where first number = row & second number = column
    # This method performs action and updates total reward received so far. Terminate if reaching the terminal
    def step(self, action):
        if action == 0:
            self.current_pos = (self.current_pos[0]-1, self.current_pos[1])
        elif action == 1:
            self.current_pos = (self.current_pos[0]+1, self.current_pos[1])
        elif action == 2:
            self.current_pos = (self.current_pos[0], self.current_pos[1]-1)
        elif action == 3:
            self.current_pos = (self.current_pos[0], self.current_pos[1]+1)

        cell_type = self.land[self.current_pos]
        cell_type = int(cell_type)
        self.traj_feature[cell_type] += 1
        # self.episode_tot_reward += self.proxy_rewards[cell_type]

        if cell_type == 2:
            done = True
        else:
            done = False

        # positionDict = {'x': self.current_pos[0], 'y': self.current_pos[1]}
        return done, self.traj_feature, self.current_pos


    def reset(self):
        # self.proxy_rewards = proxy_rewards
        self.current_pos = (5, 1) # same with what's on the paper
        # self.episode_tot_reward = 0
        num_states = 4 # dirt, grass, terminal, lava(implicit)
        self.traj_feature = np.zeros(num_states)
        return self.current_pos

    def render(self, mode='human'):
        return 0

    # define cell types
    # 0 = dirt 1 = grass 2 = terminal 3 = lava
    def define_cell_type(self):
        self.land = np.zeros((10,10))
        self.land[0:4, 2:8] = 1
        self.land[4:7, 3:7] = 3
        self.land[7:10, 4:6] = 1
        self.land[5,8] = 2
