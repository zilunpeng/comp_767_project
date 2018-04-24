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
        self.action_n = 4
        self.define_cell_type()

    # Agent has 4 types of actions
    # 0 = Up    1 = Down    2 = Left    3 = Right
    # Position is a tuple where first number = row & second number = column
    # This method performs action and updates total reward received so far. Terminate if reaching the terminal
    def step(self, action):

        def validatePos(pos): #return validated position
            if pos[0] >= self.land.shape[0]:
                pos = (pos[0]-1, pos[1])
            if pos[0] < 0:
                pos = (pos[0]+1, pos[1])
            if pos[1] >= self.land.shape[1]:
                pos = (pos[0], pos[1]-1)
            if pos[1] < 0:
                pos = (pos[0], pos[1]+1)
            return pos

        if action == 0:
            self.current_pos = validatePos((self.current_pos[0]-1, self.current_pos[1]))
        elif action == 1:
            self.current_pos = validatePos((self.current_pos[0]+1, self.current_pos[1]))
        elif action == 2:
            self.current_pos = validatePos((self.current_pos[0], self.current_pos[1]-1))
        elif action == 3:
            self.current_pos = validatePos((self.current_pos[0], self.current_pos[1]+1))

        # print("DEBUG: -11 index out of size 10: ", self.current_pos)
        cell_type = self.land[self.current_pos]
        cell_type = int(cell_type)
        self.traj_feature[cell_type] += 1
        # immediateReward = self.proxy_rewards[cell_type]
        # self.episode_tot_reward += immediateReward
        if cell_type == 2:
            done = True
        else:
            done = False

        # positionDict = {'x': self.current_pos[0], 'y': self.current_pos[1]}
        # self.printWorld()
        return done, self.traj_feature, self.current_pos, None


    def reset(self):
        # self.proxy_rewards = proxy_rewards
        self.current_pos = (5, 1) # same with what's on the paper
        self.episode_tot_reward = 0
        num_states = 4 # dirt, grass, terminal, lava(implicit)

        self.traj_feature = np.zeros(num_states)
        # print("traj_feature reset to: ", self.traj_feature, end='\r')
        return self.current_pos

    def render(self, mode='human'):
        return 0

    def changeEnv(self):
        self.land[4:7, 3:7] = 3

    def printWorld(self, scr):
        robotloc = 'X'
        for r, row in enumerate(self.land):
            row_new = []
            for c, place in enumerate(row):
                if self.current_pos[0] == r and self.current_pos[1] == c:
                    place = robotloc
                else:
                    if place == 0:
                        place = ' '
                    else:
                        place = int(place)
                place = str(place)
                row_new.append(place)
            # print(str(row_new))
            scr.addstr(r+1, 0, str(row_new))
        # return scr, r
        # scr.refresh()
        # time.sleep(0.1)
        return scr, r+1

    # define cell types
    # 0 = dirt   1 = grass  2 = terminal  3 = lava
    def define_cell_type(self):
        self.land = np.zeros((10,10))
        self.land[0:4, 2:8] = 1
        self.land[4:7, 3:7] = 1
        self.land[7:10, 4:6] = 1
        self.land[5,8] = 2

    def getDimension(self):
        return self.land.shape[0], self.land.shape[0]

    def getActionCount(self):
        return self.action_n
