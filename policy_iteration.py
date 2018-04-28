import numpy as np
import gym
import gym_lavaland
from Lavaland_spec import Lavaland_spec
import copy


class PI:
    def __init__(self):
        self.gamma = 0.9
        self.error = 0.001
        self.lavaland = Lavaland_spec(10, 10, 4, 4)

    def policy_iteration(self, proxy_reward):

        num_cells = 100  # 10*10 grid
        num_actions = 4

        proxy_reward = proxy_reward.reshape((num_actions, 1))
        cell_type = self.lavaland.form_testing_rewards(proxy_reward)
        rewards = cell_type @ proxy_reward
        state_trans_prob = self.lavaland.get_state_trans_mat()
        values = np.zeros([num_cells])

        # print(state_trans_prob[0][1])
        while True:
            values_tmp = copy.deepcopy(values)
            for s in range(num_cells):
                values[s] = max([sum(
                    [state_trans_prob[s, s1, a] * (rewards[s] + self.gamma * values_tmp[s1]) for s1 in range(num_cells)])
                    for a in range(num_actions)])

            if max([abs(values[s] - values_tmp[s]) for s in range(num_cells)]) < self.error:
                break

        temp = np.reshape(values, (10, 10))
        temp = np.transpose(temp)

        policy = np.zeros([num_cells])
        for s in range(num_cells):
            policy[s] = np.argmax([sum([state_trans_prob[s, s1, a] * (rewards[s] + self.gamma * values[s1])
                                        for s1 in range(num_cells)])
                                   for a in range(num_actions)])
        temp2 = np.reshape(policy, (10, 10))
        temp2 = np.transpose(temp2)
        return policy

if __name__ == '__main__':
    pi = PI()
    policy = pi.policy_iteration(np.array((-2, 7, 3, 0)))
    temp2 = np.reshape(policy, (10, 10))
    temp2 = np.transpose(temp2)
    print(temp2)
