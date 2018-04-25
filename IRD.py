import gym
import gym_lavaland
import numpy as np
from Lavaland_spec import Lavaland_spec
import random
import copy


def sub2ind(row_idx, col_idx):
    num_rows = 10
    # return num_rows * row_idx + col_idx
    return num_rows * col_idx + row_idx


def ind2sub(ind):
    num_cols = 10
    return (int(ind % num_cols), int(ind / num_cols))

class IRD:

    def __init__(self):
        self.num_states = 4
        self.max_step = 100
        self.num_traj = 1000
        self.num_proxy_rewards = 1
        self.beta = 1
        self.gamma = 0.9
        self.error = 0.001
        self.lavaland = Lavaland_spec(10, 10, 4, 4)

    #
    # # Calculate the distribution over trajectories (Section 4.1 of the paper)
    # def calc_traj_prob(self, w, trajectories):
    #     prob = np.exp(w @ trajectories)
    #     prob = prob / np.sum(prob)
    #     return prob
    #
    # # Calculate expected value of Phi(Epsilon)
    # # Phi_trajectories = feature vector of each trajectory
    # # traj_prob = probability of the trajectory
    # def calc_expected_phi(self, phi_trajectories, traj_prob):
    #     sumtrajprob = sum(np.asarray(traj_prob).transpose())
    #     expected_phi = np.multiply(phi_trajectories, np.transpose(traj_prob))
    #     return sum(expected_phi)
    #
    def run_ird(self, proxy_weight):

        # h_pos = horizontal position
        # v_pos = vertical position
        def sample_action(action_space, h_pos, v_pos):
            deleted_action = []
            if h_pos == 0:
                deleted_action.append(0)
            if h_pos == 9:
                deleted_action.append(1)
            if v_pos == 0:
                deleted_action.append(2)
            if v_pos == 9:
                deleted_action.append(3)

            action_space = np.delete(action_space, deleted_action)
            return np.random.choice(action_space, 1)

        # w = proxy reward
        # max_step = maximum number of steps agent will take if not reaching the terminal
        # num_traj = number of trajectories that we sample
        # RETURN:
        # phi_trajectories: Phi(Epsilon)
        # path_trajectories: the actual path of each trajectory. A path ends before -1
        def generate_trajectory(env):
            phi_trajectories = np.zeros((self.num_traj, self.num_states))
            path_trajectories = []  # np.ones((num_traj,max_step))*-1
            state_freq = np.zeros((100, 1))
            # tot_steps = 0
            for eps in range(self.num_traj):
                pos = env.reset()
                pos_idx = sub2ind(pos[0], pos[1])
                eps_trajectory = [pos_idx]
                state_freq[pos_idx] += 1
                for step in range(self.max_step):
                    action = sample_action(np.arange(4), pos[0], pos[1])
                    done, phi_epsilon, pos, _ = env.step(action)
                    pos_idx = sub2ind(pos[0], pos[1])
                    eps_trajectory.append(pos_idx)
                    state_freq[pos_idx] += 1
                    # tot_steps += 1
                    if done:
                        break
                path_trajectories.append(eps_trajectory)
                phi_trajectories[eps, :] = np.true_divide(phi_epsilon, (
                            step + 1))  # taking the average so that features are on the same scale
                # print("phi_trajectories[{},:] = {}".format(eps, phi_trajectories[eps,:]))
            # state_freq = np.true_divide(state_freq, tot_steps)
            return phi_trajectories, path_trajectories, state_freq

        def calc_Z_approx_bayes_w(expected_Phi, index, w):
                z_w = 0
                remaining_phi = np.delete(expected_Phi, index, axis=0)
                firstTerm = np.dot(w, expected_Phi[index])
                z_w = z_w + np.exp(firstTerm)
                rem = [np.exp(self.beta * np.dot(w, phi_i)) for phi_i in remaining_phi]
                z_w = z_w + sum(rem)
                return z_w

            #  Return Nx1 vector - state visitation frequencies
        def compute_state_visition_freq(state_trans_mat, policy):
                temp2 = np.reshape(policy, (10, 10))
                temp2 = np.transpose(temp2)

                N_STATES, _, N_ACTIONS = np.shape(state_trans_mat)

                mu = np.zeros([N_STATES, self.max_step])
                mu[sub2ind(5, 1), 0] = 1
                visited_states = [sub2ind(5, 1)]

                for t in range(1, self.max_step):
                    prev_s = np.where(mu[:, t - 1] > 0)[0]
                    (prev_s_rind, prev_s_cind) = ind2sub(prev_s)
                    s = self.lavaland.get_ngbr_pos_coord(prev_s_rind, prev_s_cind, policy[prev_s])
                    if s == -1 or prev_s == 85 or s in visited_states:  # terminal or out of bounds
                        break
                    else:
                        mu[s, t] += mu[prev_s, t - 1]
                    visited_states.append(s)
                p = np.sum(mu, 1)
                return p.reshape((N_STATES, 1))

        def value_iteration(state_trans_prob, rewards):
            num_cells = 100  # 10*10 grid
            num_actions = 4
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

        # run_ird code starts from here
        env = gym.make('Simple_training_lavaland-v0')
        phi_trajectories, path_trajectories, state_freq = generate_trajectory(env)
        state_freq = state_freq / self.num_traj
        # W = np.random.randint(-10, 10, (num_proxy_rewards, num_states))

        expected_telda_phi = []  # 1 * 4
        # W[0] = np.array((0.1, -10, 10, 0))
        # W[0] = np.array((1, -5, 5, 0))
        # W[0] = np.array((0.1, -0.2, 1, 0))
        # for w in W:
        w = proxy_weight.reshape((self.num_states, 1))
        cell_type = self.lavaland.form_rewards(w)
        rewards = cell_type @ w
        temp2 = np.reshape(rewards, (10, 10))
        temp2 = np.transpose(temp2)
        state_trans_prob = self.lavaland.get_state_trans_mat()
        policy = value_iteration(state_trans_prob, rewards)
        temp2 = np.reshape(policy, (10, 10))
        temp2 = np.transpose(temp2)
        expected_telda_phi_w = compute_state_visition_freq(state_trans_prob, policy)
        temp = np.reshape(expected_telda_phi_w, (10, 10))
        temp = np.transpose(temp)
        expected_telda_phi_w = np.multiply(expected_telda_phi_w, state_freq)
        expected_telda_phi_w = np.tile(expected_telda_phi_w, (1, 4))
        expected_telda_phi_w = np.multiply(cell_type, expected_telda_phi_w)
        expected_telda_phi_w = np.sum(expected_telda_phi_w, axis=0)
        expected_telda_phi.append(expected_telda_phi_w)

        # testing: input 1*4 -> 1*25
        num_true_rewards = 20
        # phi_true_trajectories, path_true_trajectories = generate_trajectory(np.array([1,1,1,1]), max_step, num_traj, num_states, env)
        phi_true_trajectories = phi_trajectories
        W_true = np.random.randint(-5, 5, (num_true_rewards, self.num_states))
        # W_true[0] = W[0]

        expected_true_phi = []  # 25 * 4
        for w in W_true:
            w = w.reshape((self.num_states, 1))
            cell_type = self.lavaland.form_rewards(w)
            rewards = cell_type @ w
            state_trans_prob = self.lavaland.get_state_trans_mat()
            policy = value_iteration(state_trans_prob, rewards)
            temp2 = np.reshape(policy, (10, 10))
            temp2 = np.transpose(temp2)
            expected_true_phi_w = compute_state_visition_freq(state_trans_prob, policy)
            temp = np.reshape(expected_true_phi_w, (10, 10))
            temp = np.transpose(temp)
            expected_true_phi_w = np.multiply(expected_true_phi_w, state_freq)
            expected_true_phi_w = np.tile(expected_true_phi_w, (1, 4))
            expected_true_phi_w = np.multiply(cell_type, expected_true_phi_w)
            expected_true_phi_w = np.sum(expected_true_phi_w, axis=0)
            expected_true_phi.append(expected_true_phi_w)

        # calculate posterior for each possible true_w:
        # input w_telda 1*4, output posterior 1 * 25
        posteriors = []
        store_z = []
        for idx, w_true in enumerate(W_true):
            expected_true_reward = np.dot(expected_telda_phi_w, w_true)
            numerator = np.exp(self.beta * expected_true_reward)
            z_w_true = calc_Z_approx_bayes_w(expected_true_phi, idx, w_true)
            store_z.append(z_w_true)
            likelihood = np.true_divide(numerator, z_w_true)
            post = likelihood
            # post = likelihood * priors[idx]
            posteriors.append(post)
        posteriors = np.asarray(posteriors).flatten()

        print(posteriors)
        print(posteriors.sum())
        print(np.divide(posteriors, posteriors.sum()))
        print(posteriors.max())

        return posteriors, W_true


# if __name__ == "__main__":
#     design_weight = np.array((1, -5, 5, 0))
#     ird = IRD()
#     posterior, _ = ird.run_ird(design_weight)
#     print("--------------------IRD MAIN END--------------------")

    # training (proxy)
    # env = gym.make('Simple_training_lavaland-v0')
    # phi_trajectories, path_trajectories, state_freq = generate_trajectory(max_step, num_traj, num_states, env)
    # state_freq = state_freq/num_traj
    # W = np.random.randint(-10,10,(num_proxy_rewards, num_states))
    #
    # expected_telda_phi = [] # 1 * 4
    # #W[0] = np.array((0.1, -10, 10, 0))
    # W[0] = np.array((1, -5, 5, 0))
    # #W[0] = np.array((0.1, -0.2, 1, 0))
    # for w in W:
    #     w = w.reshape((num_states,1))
    #     cell_type = lavaland.form_rewards(w)
    #     rewards = cell_type@w
    #     temp2 = np.reshape(rewards, (10,10))
    #     temp2 = np.transpose(temp2)
    #     state_trans_prob = lavaland.get_state_trans_mat()
    #     policy = value_iteration(state_trans_prob, rewards, gamma, error=0.01)
    #     temp2 = np.reshape(policy, (10,10))
    #     temp2 = np.transpose(temp2)
    #     expected_telda_phi_w = compute_state_visition_freq(state_trans_prob, gamma, path_trajectories, policy)
    #     temp = np.reshape(expected_telda_phi_w, (10,10))
    #     temp = np.transpose(temp)
    #     expected_telda_phi_w = np.multiply(expected_telda_phi_w, state_freq)
    #     expected_telda_phi_w = np.tile(expected_telda_phi_w, (1,4))
    #     expected_telda_phi_w = np.multiply(cell_type, expected_telda_phi_w)
    #     expected_telda_phi_w = np.sum(expected_telda_phi_w, axis=0)
    #     expected_telda_phi.append(expected_telda_phi_w)
    #
    # # testing: input 1*4 -> 1*25
    # num_true_rewards = 20
    # #phi_true_trajectories, path_true_trajectories = generate_trajectory(np.array([1,1,1,1]), max_step, num_traj, num_states, env)
    # phi_true_trajectories = phi_trajectories
    # W_true = np.random.randint(-5,5,(num_true_rewards, num_states))
    # #W_true[0] = W[0]
    #
    # expected_true_phi = [] # 25 * 4
    # for w in W_true:
    #     w = w.reshape((num_states, 1))
    #     cell_type = lavaland.form_rewards(w)
    #     rewards = cell_type @ w
    #     state_trans_prob = lavaland.get_state_trans_mat()
    #     policy = value_iteration(state_trans_prob, rewards, gamma, error=0.01)
    #     temp2 = np.reshape(policy, (10, 10))
    #     temp2 = np.transpose(temp2)
    #     expected_true_phi_w = compute_state_visition_freq(state_trans_prob, gamma, path_trajectories, policy)
    #     temp = np.reshape(expected_true_phi_w, (10, 10))
    #     temp = np.transpose(temp)
    #     expected_true_phi_w = np.multiply(expected_true_phi_w, state_freq)
    #     expected_true_phi_w = np.tile(expected_true_phi_w, (1, 4))
    #     expected_true_phi_w = np.multiply(cell_type, expected_true_phi_w)
    #     expected_true_phi_w = np.sum(expected_true_phi_w, axis=0)
    #     expected_true_phi.append(expected_true_phi_w)
    #
    # # calculate posterior for each possible true_w:
    # # input w_telda 1*4, output posterior 1 * 25
    # priors = np.ones((num_true_rewards, 1))/num_true_rewards
    # posteriors = []
    # store_z = []
    # for idx, w_true in enumerate(W_true):
    #     expected_true_reward = np.dot(expected_telda_phi_w , w_true)
    #     numerator = np.exp(beta * expected_true_reward)
    #     z_w_true = calc_Z_approx_bayes_w(expected_true_phi, idx, w_true, beta)
    #     store_z.append(z_w_true)
    #     likelihood = np.true_divide(numerator, z_w_true)
    #     post = likelihood
    #     #post = likelihood * priors[idx]
    #     posteriors.append(post)
    # posteriors = np.asarray(posteriors).flatten()
    #
    # print(posteriors)
    # print(posteriors.sum())
    # print(np.divide(posteriors, posteriors.sum()))
    # print(posteriors.max())

