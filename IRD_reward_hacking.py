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

class IRD_reward_hacking:

    def __init__(self):
        self.num_states = 4
        self.max_step = 15
        self.num_traj = 1
        self.num_proxy_rewards = 1
        self.beta = 1
        self.gamma = 0.9
        self.error = 0.001
        self.lavaland = Lavaland_spec(10, 10, 4, 4)
        self.w_true_expected_phi = None

    def run_ird(self, proxy_weight, W_true):

        # h_pos = horizontal position
        # v_pos = vertical position
        def sample_action_from_stochastic_policy(policy, h_pos, v_pos):
            return np.random.choice(4, 1, p=policy[sub2ind(h_pos, v_pos)])

        def update_counters(h_pos, v_pos, land_type, state_freq, land_type_counter):
            state_freq[sub2ind(h_pos, v_pos)] += 1
            land_type_counter[land_type] += 1
            return state_freq, land_type_counter

        # w = proxy reward
        # max_step = maximum number of steps agent will take if not reaching the terminal
        # num_traj = number of trajectories that we sample
        # RETURN:
        # phi_trajectories: Phi(Epsilon)
        # path_trajectories: the actual path of each trajectory. A path ends before -1
        def generate_trajectory_from_policy(env, policy, deterministic):
            state_freq = np.zeros((100, 1))
            tot_steps = 0
            land_type_counter = np.zeros(4)
            for eps in range(self.num_traj):
                pos = env.reset()
                state_freq, land_type_counter = update_counters(pos[0], pos[1], 0, state_freq, land_type_counter)
                for step in range(self.max_step):
                    if deterministic:
                        action = policy[sub2ind(pos[0], pos[1])]
                    else:
                        action = sample_action_from_stochastic_policy(policy, pos[0], pos[1])
                    done, phi_epsilon, pos, _ = env.step(action)
                    state_freq, land_type_counter = update_counters(pos[0], pos[1], self.lavaland.get_training_land_type(pos[0], pos[1]), state_freq, land_type_counter)
                    tot_steps += 1
                    if done:
                        break
            #state_freq = np.true_divide(state_freq, self.num_traj)
            # land_type_counter = np.true_divide(land_type_counter, self.num_traj)
            return state_freq, land_type_counter

        def calc_Z_approx_bayes_w(expected_Phi, index, w):
                z_w = 0
                remaining_phi = np.delete(expected_Phi, index, axis=0)
                firstTerm = np.dot(w, expected_Phi[index])
                z_w = z_w + np.exp(firstTerm)
                rem = [np.exp(self.beta * np.dot(w, phi_i)) for phi_i in remaining_phi]
                z_w = z_w + sum(rem)
                return z_w

        def get_opposite_action(action):
            if action == 0:
                return 1
            elif action == 1:
                return 0
            elif action == 2:
                return 3
            elif action == 3:
                return 2

            #  Return Nx1 vector - state visitation frequencies
        def compute_state_visition_freq(state_trans_mat, policy, deterministic):
                N_STATES, _, N_ACTIONS = np.shape(state_trans_mat)

                mu = np.zeros([N_STATES, self.max_step])
                mu[sub2ind(5, 1), 0] = 1
                visited_states = [sub2ind(5, 1)]

                for t in range(1, self.max_step):
                    if deterministic:
                        prev_s = np.where(mu[:, t - 1] > 0)[0]
                        (prev_s_rind, prev_s_cind) = ind2sub(prev_s)
                        s = self.lavaland.get_ngbr_pos_coord(prev_s_rind, prev_s_cind, policy[prev_s])
                        if s == -1 or prev_s == 85 or s in visited_states:  # terminal or out of bounds or cell has been visited
                            break
                        else:
                            visited_states.append(s)
                            mu[s, t] += mu[prev_s, t-1]
                    else:
                        for s in range(N_STATES):
                            s_rind, s_cind = ind2sub(s)
                            for a in range(N_ACTIONS):
                                prev_s = self.lavaland.get_ngbr_pos_coord(s_rind, s_cind, a)
                                mu[s, t] += mu[prev_s, t-1] * policy[prev_s, get_opposite_action(a)]
                p = np.sum(mu, 1)
                p[sub2ind(5, 1)] = 1
                return p.reshape((N_STATES, 1))

        def value_iteration(state_trans_prob, rewards, deterministic):
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

            if deterministic:
                # generate deterministic policy
                policy = np.zeros([num_cells])
                for s in range(num_cells):
                    policy[s] = np.argmax([sum([state_trans_prob[s, s1, a] * (rewards[s] + self.gamma * values[s1])
                                                for s1 in range(num_cells)])
                                           for a in range(num_actions)])

                return values, policy
            else:
                # generate stochastic policy
                policy = np.zeros([num_cells, num_actions])
                for s in range(num_cells):
                    v_s = np.array(
                        [sum([state_trans_prob[s, s1, a] * (rewards[s] + self.gamma * values[s1]) for s1 in range(num_cells)]) for a in
                         range(num_actions)])
                    policy[s, :] = np.transpose(v_s / np.sum(v_s))
                return values, policy

        # run_ird code starts from here
        env = gym.make('Simple_training_lavaland-v0')

        w = proxy_weight.reshape((self.num_states, 1))
        cell_type = self.lavaland.form_rewards(w)
        rewards = cell_type @ w
        state_trans_prob = self.lavaland.get_state_trans_mat()
        values, policy = value_iteration(state_trans_prob, rewards, deterministic=True)
        state_freq, land_type_counter = generate_trajectory_from_policy(env, policy, deterministic=True)
        # temp = np.reshape(state_freq, (10,10))
        # temp = np.transpose(temp)
        expected_telda_phi_w = compute_state_visition_freq(state_trans_prob, policy, deterministic=True)
        # temp2 = np.reshape(expected_telda_phi_w, (10, 10))
        # temp2= np.transpose(temp2)
        expected_telda_phi_w = np.multiply(state_freq, expected_telda_phi_w)
        expected_telda_phi_w = np.tile(expected_telda_phi_w, (1, 4))
        expected_telda_phi_w = np.multiply(cell_type, expected_telda_phi_w)
        expected_telda_phi_w = np.sum(expected_telda_phi_w, axis=0)

        expected_true_phi = []  # 25 * 4
        if self.w_true_expected_phi is not None:
            expected_true_phi = self.w_true_expected_phi
        else:
            for w in W_true:
                w = w.reshape((self.num_states, 1))
                cell_type = self.lavaland.form_rewards(w)
                rewards = cell_type @ w
                state_trans_prob = self.lavaland.get_state_trans_mat()
                values, policy = value_iteration(state_trans_prob, rewards, deterministic=True)
                state_freq, land_type_counter = generate_trajectory_from_policy(env, policy, deterministic=True)
                expected_true_phi_w = compute_state_visition_freq(state_trans_prob, policy, deterministic=True)
                expected_true_phi_w = np.multiply(state_freq, expected_true_phi_w)
                expected_true_phi_w = np.tile(expected_true_phi_w, (1, 4))
                expected_true_phi_w = np.multiply(cell_type, expected_true_phi_w)
                expected_true_phi_w = np.sum(expected_true_phi_w, axis=0)
                expected_true_phi.append(expected_true_phi_w)
            self.w_true_expected_phi = expected_true_phi

        # calculate posterior for each possible true_w:
        # input w_telda 1*4, output posterior 1 * 25
        posteriors = []
        store_numerators = []
        store_z = []
        for idx, w_true in enumerate(W_true):
            expected_true_reward = np.dot(expected_telda_phi_w, w_true)
            numerator = np.exp(self.beta * expected_true_reward)
            z_w_true = calc_Z_approx_bayes_w(expected_true_phi, idx, w_true)
            store_numerators.append(numerator)
            store_z.append(z_w_true)
            likelihood = np.true_divide(numerator, z_w_true)
            post = likelihood
            posteriors.append(post)
        posteriors = np.asarray(posteriors).flatten()

        print(posteriors)
        print(posteriors.sum())
        print(np.divide(posteriors, posteriors.sum()))
        print(posteriors.max())

        return posteriors, W_true, expected_telda_phi_w

