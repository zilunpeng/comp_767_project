import gym
import gym_lavaland
import numpy as np
from Lavaland_spec import Lavaland_spec
import random
import copy

# h_pos = horizontal position
# v_pos = vertical position
TMAX = 2000
ITERMAX = 1000

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
def generate_trajectory(w, max_step, num_traj, num_states, env):
    phi_trajectories = np.zeros((num_traj,num_states))
    path_trajectories = []#np.ones((num_traj,max_step))*-1
    for eps in range(num_traj):
        pos = env.reset(w)
        eps_trajectory = [sub2ind(pos[0], pos[1])]
        for step in range(max_step):
            action = sample_action(np.arange(4), pos[0], pos[1])
            done, phi_epsilon, pos, _ = env.step(action)
            # if eps == 0:
                # print("traj:", eps, " step:", step, " phi_traj:", phi_epsilon)

            # path_trajectories[eps, step] = pos
            eps_trajectory.append(sub2ind(pos[0], pos[1]))
            if done:
                break
        path_trajectories.append(eps_trajectory)
        # print("-------------")
        # print(phi_epsilon)
        # print(np.true_divide(phi_epsilon, (step+1)))
        # print("-------------\n")

        phi_trajectories[eps,:] = np.true_divide(phi_epsilon, (step+1)) #taking the average so that features are on the same scale
        # print("phi_trajectories[{},:] = {}".format(eps, phi_trajectories[eps,:]))
    return phi_trajectories, path_trajectories

# Calculate the distribution over trajectories (Section 4.1 of the paper)
def calc_traj_prob(w, trajectories):
    prob = np.exp(w @ trajectories)
    prob = prob / np.sum(prob)
    return prob

# Calculate expected value of Phi(Epsilon)
# Phi_trajectories = feature vector of each trajectory
# traj_prob = probability of the trajectory
def calc_expected_phi(phi_trajectories, traj_prob):
    sumtrajprob = sum(np.asarray(traj_prob).transpose())
    expected_phi = np.multiply(phi_trajectories, np.transpose(traj_prob))
    return sum(expected_phi)


def calc_Z_approx_bayes_w(expected_Phi, index, w, beta):
    z_w = 0
    remaining_phi = np.delete(expected_Phi, index, axis=0)
    firstTerm = np.dot(w, expected_Phi[index])
    z_w = z_w + np.exp(firstTerm)
    rem = [np.exp(beta*np.dot(w, phi_i)) for phi_i in remaining_phi]
    z_w = z_w + sum(rem)
    return z_w

def value_iteration(state_trans_prob, rewards, gamma, error):
    num_cells = 100 #10*10 grid
    num_actions = 4
    values = np.zeros([num_cells])

    while True:
        values_tmp = copy.deepcopy(values)

        for s in range(num_cells):
            values[s] = values[s] = max([sum([state_trans_prob[s, s1, a]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(num_cells)]) for a in range(num_actions)])

        if max([abs(values[s] - values_tmp[s]) for s in range(num_cells)]) < error:
            break

    policy = np.zeros([num_cells])
    for s in range(num_cells):
        policy[s] = np.argmax([sum([state_trans_prob[s, s1, a]*(rewards[s]+gamma*values[s1])
                              for s1 in range(num_cells)])
                              for a in range(num_actions)])
    return policy

def q_learning(env, proxy_reward, iter_max, t_max, min_lr, init_lr, eps, discount):
    WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
    state_n = WORLD_HEIGHT * WORLD_WIDTH
    action_n = env.getActionCount()

    print('----- using Q Learning -----')
    Q = np.zeros((state_n, action_n))

    for i in range(iter_max):
        pos = env.reset(proxy_reward) # coordinate
        cur_state = sub2ind(pos[0], pos[1])
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, init_lr * (0.85 ** (i // 100)))
        for j in range(t_max):

            if np.random.uniform(0, 1) < eps:
                action = np.random.randint(4)
            else:
                rewardSpace = Q[cur_state, :]
                logits_exp = np.exp(rewardSpace)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(4, p=probs)

            done, phi_epsilon, pos, reward = env.step(action)
            total_reward += reward
            next_state = sub2ind(pos[0], pos[1])
            # update q table
            Q[cur_state, action] += eta * (
                        reward + discount * np.max(Q[next_state, :]) - Q[cur_state, action])
            if done:
                print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
                break
        # if i % 100 == 0:
            # print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))

    solution_policy = np.argmax(Q, axis=1)
    return solution_policy
    # solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    # print("Average score of solution = ", np.mean(solution_policy_scores))

def sub2ind(row_idx, col_idx):
    num_rows = 10
    return num_rows*col_idx + row_idx

def getTraj(policy, w):
    env = gym.make('Simple_testing_lavaland-v0')
    pos = env.reset(w)
    WORLD_HEIGHT, WORLD_WIDTH = env.getDimension()
    total_reward = 0
    optimal_traj = []
    for step_idx in range(TMAX):
        # print("step [{}] at pos [{}, {}]".format(step_idx, pos[0], pos[1]))
        optimal_traj.append(pos)
        cur_state = sub2ind(pos[0], pos[1])
        action = policy[cur_state]
        done, phi_epsilon, pos, reward = env.step(action)
        total_reward += reward
        # total_reward += discount ** step_idx * reward #bellman
        if done:
            optimal_traj.append(pos)
            # print("step [{}] at pos [{}, {}] hit the GOLD with reward [{}]!".format(step_idx, pos[0], pos[1], total_reward))
            break
    return total_reward, phi_epsilon, optimal_traj

if __name__ == "__main__":
    num_states = 4
    max_step = TMAX
    num_traj = 50
    num_proxy_rewards = 1
    beta = 1
    gamma = 0.9
    lavaland = Lavaland_spec(10,10,4,4)
    agent_type = {"VI", "QL", "DQ"}

    # training (proxy)
    env = gym.make('Simple_training_lavaland-v0')
    # proxy_reward = np.array([1,1,1,1])
    proxy_reward = np.array([-0.1, -0.5, 10, 0])
    phi_trajectories, path_trajectories = generate_trajectory(proxy_reward, max_step, num_traj, num_states, env)
    W = np.random.randint(-10,10,(num_proxy_rewards, num_states))

    expected_telda_phi = [] # 1 * 4

    for w in W:
        w = w.reshape((num_states,1))
        # rewards = lavaland.form_rewards(w)
        # rewards = rewards@w
        # state_trans_prob = lavaland.get_state_trans_mat()
        # policy = value_iteration(state_trans_prob, rewards, gamma, error=0.01)
        # policy = q_learning(env, proxy_reward, 100, 1000, 0.003, 1.0, 0.02, gamma)
        traj_prob_dist = calc_traj_prob(w.reshape((1, num_states)), phi_trajectories.reshape((num_states, num_traj)))
        expected_telda_phi_w = calc_expected_phi(phi_trajectories, traj_prob_dist)
        expected_telda_phi.append(expected_telda_phi_w)

    # testing: input 1*4 -> 1*25
    num_true_rewards = 5
    #phi_true_trajectories, path_true_trajectories = generate_trajectory(np.array([1,1,1,1]), max_step, num_traj, num_states, env)
    phi_true_trajectories = phi_trajectories
    W_true = np.random.randint(-10,10,(num_true_rewards, num_states))
    W_true[0] = W[0]

    expected_true_phi = [] # 25 * 4
    for w in W_true:
        traj_prob_dist = calc_traj_prob(w.reshape((1, num_states)), phi_true_trajectories.reshape((num_states, num_traj)))
        expected_true_phi_w = calc_expected_phi(phi_true_trajectories, traj_prob_dist)
        expected_true_phi.append(expected_true_phi_w)

    # calculate posterior for each possible true_w:
    # input w_telda 1*4, output posterior 1 * 25
    priors = np.ones((num_true_rewards, 1))/num_true_rewards
    posteriors = []
    store_z = []
    for idx, w_true in enumerate(W_true):
        expected_true_reward = np.dot(w_true.reshape((1, num_states)), np.asarray(expected_telda_phi).reshape((num_states, num_proxy_rewards)))
        numerator = np.exp(beta * expected_true_reward)
        z_w_true = calc_Z_approx_bayes_w(expected_true_phi, idx, w_true, beta)
        store_z.append(z_w_true)
        likelihood = np.true_divide(numerator, z_w_true)
        post = likelihood
        #post = likelihood * priors[idx]
        posteriors.append(post)
    posteriors = np.asarray(posteriors).flatten()

    print(posteriors)
    print(posteriors.sum())
    print(np.divide(posteriors, posteriors.sum()))
    print(posteriors.max())

    # risk-averse policy: Given those weights, pick a trajectory that maximizes the reward in the worst case
    # [the optimal traj returned by optimal policy by nature maximizes the reward for true w,
    #  so we simply need to find the "worst reward case", and this w_i's traj is the optimal traj wanted]
    # 1. sampled weights from posterior distribution
    traj_list = []
    reward_list = []
    for w_i in W_true:
        env_test = gym.make('Simple_testing_lavaland-v0')
        # q_learning(env, proxy_reward, iter_max, t_max, min_lr, init_lr, eps, discount)
        policy = q_learning(env_test, w_i, ITERMAX, TMAX, 0.003, 1.0, 0.02, gamma)
        total_reward_w_i, phi_epsilon_w_i, traj_w_i = getTraj(policy, w_i)
        reward_list.append(w_i @ phi_epsilon_w_i)
        traj_list.append(traj_w_i)
    min_reward_idx = np.argmin(reward_list)
    chosen_traj = traj_list[min_reward_idx]

    print(chosen_traj)




