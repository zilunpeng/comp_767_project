import gym
import gym_lavaland
import numpy as np
env = gym.make('Simple_training_lavaland-v0')

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
def generate_trajectory(w, max_step, num_traj, num_states):
    phi_trajectories = np.zeros((num_traj,num_states))
    path_trajectories = np.ones((num_traj,max_step))*-1
    for eps in range(num_traj):
        pos = env.reset(w)
        for step in range(max_step):
            action = sample_action(np.arange(4), pos[0], pos[1])
            done, phi_epsilon, pos = env.step(action)
            path_trajectories[eps, step] = pos
            if done:
                break
                phi_trajectories[eps,:] = phi_epsilon/(step+1) #taking the average so that features are on the same scale
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
    expected_phi = np.multiply(phi_trajectories, traj_prob)
    return sum(expected_phi)

#Note that this is not exactly same as what's in the paper (Equation 3)
def calc_Z_approx_bayes(num_proxy_rewards, phi_trajectories, W, beta):
    Z = 0
    for i in range(num_proxy_rewards):
        w = W[i,:]
        traj_prob_dist = calc_traj_prob(w.reshape((1,num_states)), phi_trajectories.reshape((num_states,num_traj)))
        expected_phi = calc_expected_phi(phi_trajectories, traj_prob_dist)
        Z = Z + np.exp(beta*np.dot(w, expected_phi))
    return Z

if __name__ == "__main__":
    num_states = 4
    max_step = 100
    num_traj = 1000
    num_proxy_rewards = 500
    beta = 1
    phi_trajectories, path_trajectories = generate_trajectory(np.array([1,1,1,1]), max_step, num_traj, num_states)
    W = np.random.uniform(-10,10,(num_proxy_rewards, num_states))
    Z = calc_Z_approx_bayes(num_proxy_rewards, phi_trajectories, W, beta)

    #Calculate the posterior
    num_test_proxy_rewards = 100
    W_test = np.random.uniform(-10,10,(num_test_proxy_rewards,num_states))
    for i in range(num_proxy_rewards):
        w = W[i,:]
