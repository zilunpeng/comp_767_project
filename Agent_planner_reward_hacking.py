from Lavaland_spec import Lavaland_spec
import numpy as np
from scipy.optimize import linprog
from IRD_reward_hacking import IRD_reward_hacking as IRD
from baseline import Baseline
from value_iteration import VI

def get_opposite_action(action):
    if action==0:
        return 1
    elif action==1:
        return 0
    elif action==2:
        return 3
    elif action==3:
        return 2

# Return the linear index given a (cell) position-action pair
def pos_action_pair_2_ind(row_ind, col_ind, action):
    return (num_rows*(col_ind) + row_ind)*num_actions + action

def sub2ind(row_idx, col_idx):
    return num_rows * col_idx + row_idx

def ind2sub(ind):
    return (int(ind % num_cols), int(ind / num_cols))

def form_bounds():
    bounds = []
    for i in range(num_cells*num_actions):
        bounds.append((0, None))
    bounds.append((None, None))
    return bounds

def form_ineq_vec(expected_telda_phi_w):
    vec = np.zeros((num_sampled_w, 1))
    for w_ind in range(len(sampled_w)):
        vec[w_ind] = -np.dot(sampled_w[w_ind], expected_telda_phi_w)
    return vec

def form_ineq_mat():
    linprog_ineq_mat = np.zeros((num_sampled_w, num_cells*num_actions + 1))
    ind = 0
    for w in sampled_w:
        # print(w)
        w = np.asarray(w).reshape((4,1))
        cell_type = lavaland.form_testing_rewards(w)
        rewards = cell_type @ w
        for r_ind in range(num_rows):
            for c_ind in range(num_cols):
                available_actions = lavaland.get_available_action(r_ind, c_ind)
                for a_ind in range(len(available_actions)):
                    pa_idx = pos_action_pair_2_ind(r_ind, c_ind, available_actions[a_ind])
                    ngbr_pos = lavaland.get_ngbr_pos_coord(r_ind, c_ind, available_actions[a_ind])
                    reward = rewards[ngbr_pos]
                    linprog_ineq_mat[ind, pa_idx] = -1*reward
        linprog_ineq_mat[ind, -1] = 1
        ind = ind + 1
    # initial_state_pos = pos_action_pair_2_ind(5,1,0)
    # linprog_ineq_mat[0:num_sampled_w, initial_state_pos:initial_state_pos+num_actions] = 0
    return linprog_ineq_mat

def form_eq_vec():
    linprog_eq_vec = np.zeros((num_cells,1))
    linprog_eq_vec[sub2ind(5,1)] = 1 #(5,1) is the initial position
    return linprog_eq_vec


# Form the equality matrix (LHS) (|S|+1 * |S||A|+1) to the linear programming solver
def form_eq_mat():
    linprog_eq_mat = np.zeros((num_cells, num_cells*num_actions + 1))
    for r_ind in range(num_rows):
        for c_ind in range(num_cols):
            p_idx = sub2ind(r_ind, c_ind)
            available_actions = lavaland.get_available_action(r_ind, c_ind)
            for a_ind in range(len(available_actions)):
                pa_idx = pos_action_pair_2_ind(r_ind, c_ind, available_actions[a_ind])
                linprog_eq_mat[p_idx, pa_idx] = 1

                ngbr_pos_coord = lavaland.get_ngbr_pos_coord(r_ind, c_ind, available_actions[a_ind])
                ngbr_pos_coord = ind2sub(ngbr_pos_coord)
                pa_idx = pos_action_pair_2_ind(ngbr_pos_coord[0], ngbr_pos_coord[1], get_opposite_action(available_actions[a_ind]))
                linprog_eq_mat[p_idx, pa_idx] = -gamma
    return linprog_eq_mat


'''
convert a |S|*|S|*|A| vector into policy
return -1 if there is no action with > 0 probability
'''
def convert2policy(x):
    x = x['x']
    policy = np.zeros((num_rows, num_cols))
    for r_ind in range(num_rows):
        for c_ind in range(num_cols):
            idx = pos_action_pair_2_ind(r_ind, c_ind, 0)
            action_prob = x[idx:idx+num_actions]
            if np.sum(action_prob) == 0:
                policy[r_ind, c_ind] = -1
            else:
                action_prob = np.divide(action_prob, np.sum(action_prob))
                policy[r_ind, c_ind] = np.argmax(action_prob)
    return policy


def policy_leads_to_lava(lavaland, policy):
    position = 51
    # pos_x, pos_y = ind2sub(position)
    pos_x, pos_y = num_to_coord(position)
    for _ in range(100): # max traj length = 100
        action = policy[pos_x][pos_y]
        if action == -1:
            action = np.random.randint(4)
        pos_x, pos_y = lavaland.get_next_state(pos_x, pos_y, action)
        terrain = lavaland.get_testing_land_type(pos_x, pos_y)
        if terrain == 3: #hit laba
            return True
    return False

def num_to_coord(num):
    r, c = num // 10, num % 10
    return (r, c)


def coord_To_Num(coord):
    r, c = coord[0], coord[1]
    return r * 10 + c  # num from 0 to 99


def baseline_policy_leads_to_lava(lavaland, policy):
    position = 51
    # pos_x, pos_y = ind2sub(position)
    pos_x, pos_y = num_to_coord(position)
    for _ in range(100): # max traj length = 100
        action = policy[pos_x][pos_y]
        pos_x, pos_y = lavaland.get_next_state(pos_x, pos_y, action)
        if pos_x < 0:
            return False
        terrain = lavaland.get_testing_land_type(pos_x, pos_y)
        if terrain == 3: #hit laba
            return True
    return False

if __name__ == "__main__":

    # baseline_agent = VI()
    # baseline_policy = baseline_agent.value_iteration(np.array((-2, 7, 3, 0)))
    # temp_baseline_policy = np.reshape(baseline_policy, (10, 10))
    # temp_baseline_policy = np.transpose(temp_baseline_policy)
    # print("--------baseline policy--------")
    # print(temp_baseline_policy)
    # lavaland = Lavaland_spec(10, 10, 4, 4)
    # hit_lava = baseline_policy_leads_to_lava(lavaland, temp_baseline_policy)
    # print(hit_lava)

    # sampled_w = [np.array((0.1, -10, 10, 0)),np.array((0.1, -10, 10, -5)), np.array((0.1, -10, 10, 10)),np.array((0.1, -10, 10, -10)),] #just for testing
    # sampled_w = [np.array((0.1, 0.1, 10, -10))]
    # sampled_w = [np.array((1, -5, 5, 0))]

    hit_lava_baseline_policy = []

    hit_lava_proxy_w_list = []
    hit_lava_sampled_w_list = []
    hit_lava_policy_list = []
    experiment_num = 100

    w_true = np.random.randint(-10, 10, (50, 4))
    ird = IRD()

    for _ in range(experiment_num):
        design_weight = np.array(np.random.randint(-10, 10, (1, 4))).flatten()
        #design_weight = np.array((1, 0, 10, 0))
        # design_weight[3] = 0

        print("using proxy weight: ", design_weight)

        posterior, true_W, expected_telda_phi_w = ird.run_ird(design_weight, w_true)

        # sample few candidate true_weight from posterior

        #print(true_W)
        num = true_W.shape[0]
        # true_W.reshape((num, 4))
        # sample_space = true_W.tolist()
        # print(sample_space)
        num_sampled_w = 7
        pos = np.divide(posterior, posterior.sum())
        sampled_w_indices = np.random.choice(num, num_sampled_w, p=pos)
        sampled_w = true_W[sampled_w_indices].tolist()

        lavaland = Lavaland_spec(10, 10, 4, 4)
        num_rows = 10
        num_cols = 10
        num_cells = num_rows * num_cols
        num_actions = 4
        gamma = 0.9
        linprog_eq_mat = form_eq_mat()
        linprog_eq_vec = form_eq_vec()
        linprog_ineq_mat = form_ineq_mat()
        linprog_ineq_vec = form_ineq_vec(expected_telda_phi_w)

        c = np.zeros(num_cells*num_actions+1)
        c[-1] = -1

        bounds = form_bounds()


        x = linprog(c, A_ub=linprog_ineq_mat, b_ub=linprog_ineq_vec, A_eq=linprog_eq_mat, b_eq=linprog_eq_vec, bounds=bounds, options={"disp": True})
        if x.success == False:
            continue

        policy = convert2policy(x)
        print(policy)
        # print(x)

        if policy_leads_to_lava(lavaland, policy):
            hit_lava_proxy_w_list.append(design_weight)
            hit_lava_sampled_w_list.append(sampled_w)
            hit_lava_policy_list.append(policy)
            print("IRD hit lava, [{}]/[{}]".format(len(hit_lava_policy_list), experiment_num))

        # start to run baseline
        # baseline_agent = Baseline()
        # baseline_policy = baseline_agent.agent_learn(design_weight)
        # if policy_leads_to_lava(lavaland, baseline_policy):
        #     hit_lava_baseline_policy.append(baseline_policy)

        baseline_agent = VI()
        baseline_policy = baseline_agent.value_iteration(design_weight)
        temp_baseline_policy = np.reshape(baseline_policy, (10, 10))
        temp_baseline_policy = np.transpose(temp_baseline_policy)
        print("--------baseline policy--------")
        print(temp_baseline_policy)
        if baseline_policy_leads_to_lava(lavaland, temp_baseline_policy):
            hit_lava_baseline_policy.append(temp_baseline_policy)
            print("VI hit lava, [{}]/[{}]".format(len(hit_lava_baseline_policy), experiment_num))

    ratio_hit_traj = len(hit_lava_policy_list)/experiment_num
    ratio_hit_traj_baseline = len(hit_lava_baseline_policy)/experiment_num
    print(ratio_hit_traj, ratio_hit_traj_baseline)
    # print(ratio_hit_traj)
    print("-------------the end-------------")

    # file = open(“output.txt”, ”w”)
    # hit_lava_baseline_policy = []
    # hit_lava_proxy_w_list = []
    # hit_lava_sampled_w_list = []
    # hit_lava_policy_list = []
    #
    # file.write(hit_lava_baseline_policy)
    # file.write(hit_lava_sampled_w_list)
    # file.write(hit_lava_policy_list)
    # file.write(hit_lava_proxy_w_list)
    # file.close()
