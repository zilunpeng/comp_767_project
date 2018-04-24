from Lavaland_spec import Lavaland_spec
import numpy as np
from scipy.optimize import linprog

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

def form_ineq_vec():
    return np.zeros((num_sampled_w, 1))

def form_ineq_mat():
    linprog_ineq_mat = np.zeros((num_sampled_w, num_cells*num_actions + 1))
    ind = 0
    for w in sampled_w:
        w = w.reshape((4,1))
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

#convert a |S|*|S|*|A| vector into policy
#return -1 if there is no action with > 0 probability
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


if __name__ == "__main__":
    lavaland = Lavaland_spec(10, 10, 4, 4)
    sampled_w = [np.array((5, -10, 10, 0)),np.array((5, -10, 10, -5)), np.array((5, -10, 10, 10)),np.array((5, -10, 10, -10)),] #just for testing
    #sampled_w = [np.array((0.1, -10, 10, 0))]
    num_sampled_w = len(sampled_w)
    num_rows = 10
    num_cols = 10
    num_cells = num_rows * num_cols
    num_actions = 4
    gamma = 0.9
    linprog_eq_mat = form_eq_mat()
    linprog_eq_vec = form_eq_vec()
    linprog_ineq_mat = form_ineq_mat()
    linprog_ineq_vec = form_ineq_vec()

    c = np.zeros(num_cells*num_actions+1)
    c[-1] = -1

    bounds = form_bounds()

    x = linprog(c, A_ub=linprog_ineq_mat, b_ub=linprog_ineq_vec, A_eq=linprog_eq_mat, b_eq=linprog_eq_vec, bounds=bounds, options={"disp": True})
    policy = convert2policy(x)
    print(x)