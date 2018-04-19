from Lavaland_spec import Lavaland_spec as Lavaland
import numpy as np

class Agent_planner(object):

    num_rows = 10
    num_cols = 10
    num_actions = 4

    def __init__(self, sampled_w, init_prob):
        self.num_cells = self.num_rows*self.num_cols
        self.sampled_w = sampled_w #assume sampled_w is an array of vectors
        self.init_prob = init_prob
        self.linprog_eq_mat = np.zeros((self.num_rows+1, self.num_cells+1))
        self.linprog_eq_vec = np.zeros(self.num_cells)
        self.linprog_ineq_mat = np.zeros(len(sampled_w))
        self.linprog_ineq_vec = np.ones(len(sampled_w))*self.num_cells


    # Return the linear index given a (cell) position-action pair
    def pos_action_pair_2_ind(self, row_ind, col_ind, action):
        linear_idx = (self.num_cols*(col_ind-1) + row_ind-1)*self.num_actions + action
        return linear_idx

    def pos_2_ind(self, row_ind, col_ind):
        linear_idx = self.num_cols*(col_ind-1) + row_ind
        return linear_idx

    def form_ineq_mat(self):
        for w_ind in range(1, len(self.sampled_w)):
            for r_ind in range(1,self.num_rows+1):
                for c_ind in range(1,self.num_cols+1):
                    p_idx = self.pos_2_ind(r_ind, c_ind)
                    land_type = Lavaland.get_land_type(r_ind, c_ind)
                    if land_type == w_ind:
                        self.linprog_ineq_mat[w_ind][p_idx] = self.linprog_ineq_mat[w_ind][p_idx] - self.sampled_w[w_ind]


    def form_eq_vec(self):
        for r_ind in range(1,self.num_rows+1):
            for c_ind in range(1,self.num_cols+1):
                p_idx = self.pos_2_ind(r_ind, c_ind)
                self.linprog_eq_vec[p_idx] = self.init_prob[r_ind][c_ind]


    # Form the equality matrix (LHS) (|S|+1 * |S||A|+1) to the linear programming solver
    def form_eq_mat(self, gamma):
        for r_ind in range(1,self.num_rows+1):
            for c_ind in range(1,self.num_cols+1):
                p_idx = self.pos_2_ind(r_ind, c_ind)
                available_actions = Lavaland.get_available_action(r_ind, c_ind)
                for a_ind in range(len(available_actions)):
                    pa_idx = self.pos_action_pair_2_ind(r_ind, c_ind, available_actions[a_ind])
                    self.linprog_eq_mat[p_idx][pa_idx] = 1

                ngbr_pos_coords = Lavaland.get_ngbr_pos_coord(r_ind, c_ind)
                for n_ind in range(len(ngbr_pos_coords)):
                    ngbr_pos_coord = ngbr_pos_coords[n_ind]
                    ngbr_row_idx = ngbr_pos_coord[0]
                    ngbr_col_idx = ngbr_pos_coord[1]
                    available_actions = Lavaland.get_available_action(ngbr_row_idx, ngbr_col_idx)
                    for a_ind in range(len(available_actions)):
                        pa_idx = self.pos_action_pair_2_ind(ngbr_row_idx, ngbr_col_idx, available_actions[a_ind])
                        self.linprog_eq_mat[p_idx][pa_idx] = -gamma




if __name__ == "__main__":
    Agent_planner(np.array((1,1)), np.array((1,1))).form_eq_mat(np.array((1,1)))
