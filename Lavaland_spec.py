import numpy as np

# Define the landscape of lavaland.
class Lavaland_spec:

    max_num_ngbrs = 4
    num_rows = 10
    num_cols = 10

    # # gamma = discount factor
    # # init_prob = |S|*1 vector where each entry denotes the probability of state s being the initial state
    # def __init__(self, gamma, init_prob):
    #     self.gamma = gamma
    #     self.init_prob = init_prob

    # Return the coordinates of a cell's neighbors
    @staticmethod
    def get_ngbr_pos_coord(row_idx, col_idx):
        ngbr_pos_coords = []

        if col_idx > 1:
            ngbr_pos_coords.append(np.array([row_idx, col_idx-1])) #cell to the right of the current cell
        if col_idx < Lavaland_spec.num_cols:
            ngbr_pos_coords.append(np.array([row_idx, col_idx+1])) #cell to the left of the current cell
        if row_idx > 1:
            ngbr_pos_coords.append(np.array([row_idx-1, col_idx])) #cell below the current cell
        if row_idx < Lavaland_spec.num_rows:
            ngbr_pos_coords.append(np.array([row_idx+1, col_idx])) #cell above the current cell
        return ngbr_pos_coords

    # 0 = Up    1 = Down    2 = Left    3 = Right
    @staticmethod
    def get_available_action(row_idx, col_idx):
        available_actions = []

        if col_idx >= 1 and col_idx <= Lavaland_spec.num_cols-1:
            available_actions.append(3)
        if col_idx >= 2 and col_idx <= Lavaland_spec.num_cols:
            available_actions.append(2)
        if row_idx >= 1 and row_idx <= Lavaland_spec.num_rows-1:
            available_actions.append(1)
        if row_idx >= 2 and row_idx <= Lavaland_spec.num_rows:
            available_actions.append(0)
        return available_actions