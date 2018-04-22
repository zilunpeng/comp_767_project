import numpy as np

# Define the landscape of lavaland.
class Lavaland_spec:

    num_features = 4

    # # gamma = discount factor
    # # init_prob = |S|*1 vector where each entry denotes the probability of state s being the initial state
    def __init__(self, num_rows, num_cols, max_num_ngbrs):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_num_ngbrs = max_num_ngbrs

    # Return the coordinates of a cell's neighbors
    def get_ngbr_pos_coord(self, row_idx, col_idx, action):
        if col_idx < self.num_cols-1 and action == 3:
            return self.sub2ind(row_idx, col_idx+1) #cell to the right of the current cell
        elif col_idx > 0 and action == 2:
            return self.sub2ind(row_idx, col_idx-1) #cell to the left of the current cell
        elif row_idx > 0 and action == 0:
            return self.sub2ind(row_idx-1, col_idx) #cell below the current cell
        elif row_idx < self.num_rows-1 and action == 1:
            return self.sub2ind(row_idx+1, col_idx) #cell above the current cell

    # 0 = Up    1 = Down    2 = Left    3 = Right
    def get_available_action(self, row_idx, col_idx):
        available_actions = []

        if col_idx >= 1 and col_idx <= self.num_cols-1:
            available_actions.append(3)
        if col_idx >= 2 and col_idx <= self.num_cols:
            available_actions.append(2)
        if row_idx >= 1 and row_idx <= self.num_rows-1:
            available_actions.append(1)
        if row_idx >= 2 and row_idx <= self.num_rows:
            available_actions.append(0)
        return available_actions

    # state_trans_mat is a |S|*|S|*|A| matrix
    # state_trans_mat[s0, s1, a] is the probability of landing at s1 when taking action a in state s0
    def get_state_trans_mat(self):
        n_states = self.num_rows * self.num_cols
        n_actions = self.max_num_ngbrs
        state_trans_mat = np.zeros((n_states, n_states, n_actions))
        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_cols):
                state0_idx = self.sub2ind(self, row_idx, col_idx)
                for action in self.get_available_action(row_idx, col_idx):
                    state1_idx = self.get_ngbr_pos_coord(row_idx, col_idx, action)
                    state_trans_mat[state0_idx, state1_idx, action] = 1

    def sub2ind(self, row_idx, col_idx):
        return self.num_rows*col_idx + row_idx

    @staticmethod
    def get_testing_land_type(row, col):
        if row>=0 and row<=4 and col>=2 and col<=8:
            return 1
        if row>=4 and row<=7 and col>=3 and col<=7:
            return 3
        if row>=7 and row<=10 and col>=4 and col<=6:
            return 1
        if row==5 and col==8:
            return 2
        return 0

    @staticmethod
    def get_training_land_type(row, col):
        if row>=0 and row<=4 and col>=2 and col<=8:
            return 1
        if row>=7 and row<=10 and col>=4 and col<=6:
            return 1
        if row==5 and col==8:
            return 2
        return 0



    def get_num_rows(self):
        return 10

    @staticmethod
    def get_num_cols():
        return 10

    @staticmethod
    def get_num_features():
        return 4