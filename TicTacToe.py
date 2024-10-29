import numpy as np

class TicTacToe:
    
    def __init__(self):
        self.rows = 3
        self.columns = 3
        self.total_actions = self.rows * self.columns

    def get_initial_state(self):
        return np.zeros((self.rows, self.columns))
    
    def get_next_state(self, action, state, player):
        row = action // self.columns
        column = action % self.columns
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        row = action // self.columns
        column = action % self.columns
        player = state[row, column]

        return(
            np.sum(state[row, :]) == player * self.columns
            or np.sum(state[:, column]) == player * self.rows
            or np.sum(np.diag(state)) == player * self.rows
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.rows
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False