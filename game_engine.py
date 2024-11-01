import numpy as np

class TicTacToe:
    
    def __init__(self):
        self.rows = 3
        self.columns = 3
        self.total_actions = self.rows * self.columns

    def get_initial_state(self):
        return np.zeros((self.rows, self.columns))
    
    def get_next_state(self, state, action, player):
        row = action // self.columns
        column = action % self.columns
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False

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
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state