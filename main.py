import torch
import numpy as np
import matplotlib.pyplot as plt
from game_engine import TicTacToe
from MCTS import MCTS
from resnet import ResNet
from model import AlphaZero

tictactoe = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(tictactoe, 4, 64, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

training_args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

print("Starting AlphaZero training...")
alphaZero = AlphaZero(model, optimizer, tictactoe, training_args)
alphaZero.learn()
print("Training completed.")

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)
state = tictactoe.get_next_state(state, 6, 1)
state = tictactoe.get_next_state(state, 8, 1)

encoded_state = tictactoe.get_encoded_state(state)
tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model.eval()
with torch.no_grad():
    policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print("Predicted Value:", value)
print("Current Game State:\n", state)

plt.bar(range(tictactoe.total_actions), policy)
plt.xlabel("Actions")
plt.ylabel("Policy Probability")
plt.title("Action Policy Distribution")
plt.show()

print("\nStarting a TicTacToe game simulation using MCTS...\n")
player = 1
mcts_args = {'C': 2, 'num_searches': 1000}
mcts = MCTS(tictactoe, mcts_args, model)

state = tictactoe.get_initial_state()

while True:
    print("Current State:\n", state)
    
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("Valid Moves:", [i for i in range(tictactoe.total_actions) if valid_moves[i] == 1])
        action = int(input(f"Player {player}'s Move (Choose an action): "))
        
        if valid_moves[action] == 0:
            print("Invalid move. Try again.")
            continue
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = tictactoe.get_next_state(state, action, player)
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    
    if is_terminal:
        print("Final Game State:\n", state)
        if value == 1:
            print(f"Player {player} wins!")
        else:
            print("It's a draw!")
        break

    player = tictactoe.get_opponent(player)
