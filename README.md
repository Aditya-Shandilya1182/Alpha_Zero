# AlphaZero Implementation 

This project implements the AlphaZero algorithm for playing Tic-Tac-Toe using a Residual Neural Network (ResNet) for state representation and Monte Carlo Tree Search (MCTS) for decision making. The goal is to create an AI that can learn to play Tic-Tac-Toe at a superhuman level through self-play. This project is made for educational purposes.

## Introduction

AlphaZero is a groundbreaking AI algorithm developed by DeepMind that learns to play games through self-play. This project adapts AlphaZero for the game of Tic-Tac-Toe, employing a ResNet architecture to evaluate board states and MCTS to determine optimal moves.

## Features

- Self-play training using the AlphaZero algorithm.
- Residual Neural Network for efficient state evaluation.
- Monte Carlo Tree Search for move selection.
- Ability to play against itself or a human player.

## Architecture

The architecture consists of:

1. **ResNet**: A deep convolutional neural network that processes the board state and predicts the probability of winning for each possible move and the value of the current state.
2. **Monte Carlo Tree Search (MCTS)**: A search algorithm that explores possible future game states and selects the most promising moves based on simulated outcomes.

## Getting Started

### Prerequisites

```bash
pip install torch numpy
```

### Clone the Repository

```bash
git clone https://github.com/Aditya-Shandilya1182/Alpha_Zero
cd Alpha_Zero
```

### Trainig and Evaluation

```bash
python main.py
```
