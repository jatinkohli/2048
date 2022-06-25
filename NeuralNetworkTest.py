import torch
from torch import nn
import numpy as np
from game import Game

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(0, 1) # 4x4 matrix to 1x16 tensor
        self.linear_relu_stack = nn.Sequential(
            self.randomizeWeights(nn.Linear(16, 8), [-2,2]),
            self.randomizeWeights(nn.Linear(8, 4), [-2,2]),
            nn.ReLU(),
            nn.Softmax()
        )

    def randomizeWeights(self, x, bounds): # bounds = [low, high)
        x.weight = nn.Parameter(data=((bounds[1] - bounds[0]) * torch.rand_like(x.weight, device=device) + bounds[0]))
        return x

    def run(self, x):
        x = torch.as_tensor(x, dtype=torch.float, device=device)
        x = self.flatten(x)
        return self.linear_relu_stack(x)

game = Game()
model = NeuralNetwork().to(device)
result = model.run(game.board)
print(result)